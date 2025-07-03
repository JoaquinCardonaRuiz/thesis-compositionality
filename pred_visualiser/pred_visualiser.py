from flask import Flask, render_template, jsonify, send_from_directory
import os
import csv
import ast
import re
import boto3
import botocore.exceptions
import threading                         # tiny in-memory cache (optional)


app = Flask(__name__)

TSV_DIR = './input_tsvs'

_cache = {
    "streams": {},
    "events": {}
}

def _cw_client():
    """Return a boto3 CloudWatch Logs client that honours the aws.creds switch."""
    creds_path = os.path.join(os.getcwd(), "aws.creds")
    if os.path.isfile(creds_path):                       # ← local laptop
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = creds_path
        sess = boto3.Session(profile_name="thesis_logger")
    else:                                                # ← remote cluster
        sess = boto3.Session(profile_name="default")
    return sess.client("logs")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/files')
def list_files():
    files = [f for f in os.listdir(TSV_DIR) if f.endswith('.tsv')]
    return jsonify(files)

def clean_model_objects(tsv_str):
    return re.sub(r'<model\.(\w+) object at [^>]+>', r"'\1'", tsv_str)

def build_tree(row_dict):
    span2label = {tuple(span): label for span, label in row_dict['span2output_token']}
    span2sem = {tuple(eval(k)): v for k, v in row_dict['span2semantic'].items()}
    sentence_words = row_dict['pair'][0].split()

    def get_tokens_for_span(span):
        start, end = span
        return [[sentence_words[i], i] for i in range(start, end + 1)]

    span_dict = {}
    for span in set(list(span2label.keys()) + list(span2sem.keys())):
        tokens = get_tokens_for_span(span)
        label = span2label.get(span, '')
        sem = span2sem.get(span, '?')
        name = ' '.join([w for w, _ in tokens]) if tokens else label
        subs = [str(i) for _, i in tokens]
        span_dict[span] = {
            "tokens": tokens,
            "semantic": sem,
            "name": name,
            "subs": subs
        }

    for parent, children in row_dict['parent_child_spans']:
        parent_t = tuple(parent)
        if parent_t not in span_dict:
            tokens = get_tokens_for_span(parent_t)
            sem = span2sem.get(parent_t, '?')
            name = ' '.join([w for w, _ in tokens])
            span_dict[parent_t] = {
                "tokens": tokens,
                "semantic": sem,
                "name": name,
                "subs": [str(i) for _, i in tokens]
            }
        span_dict[parent_t]['children'] = [span_dict[tuple(c)] for c in children if tuple(c) in span_dict]

    root_span = tuple(row_dict['end_span'])
    return span_dict.get(root_span, {"name": f"Missing root span {root_span}"})

@app.route('/api/file/<filename>')
def load_file(filename):
    print(f"Loading file: {filename}")
    path = os.path.join(TSV_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({'error': 'File not found'}), 404

    response = []
    with open(path, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            try:
                cleaned = clean_model_objects(row[0])
                parsed_row = ast.literal_eval(cleaned)
                # Base response fields
                rec = {
                    "sentence": parsed_row.get("pair", [""])[0],
                    "category": parsed_row.get("category"),
                    "full_json": parsed_row
                }
                if "pred_edges" in parsed_row:
                    # New format: front-end will draw edges directly
                    rec.update({
                        "gold": parsed_row.get("pair", ["", ""])[1],
                        "correct": parsed_row.get("comp reward", 0.0) == 1.0,
                        # leave these empty so old renderers won’t run
                        "tree": {},
                        "sem_tree": {},
                    })
                else:
                    # Old format: build span-trees as before
                    tree = build_tree(parsed_row)
                    tree["span2output_token"] = parsed_row.get("span2output_token", [])
                    rec.update({
                        "gold": parsed_row.get("label_chain"),
                        "correct": parsed_row.get("pred_chain") == parsed_row.get("label_chain"),
                        "tree": tree,
                        "sem_tree": parsed_row.get("parent_json"),
                        "output": parsed_row.get("pred_chain"),
                    })
                response.append(rec)

            except Exception as e:
                response.append({
                    "sentence": "<Parse Error>",
                    "gold": "<Parse Error>",
                    "correct": False,
                    "tree": {"name": f"Top-level error: {str(e)}"},
                    "sem_tree": {"name": f"Top-level error: {str(e)}"},
                    "category": "Parse Error",
                    "output": "<Parse Error>",
                    "gold": "<Parse Error>",
                    "full_json": {"error": str(e)},
                })

    return jsonify(response)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route('/api/cw/<path:log_group>/streams')
def list_log_streams(log_group):
    """
    Return every stream in *log_group* (newest first).

    JSON schema:
        [ { "logStreamName": str,
            "logStreamDisplayName": str,
            "logStreamUid": str,
            "firstEventTimestamp": int,
            "lastEventTimestamp":  int,
            "storedBytes": int } , … ]
    """
    try:
        if log_group not in _cache["streams"]:
            client = _cw_client()
            paginator = client.get_paginator('describe_log_streams')
            streams = []
            for page in paginator.paginate(
                    logGroupName=log_group,
                    orderBy='LastEventTime',
                    descending=True):
                for stream in page.get('logStreams', []):
                    full_name = stream.get("logStreamName", "")
                    if '-' in full_name:
                        els = full_name.split('-')
                        uid = els[-1]
                        display_name = '-'.join(els[:-1])
                    else:
                        display_name, uid = full_name, "(no UID)"
                    stream["logStreamDisplayName"] = display_name
                    stream["logStreamUid"] = uid
                    streams.append(stream)
            _cache["streams"][log_group] = streams  # memoise
        return jsonify(_cache["streams"][log_group])
    except botocore.exceptions.ClientError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cw/<log_group>/<log_stream>/events')
def stream_events(log_group, log_stream):
    try:
        client = _cw_client()
        events = []
        next_token = None

        while True:
            kwargs = {
                "logGroupName": log_group,
                "logStreamName": log_stream,
                "startFromHead": True
            }
            if next_token is not None:
                kwargs["nextToken"] = next_token

            resp = client.get_log_events(**kwargs)
            events.extend(resp.get('events', []))
            if next_token == resp.get('nextForwardToken'):
                break
            next_token = resp.get('nextForwardToken')

        return jsonify(events)
    except botocore.exceptions.ClientError as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
