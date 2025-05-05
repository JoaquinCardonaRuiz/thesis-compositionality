from flask import Flask, render_template, jsonify, send_from_directory
import os
import csv
import ast
import re

app = Flask(__name__)

TSV_DIR = './input_tsvs'

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
                tree = build_tree(parsed_row)
                tree["span2output_token"] = parsed_row.get("span2output_token", [])
                response.append({
                    "sentence": parsed_row.get("pair", [""])[0],
                    "gold": parsed_row.get("pair", [""])[1],
                    "correct": parsed_row.get("pred_chain") == parsed_row.get("label_chain"),
                    "tree": tree,
                    "sem_tree": parsed_row.get("parent_json"),
                    "category": parsed_row.get("category"),
                    "output": parsed_row.get("pred_chain"),
                    "gold": parsed_row.get("label_chain"),
                })
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
                })

    return jsonify(response)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
