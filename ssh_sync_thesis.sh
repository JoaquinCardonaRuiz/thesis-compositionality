#!/usr/bin/env bash
set -e

LOCAL_DIR="/mnt/c/Users/joaqu/Documents/Code/Thesis/thesis-compositionality"
REMOTE="dl24035@fs0.das5.cs.vu.nl:/var/scratch/dl24035/Thesis/"

rsync -av --info=progress2 --exclude '.git/'  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.mdl' --exclude '*.creds' --exclude 'LEAR/*/checkpoint/logs/test/**.tsv' --include 'LEAR/*/cogs_data/*.tsv' --include 'LEAR/*/slog_data/*.tsv' --include 'LEAR/*/DeAR_experiment/cogs_data/*.tsv' --include 'LEAR/*/DeAR_experiment/slog_data/*.tsv' --exclude '*.tsv' -e "ssh -o ProxyJump=jca100@ssh.data.vu.nl"  "${LOCAL_DIR}/"  "${REMOTE}"
