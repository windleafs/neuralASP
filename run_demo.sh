#!/usr/bin/env bash
# One-command demo: unit tests -> staged training -> evaluation with figures.
# Uses the py310 conda environment's python if available.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
command -v "$PY" >/dev/null || PY=/home/zhuangyang/miniconda3/envs/py310/bin/python

echo "== 1/3 unit tests =="
"$PY" -m pytest tests -q

echo "== 2/3 staged training (m -> eta -> joint) =="
"$PY" train.py --stage all --out demo

echo "== 3/3 evaluation =="
"$PY" eval.py --ckpt runs/demo/joint.pt --out runs/demo/eval

echo "Done. See runs/demo/eval for figures and metrics.json"
