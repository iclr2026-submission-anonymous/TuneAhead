#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
python -m src.train --config configs/main.yaml
