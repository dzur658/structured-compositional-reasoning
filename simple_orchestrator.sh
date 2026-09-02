#!/bin/bash

# activate uv enviornment
source .venv/bin/activate

# run lcsqa evals
uv run python -m lcsqa.run_direct_prompting

uv run python -m lcsqa.run_structured_inference

# run lsata evals
uv run python -m lsata.run_direct_prompting

uv run python -m lsata.run_structured_inference