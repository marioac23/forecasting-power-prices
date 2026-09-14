#!/bin/bash
# Wrapper for workdir/run_pipeline.py
# Usage:
#   ./run.sh history
#   ./run.sh history future forecast
#   ./run.sh backtest
#   ./run.sh all                     # rebuild history, build future, run model, check backtest
#   ./run.sh history backtest        # rebuild history, then check backtest, skip future/forecast
#
# Valid stage names: history, future, forecast, backtest, all

set -e  # stop immediately if any step fails, instead of continuing on a broken state

if [ "$#" -eq 0 ]; then
    echo "Usage: ./run.sh <stage> [<stage> ...]"
    echo "Valid stages: history, future, forecast, backtest, all"
    exit 1
fi

RUN_PIPELINE=workdir/runpipeline.py

if [ ! -f "$RUN_PIPELINE" ]; then
    echo "Error: run_pipeline.py not found at $RUN_PIPELINE"
    echo "This script expects a workdir/ subfolder next to it, containing run_pipeline.py."
    exit 1
fi

# translate plain stage names into run_pipeline.py's --flags
FLAGS=()
for arg in "$@"; do
    case "$arg" in
        history)  FLAGS+=(--history) ;;
        future)   FLAGS+=(--future) ;;
        forecast) FLAGS+=(--forecast) ;;
        backtest) FLAGS+=(--backtest) ;;
        all)      FLAGS+=(--all) ;;
        *)
            echo "Error: unknown stage '$arg'"
            echo "Valid stages: history, future, forecast, backtest, all"
            exit 1
            ;;
    esac
done

echo "Running from: $WORKDIR"
echo "Stages: $*"
echo

cd workdir
python3 runpipeline.py "${FLAGS[@]}"