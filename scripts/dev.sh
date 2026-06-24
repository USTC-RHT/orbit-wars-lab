#!/usr/bin/env bash
# Uruchamia backend FastAPI (:8000) + Vite dev (:6001) jednocześnie.
# Ctrl+C zabija oba procesy.
#
# Usage:
#   bash scripts/dev.sh                        # use .venv (default)
#   bash scripts/dev.sh --conda orbit-wars-lab # use a conda environment
set -e

cd "$(dirname "$0")/.."

BACKEND_PORT="${BACKEND_PORT:-8000}"
VIEWER_PORT="${VIEWER_PORT:-6001}"

# --- environment activation ---
CONDA_ENV_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -n "$CONDA_ENV_NAME" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda")"
    CONDA_ENV_PATH="$CONDA_BASE/envs/$CONDA_ENV_NAME"
    if [[ ! -d "$CONDA_ENV_PATH" ]]; then
        echo "Conda environment '$CONDA_ENV_NAME' not found at $CONDA_ENV_PATH"
        exit 1
    fi
    echo "Activating conda env: $CONDA_ENV_NAME"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
else
    if [ ! -d ".venv" ]; then
        echo "Run 'make setup' first to create .venv (or use --conda <env>)"
        exit 1
    fi
    echo "Activating .venv"
    source .venv/bin/activate
fi

trap 'kill 0' SIGINT SIGTERM

uvicorn orbit_wars_app.main:app --reload --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

(cd viewer && pnpm dev --port "$VIEWER_PORT" --strictPort) &
FRONTEND_PID=$!

echo ""
echo "================================"
echo " Backend : http://localhost:$BACKEND_PORT"
echo " Viewer  : http://localhost:$VIEWER_PORT"
echo " Ctrl+C  : stop both"
echo "================================"

wait $BACKEND_PID $FRONTEND_PID
