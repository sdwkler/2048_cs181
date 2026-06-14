#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pypy_env"
MODE="full"
SEED="181"
WORKERS=""
OUTPUT_DIR="models/eval_results"
INSTALL_DEPS="0"

usage() {
  cat <<'EOF'
Run all phase-1 2048 experiments with a Conda PyPy environment.

Usage:
  ./run_all_experiments.sh [options]

Options:
  --env NAME          Conda environment name. Default: pypy_env
  --mode MODE         Experiment mode: smoke or full. Default: full
  --seed N            Base random seed. Default: 181
  --workers N         Worker process count. Default: script entry defaults
  --output-dir DIR    Result directory. Default: models/eval_results
  --install-deps      Run pip install -r requirement.txt inside the env first
  -h, --help          Show this help message

Examples:
  ./run_all_experiments.sh --mode smoke --workers 1
  ./run_all_experiments.sh --mode full --workers 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "--mode must be smoke or full, got: $MODE" >&2
  exit 2
fi

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH." >&2
  exit 1
fi

PY=(conda run -n "$ENV_NAME" python)
COMMON_ARGS=(--mode "$MODE" --seed "$SEED" --output-dir "$OUTPUT_DIR")
if [[ -n "$WORKERS" ]]; then
  COMMON_ARGS+=(--workers "$WORKERS")
fi

echo "==> Checking Python in Conda env: $ENV_NAME"
"${PY[@]}" --version

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "==> Installing dependencies from requirement.txt"
  "${PY[@]}" -m pip install -r requirement.txt
fi

if [[ ! -f models/2048_state.bin || ! -f models/2048_afterstate.bin ]]; then
  echo "Missing N-tuple weights under models/." >&2
  echo "Expected models/2048_state.bin and models/2048_afterstate.bin." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "==> Running group 1: search experiments 1-A through 1-G plus 1-Regret"
"${PY[@]}" -m src.phase_1.search.run_search "${COMMON_ARGS[@]}"

echo "==> Running group 2: MCTS planning experiments 2-A through 2-H"
"${PY[@]}" -m src.phase_1.planning.run_planning "${COMMON_ARGS[@]}"

echo "==> Running group 3: Q-learning experiments 3-A through 3-D"
"${PY[@]}" -m src.phase_1.learning.run_qlearning "${COMMON_ARGS[@]}"

echo "==> Collecting latest phase-1 result files"
"${PY[@]}" -m src.phase_1.report.collect_results --output-dir "$OUTPUT_DIR"

echo "==> Done. Results are in: $OUTPUT_DIR"
