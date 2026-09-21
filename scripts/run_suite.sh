#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-7}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
ENVS=(portfolio lq advertising cybersecurity twostate distribution)

DETECTED="$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null \
            || nproc --all 2>/dev/null || echo 1)"
CORES="${CORES:-${DETECTED}}"

if [[ -z "${CORES_CONFIRMED:-}" && "${DETECTED}" -gt 32 ]]; then
    echo "WARNING: the container sees ${DETECTED} CPUs, which is almost certainly the"
    echo "  host's count rather than your allocation. Set CORES to the number of vCPU you"
    echo "  were actually given, e.g. CORES=16 $0"
    echo "  Set CORES_CONFIRMED=1 to silence this."
    echo
fi
if [[ "${CORES}" -lt "${WORKERS}" ]]; then
    echo "WARNING: CORES=${CORES} is below WORKERS=${WORKERS}; each worker gets 1 thread"
    echo "  and the box will be oversubscribed ${WORKERS}:${CORES}."
    echo
fi

THREADS=$(( CORES / WORKERS ))
(( THREADS < 1 )) && THREADS=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

mkdir -p "${RESULTS_ROOT}"
echo "cores=${CORES} (detected ${DETECTED}) workers=${WORKERS}"
echo "threads/worker=${THREADS} -> $(( WORKERS * THREADS )) of ${CORES} cores in use"
echo "results=${RESULTS_ROOT}"
echo "order: ${ENVS[*]}"
echo

for env in "${ENVS[@]}"; do
    echo "================ ${env} ================"
    started=$(date +%s)
    uv run python scripts/parallel_train.py \
        --env "${env}" \
        --device cpu \
        --budget-mode fair \
        --workers "${WORKERS}" \
        --results-root "${RESULTS_ROOT}" \
        --logs-root "${RESULTS_ROOT}/logs" \
        "$@"
    echo "${env} finished in $(( ($(date +%s) - started) / 60 )) min"
    echo
done

echo "================ figures and tables ================"
uv run python scripts/make_outputs.py --results-root "${RESULTS_ROOT}" \
    --output-root outputs
