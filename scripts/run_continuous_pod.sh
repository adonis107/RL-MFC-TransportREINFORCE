#!/usr/bin/env bash
# Continuous-state rerun, reusing the recorded finite-state results.
#
# The continuous estimator changed, so the linear-quadratic and portfolio benchmarks are rerun;
# the finite-state benchmarks are unchanged and are seeded from a previous run so that the
# figures and tables are produced over the full suite.
#
#   WORKERS=<vCPU> ./scripts/run_continuous_pod.sh
#
# Resume is on by default: rerunning skips any job whose summary.json already exists, so this is
# safe to relaunch after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_ROOT="${RESULTS_ROOT:-results}"
PREVIOUS="${PREVIOUS:-previous_results}"

mkdir -p "${RESULTS_ROOT}"
for env in twostate cybersecurity distribution advertising; do
    if [[ ! -d "${PREVIOUS}/${env}" ]]; then
        echo "WARNING: ${PREVIOUS}/${env} is missing; the finite-state half will be incomplete."
        continue
    fi
    if [[ -d "${RESULTS_ROOT}/${env}" ]]; then
        echo "keeping existing ${RESULTS_ROOT}/${env}"
    else
        cp -r "${PREVIOUS}/${env}" "${RESULTS_ROOT}/${env}"
        echo "seeded ${env} from ${PREVIOUS}"
    fi
done
echo

ENVS="lq portfolio" exec ./scripts/run_suite.sh "$@"
