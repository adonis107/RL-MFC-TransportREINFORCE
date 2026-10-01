#!/usr/bin/env bash
# Regenerate the diagnostic CSVs that make_outputs.py reads from results/figures/.
# They do not depend on the training runs, so this can run beside run_suite.sh.
#   THREADS=2 scripts/run_diagnostics.sh
set -euo pipefail

cd "$(dirname "$0")/.."

THREADS="${THREADS:-2}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
LOGS=results/logs/diagnostics
mkdir -p "$LOGS"

run() {
    local name="$1"; shift
    echo "start $name"
    if uv run python "$@" > "$LOGS/$name.log" 2>&1; then echo "done  $name"; else echo "FAILED $name (see $LOGS/$name.log)"; fi
}

run theory_estimate     scripts/verify_theory.py --part v1 --output-root results/figures/theory &
run theory_consistency  scripts/verify_theory.py --part v2 --paths 400 --output-root results/figures/theory_400 &
run bounds_continuous   scripts/verify_bounds.py --env all --sweep all &
run perturbation_scales scripts/tune_perturbation.py --env all --force &
for env in twostate cybersecurity distribution advertising; do
    run "discrete_eta_$env" scripts/verify_discrete_eta.py --env "$env" --compare-batches \
        --output "results/figures/discrete_eta/discrete_eta_$env.csv" &
    run "bounds_$env" scripts/verify_discrete_bounds.py --env "$env" \
        --output "results/figures/bounds/bounds_discrete_$env.csv" &
done
wait

# make_outputs.py reads the merged eta table.
uv run python - <<'PY'
import glob
import pandas as pd
files = sorted(glob.glob("results/figures/discrete_eta/discrete_eta_*.csv"))
pd.concat(pd.read_csv(f) for f in files).to_csv("results/figures/discrete_eta/discrete_eta.csv", index=False)
print(f"merged {len(files)} eta tables")
PY
echo "diagnostics finished"
