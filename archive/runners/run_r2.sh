#!/usr/bin/env bash
# On-box: Round-2 = the latent_skeptic author_queue items from R1 Pass-1 triage (gemma-2-2b, A10).
#   R2a/R2b (instr robustness): the two NEEDS_RUN against instr-concordant --
#     noise-floor (bootstrap reader rank-CI + random-label null) and single-case-overfit (held-out pairs).
#   R2c (motif clean test): gate_dont_delete with the basket re-selected by base OV-copy capability
#     (the basket_selection confound that EXPLAINS-ed the 0/10 null).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== selftests (must pass before model runs) ==="
python instr_triangulation.py --selftest || { echo "SELFTEST_FAIL instr"; exit 1; }
python gate_dont_delete.py --selftest  || { echo "SELFTEST_FAIL motif"; exit 1; }

echo "=== R2a instr CURATED + bootstrap rank-CI + random-label null (noise-floor NEEDS_RUN) ==="
python instr_triangulation.py --pairs curated --no-knockout-sweep --seed 0 > out/r2_instr_curated.log 2>&1; echo "exit=$?"
tail -25 out/r2_instr_curated.log

echo "=== R2b instr HELD-OUT pairs + full knockout cross-check (single-case-overfit NEEDS_RUN) ==="
python instr_triangulation.py --pairs heldout --seed 0 > out/r2_instr_heldout.log 2>&1; echo "exit=$?"
tail -35 out/r2_instr_heldout.log

echo "=== R2c gate_dont_delete --select copy (clean motif test on OV-copy heads) ==="
python gate_dont_delete.py --select copy > out/r2_motif_copy.log 2>&1; echo "exit=$?"
tail -40 out/r2_motif_copy.log
echo "ALLDONE_R2"
