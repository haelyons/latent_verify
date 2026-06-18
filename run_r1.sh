#!/usr/bin/env bash
# On-box: Round-1 cheap cluster (RESEARCH_QUESTIONS Part 2) -- R1-INSTR + R1-MOTIF, gemma-2-2b.
# Both are 2b -> a single A10 is sufficient ($1.29/hr, FRAMING sec-11). R1-DIFF (9b) is GATED on
# R1-INSTR returning CONCORDANT and is launched separately (run_r1_diff.sh, after calibration).
# Faithfulness-first: the model-free selftests must pass on-box before any model load (cheap guard;
# they also prove the metric/decision math survived the scp).
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== selftests (must pass before model runs) ==="
python instr_triangulation.py --selftest || { echo "SELFTEST_FAIL instr"; exit 1; }
python gate_dont_delete.py --selftest  || { echo "SELFTEST_FAIL motif"; exit 1; }

echo "=== R1-INSTR instrument triangulation (2b; incumbent knockout sweep + AtP + activation-patch) ==="
python instr_triangulation.py --name google/gemma-2-2b > out/r1_instr.log 2>&1; echo "instr exit=$?"
echo "--- tail r1_instr.log ---"; tail -45 out/r1_instr.log

echo "=== R1-MOTIF gate-dont-delete (2b base+it; weights-only OV + induction-pattern QK) ==="
python gate_dont_delete.py > out/r1_motif.log 2>&1; echo "motif exit=$?"
echo "--- tail r1_motif.log ---"; tail -45 out/r1_motif.log
echo "ALLDONE_R1"
