#!/usr/bin/env bash
# SEQUENCE_170626 batch 2: recurrence D2-fix (N-1 repro) -> localize208 (P-C/N-2) ->
# distractor/task base+it (N-6) -> copyscore (N-3/N-4). recurrence runs first; its faith
# block (reader->Sydney ~0.836, induction ~0.192) is the session faithfulness gate.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
echo "=== recurrence D2-fix (N-1) ==="; python job_recurrence.py > out/rec2.log 2>&1; echo "rec exit=$?"
echo "=== localize208 (P-C/N-2) ===";   python job_localize208.py > out/loc208.log 2>&1; echo "loc exit=$?"
echo "=== distractor base (N-6) ===";   python job_distractor_task.py --model base > out/dt_base.log 2>&1; echo "dt_base exit=$?"
echo "=== distractor it (N-6) ===";     python job_distractor_task.py --model it   > out/dt_it.log 2>&1; echo "dt_it exit=$?"
echo "=== copyscore (N-3/N-4) ===";     python job_copyscore.py > out/cs.log 2>&1; echo "cs exit=$?"
echo "ALLDONE2"
