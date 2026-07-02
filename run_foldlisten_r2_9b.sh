#!/usr/bin/env bash
# On-box round-2 job, gemma-2-9b-it: (1) faithfulness anchor (committed n=22 repro), (2) screen the round-2
# candidate family verifier_family_ext2.json, (3) Phase-0.5 THINK-probe capture on the COMBINED family
# (combined_family.json = base + ext + ext2 candidates) + pure-numpy fit, (4) gate v2 on the fresh summaries.
# Readout includes the NFKD accent-fold (family_generate_judge 2026-07-02). Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python think_probe_identity.py --selftest || { echo "PROBE_SELFTEST_FAIL"; exit 1; }

echo "=== faithfulness anchor: 9b-it verifier_family (n=22) ==="
python foldlisten_judge.py --family verifier_family --name google/gemma-2-9b-it --tag fl_9bit_anchor2 \
  --device cuda --chat > out/foldlisten_9bit_anchor2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_9bit_anchor2.log

echo "=== round-2 candidates: 9b-it verifier_family_ext2.json ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-9b-it --tag fl_9bit_ext2 \
  --device cuda --chat > out/foldlisten_9bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_9bit_ext2.log

echo "=== THINK-probe capture (combined family) + fit ==="
python think_probe_identity.py --capture --family combined_family.json --name google/gemma-2-9b-it \
  --tag tp_9bit_comb --device cuda --chat > out/think_probe_9bit.log 2>&1; echo "exit=$?"
tail -3 out/think_probe_9bit.log
python think_probe_identity.py --fit out/think_probe_capture_tp_9bit_comb.npz 2>&1 | tee -a out/think_probe_9bit.log

echo "=== GATE v2 (pure) on fresh summaries ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_9bit_anchor2_summary.json \
  out/foldlisten_judge_fl_9bit_ext2_summary.json --v2 2>&1 | tee out/foldlisten_gate_r2.log

echo "ALLDONE_FOLDLISTEN_R2_9B"
