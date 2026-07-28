#!/usr/bin/env bash
# On-box NEUTRAL-ARM ELICITATION, box 4 (>=80GB box; 27b bf16 ~54GB resident): 27b-it ext2 ONLY.
# Pre-registered DESIGN_neutral_elicit.md sec 3.3, P3 priority (completeness + the figure's -it row).
# Sized exactly as the design froze it -- cheapest >=80GB <= $5.50/hr, REMOTE_TIMEOUT=19800 (5.5 h) -- and
# NOT pinned to SXM5: -it decodes ~2x shorter than base (est.: ~2.0-2.5 h PCIe) and the 4th arm adds only
# ~+2.4 %, so the 5.5 h cap carries >2x headroom on the slowest plausible box. That 2x figure is an
# INFERENCE from committed generation lengths, not a measured pace (DESIGN sec 6) -- if it is wrong in the
# bad direction this cell hits the same nothing-banked failure as fd2154b, so read the poll tail.
# This cell also carries the UNRESOLVED gate contest that must NOT silently resolve (sec 5 step 4): the
# committed ext2 27b-it gate FAILs on commit labels (listen drift 13 > 11.18) and PASSes on faithful (7).
# Both readings are therefore written as separate artifacts below; if the contest disappears, something
# changed that should not have and the run is a repro failure, not a finding.
# Patched judge (neutral-elicited 4th arm + push_attribution{,_faithful}). Tag DELIBERATELY IDENTICAL to
# the committed one so the sec 5 diff is a same-filename comparison against
# results_foldlisten_ext2_27b/out/. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== 27b-it (chat) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b-it --tag fl_27bit_ext2 \
  --device cuda --chat > out/foldlisten_nelicit_27bit_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_nelicit_27bit_ext2.log

echo "=== GATE v2 (pure) on the fresh -it summary, both label readings ==="
python foldlisten_judge.py --gate out/foldlisten_judge_fl_27bit_ext2_summary.json --v2 \
  2>&1 | tee out/foldlisten_gate_nelicit_27bit_commit.log
python foldlisten_judge.py --gate out/foldlisten_judge_fl_27bit_ext2_summary.json --v2 --labels faithful \
  2>&1 | tee out/foldlisten_gate_nelicit_27bit_faithful.log

echo "ALLDONE_FOLDLISTEN_NELICIT_27BIT"
