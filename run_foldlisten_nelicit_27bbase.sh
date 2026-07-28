#!/usr/bin/env bash
# On-box NEUTRAL-ARM ELICITATION, box 3 (>=80GB box; 27b bf16 ~54GB resident): 27b-base ext2 ONLY.
# Pre-registered DESIGN_neutral_elicit.md sec 3.3. This is the most load-bearing and the most cap-fragile
# box in the run, so the two deviations from the sec 3.2 table are recorded here rather than in prose:
#
#   (1) INSTANCE IS PINNED TO H100 **SXM5**, NOT PCIe (see run_poll_launch_nelicit_27b.sh). One 27b cell
#       measures ~4.3 h on H100 PCIe (89 s/record, commit fd2154b; docs/lambda-gpu-access.md:41-42) and the
#       neutral-elicited 4th arm adds ~+7 % base decode -> ~4.6 h, against the design's 5.5 h cap: ~20 %
#       headroom. A single-cell box that hits its cap banks NOTHING (the precedent is exactly this cell:
#       fd2154b, rc=124, 128/164 items, zero artifact, ~$15 burnt). SXM5 is ~1.4-1.5 h/cell, i.e. cheaper
#       OVERALL (~$7 at $4.29/hr) than PCIe (~$15 at $3.29/hr) as well as far outside the cap.
#   (2) REMOTE_TIMEOUT IS 25200 (7 h), NOT 19800. The cap is sized so this cell still FITS even if the SXM5
#       box turns out to be no faster than the measured PCIe pace (4.6 h x ~1.5). A cap that is never
#       reached costs nothing; expected billing stays ~1.5 h. Reviewer note: the on-box self-destruct
#       backstop fires at REMOTE_TIMEOUT + REATTACH_GRACE (default 1800) = 27000 s, so a launcher death with
#       nobody reattaching bounds the orphan bill at ~7.5 h x $4.29 ~= $32; raise REATTACH_GRACE only if you
#       want a longer offline recovery window and accept that bound moving with it.
#
# No gate here: the gate cells are the -it ones (boxes 1/2/4); this cell's numbers are read by sec 5 step 3
# (byte-identity diff vs results_foldlisten_ext2_27b/out/foldlisten_judge_fl_27bbase_ext2_summary.json,
# committed faithful-strict elicited 11/39/32 fold & 20/34/28 listen) and only then by sec 2.2.
# Patched judge (neutral-elicited 4th arm + push_attribution{,_faithful}). Tag DELIBERATELY IDENTICAL to
# the committed one so the diff is a same-filename comparison. Files FLAT in ~/latent_verify.
set -uo pipefail
cd ~/latent_verify
. .venv/bin/activate
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p out

echo "=== selftests (model-free) ==="
python foldlisten_judge.py --selftest || { echo "SELFTEST_FAIL"; exit 1; }
python faithful_rescore.py --selftest || { echo "RESCORE_SELFTEST_FAIL"; exit 1; }

echo "=== 27b base (qa) ext2 ==="
python foldlisten_judge.py --family verifier_family_ext2.json --name google/gemma-2-27b --tag fl_27bbase_ext2 \
  --device cuda > out/foldlisten_nelicit_27bbase_ext2.log 2>&1; echo "exit=$?"
tail -4 out/foldlisten_nelicit_27bbase_ext2.log

echo "ALLDONE_FOLDLISTEN_NELICIT_27BBASE"
