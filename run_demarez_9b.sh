#!/usr/bin/env bash
# On-box DE MAREZ span runs at 9b-it on the frozen 74 (docs/drafts/REGISTRATION_demarez_spans.md
# SS11/SS13). Run A = hook-free token-span SUBSTITUTION (foldlisten_demarez_subst.py, tag dmz_9bit_a);
# Run B = span-MASK (foldlisten_demarez_mask.py, tag dmz_9bit_b). Order is REGISTERED (SS13.4):
# model-free selftests (hard exit) -> --n 6 smoke of BOTH instruments -> Run A FULL, TO COMPLETION ->
# Run B full. A cap-hit that loses Run B voids SS6.7-SS6.9 only and Run A's verdicts survive (SS1),
# so Run B never starts unless Run A exited 0 WITH its summary on disk. Floors/anchors are CITED via
# --floor-* / --*-committed flags (SS5, SS13.1), never recomputed; --nomask-ref / --p3c are left to
# the offline join (controls/foldlisten_demarez_join.py, never shipped -- the only verdict source).
# Files FLAT in ~/latent_verify; remote_run.sh provides the venv, so plain `python` here.
set -uo pipefail
cd ~/latent_verify
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SS1.1 LAUNCH CONDITION (added 2026-07-31, AFTER the first run): the same-session test requires
# cuda_visible_devices, and the 2026-07-30 run left it unset, so both artifacts stamped it null, the
# pair read NOT_SAME_BOX and SS6.7/SS6.8/SS6.9 returned UNEVALUABLE on complete data (n_common=74,
# n_span_located=74). This exports it so a re-run satisfies the test. It is a LAUNCH CONDITION only:
# no threshold, arm, span rule or decision rule is touched, and the rule itself was NOT relaxed.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p out

echo "=== [a] subst selftest (model-free, CPU) ==="
python foldlisten_demarez_subst.py --selftest
rc=$?; echo "[a] rc=$rc"
if [ $rc -ne 0 ]; then echo "SUBST_SELFTEST_FAIL rc=$rc"; exit 1; fi

echo "=== [b] mask selftest (model-free, CPU) ==="
python foldlisten_demarez_mask.py --selftest
rc=$?; echo "[b] rc=$rc"
if [ $rc -ne 0 ]; then echo "MASK_SELFTEST_FAIL rc=$rc"; exit 1; fi

echo "=== [c] subst smoke: --n 6, tag dmz_9bit_a_smoke ==="
python foldlisten_demarez_subst.py --run --family mechanism_family_9bit.json \
  --name google/gemma-2-9b-it --tag dmz_9bit_a_smoke --device cuda --chat --n 6 \
  --floor-nc 0.0 --floor-fold-nomask 1.0 --floor-parametric 0.013513513513513514 \
  > out/foldlisten_demarez_subst_smoke.log 2>&1
rc=$?; echo "[c] rc=$rc"
if [ $rc -ne 0 ] || [ ! -f out/foldlisten_demarez_subst_dmz_9bit_a_smoke_summary.json ]; then
  echo "SUBST_SMOKE_FAIL rc=$rc"; tail -30 out/foldlisten_demarez_subst_smoke.log; exit 1
fi
echo "subst smoke OK"

echo "=== [d] mask smoke: --n 6, tag dmz_9bit_b_smoke ==="
python foldlisten_demarez_mask.py --run --family mechanism_family_9bit.json \
  --name google/gemma-2-9b-it --tag dmz_9bit_b_smoke --device cuda --chat --n 6 \
  --floor-nc-masked 0.02702702702702703 --floor-nw-masked 0.2714285714285714 \
  --fold-mask-committed 0.0273972602739726 --padding-committed 0.013888888888888888 \
  > out/foldlisten_demarez_mask_smoke.log 2>&1
rc=$?; echo "[d] rc=$rc"
if [ $rc -ne 0 ] || [ ! -f out/foldlisten_demarez_mask_dmz_9bit_b_smoke_summary.json ]; then
  echo "MASK_SMOKE_FAIL rc=$rc"; tail -30 out/foldlisten_demarez_mask_smoke.log; exit 1
fi
echo "mask smoke OK"

# AUDIT N3: the SS3.3 span locator round-trips decode->re-encode on the gemma-2 chat prompt and cannot be
# tested off-box. One divergence makes EVERY item SPAN_UNLOCATABLE, which guts SS6.7/SS6.8 -- and the smoke
# still exits 0, so rc alone will not catch it. Gate the full runs on the smoke's own locatability category.
LOC=$(python -c "import json;print(json.load(open('out/foldlisten_demarez_mask_dmz_9bit_b_smoke_summary.json'))['span_locatability']['category'])" 2>/dev/null)
echo "[d] smoke span_locatability.category=$LOC"
if [ "$LOC" != "SPAN_LOCATED_ALL" ]; then
  echo "SPAN_LOCATOR_FAIL category=$LOC -- the SS3.3 locator does not hold on this tokenizer; NOT starting the full runs"
  python -c "import json;d=json.load(open('out/foldlisten_demarez_mask_dmz_9bit_b_smoke_summary.json'))['span_locatability'];print('n_located=%s/%s'%(d['n_located'],d['n_items']));print(json.dumps(d['unlocatable_log'][:5],indent=1))" 2>/dev/null
  exit 1
fi

echo "=== [e] Run A FULL: substitution arms A1-A8 on the frozen 74, tag dmz_9bit_a ==="
python foldlisten_demarez_subst.py --run --family mechanism_family_9bit.json \
  --name google/gemma-2-9b-it --tag dmz_9bit_a --device cuda --chat \
  --floor-nc 0.0 --floor-fold-nomask 1.0 --floor-parametric 0.013513513513513514 \
  > out/foldlisten_demarez_subst_dmz_9bit_a.log 2>&1
rc=$?; echo "[e] rc=$rc"
tail -12 out/foldlisten_demarez_subst_dmz_9bit_a.log
if [ $rc -ne 0 ] || [ ! -f out/foldlisten_demarez_subst_dmz_9bit_a_summary.json ]; then
  echo "RUN_A_FAIL rc=$rc -- Run B NOT started (SS1: Run A must complete before Run B)"; exit 1
fi
echo "Run A complete (summary on disk)"

echo "=== [f] Run B FULL: mask arms B1-B8 on the frozen 74, tag dmz_9bit_b ==="
python foldlisten_demarez_mask.py --run --family mechanism_family_9bit.json \
  --name google/gemma-2-9b-it --tag dmz_9bit_b --device cuda --chat \
  --floor-nc-masked 0.02702702702702703 --floor-nw-masked 0.2714285714285714 \
  --fold-mask-committed 0.0273972602739726 --padding-committed 0.013888888888888888 \
  > out/foldlisten_demarez_mask_dmz_9bit_b.log 2>&1
rc=$?; echo "[f] rc=$rc"
tail -12 out/foldlisten_demarez_mask_dmz_9bit_b.log
# AUDIT S5: steps [c][d][e] all test for their summary; [f] tested rc only, so a writer that exited 0
# without writing would have been reported as success. Run A's verdicts survive a lost Run B (SS1).
if [ $rc -ne 0 ] || [ ! -f out/foldlisten_demarez_mask_dmz_9bit_b_summary.json ]; then
  echo "RUN_B_FAIL rc=$rc -- Run A stands (SS1: a cap-hit that loses Run B voids SS6.7-SS6.9 only)"
fi

echo "ALLDONE_DEMAREZ_9B"
exit $rc
