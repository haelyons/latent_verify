# Failed attempt: missing transitive dependency in the launcher copy's scp list

2026-07-29. Box `896e334ca3ec4f9e9c335064206edcb8` (A100-SXM4-40GB, us-east-1, driver 570.148.08).
Died 60 s into the run, before any model load. Cost ~2 min of box time.

Cause: `controls/family_cave_diagnose_fmt.py:200` does
`from gapclose_item_joins import STAMP_KEYS, join_key` at MODULE level, and
`controls/gapclose_item_joins.py` was not in the launcher copy's hardcoded scp list
(`lambda_run.sh:93-135`). The registration's §12/E2 insert named only the two `_fmt`
files; the transitive dependency was missed by the runner author AND by the runner review.

`family_topk_shift_fmt.py` degrades gracefully when the module is absent (it skips only a
selftest transcription check, see `out/run_detached.log`), which is why the rank selftest
passed on-box and the diagnose one did not — the failure is asymmetric between the two
instruments and would not have been caught by testing the rank instrument alone.

Verbatim, from `out/run_detached.log`:

    Traceback (most recent call last):
      File "/home/ubuntu/latent_verify/family_cave_diagnose_fmt.py", line 200, in <module>
        from gapclose_item_joins import STAMP_KEYS, join_key
    ModuleNotFoundError: No module named 'gapclose_item_joins'
    SELFTEST_FAIL_DIAGNOSE_FMT (missing from the launcher copy's scp list?)

What worked, and should be kept: the runner runs both instruments' model-free selftests
BEFORE any model load, so a missing dependency costs ~2 minutes rather than a box-hour, and
it names the likely cause in its own failure line.

Fix applied: `controls/gapclose_item_joins.py` added to both launcher copies. The fix was
then verified by a recursive walk of module-level imports from both entry points against
each copy's scp list — `missing = NONE` for both — rather than by patching the one import
the traceback happened to name.
