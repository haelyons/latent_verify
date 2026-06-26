"""ANY-SCALE residual-state restoration battery: run cave_residstate_decisive's PART8-v7 battery (ALL-attention +
ALL-MLP output-patch restoration, the +/- steer -it positive control, the label-match re-read, and the bootstrap
CIs) at an ARBITRARY model scale by picking the cave-axis layer SCALE-RELATIVELY (~0.667 depth) instead of the
hardcoded AXIS_LAYER=28. A hardcoded layer-28 readout KeyErrors / mis-reads on any model whose block count != 42
(e.g. gemma-2-2b has 26 blocks, a 27b has 46); this wrapper makes the SAME battery scale-portable.

NOTHING about the measurement changes except the layer selection. Every measured quantity, every intervention,
every threshold, and the decision function are IMPORTED VERBATIM from cave_residstate_decisive / cave_residstate_close
(no copy-paste of their bodies). The ONLY behavioural change vs cave_residstate_decisive is:

    AXIS_LAYER  := round(0.667 * model.cfg.n_layers) clamped to [1, n_layers-1]     (was the constant 28)
    READ_LAYERS := sorted({axis-4, axis, axis+4}) each clamped to [1, n_layers-1]   (was the constant [24,28,32])

The clamped 0.667-depth picker is ported VERBATIM from cave_fold_vs_listen.pick_read_layer
(max(1, min(n_layers - 1, round(0.667 * n_layers)))); READ_LAYERS mirrors the decisive/close +/-4 window around it.

HOW THE LAYER REACHES THE REUSED MACHINERY (no edit to the siblings): cave_residstate_decisive's _measure/_decisive/
_arm/decisive_decision and cave_residstate_close's _measure read AXIS_LAYER / READ_LAYERS / AUROC_THR as MODULE-LEVEL
names in their own modules. So at run time, the instant a model is constructed we read model.cfg.n_layers off it,
compute the scale-relative layer, and REBIND those module attributes on both sibling modules (a temporary monkeypatch
of HookedTransformer.from_pretrained_no_processing installs the read-and-rebind at load time, restored in a finally).
We then delegate to cave_residstate_decisive.run unchanged. Both models in a base/it pair share n_layers, so the
rebind is stable across the two loads.

CONSTRAINT OF RECORD (faithfulness-gate-able): at n_layers=42 (gemma-2-9b) the picker returns 28 and READ_LAYERS
returns [24, 28, 32] -- i.e. EXACTLY the constants cave_residstate_decisive/cave_residstate_close hardcode -- so
running THIS file at the 9b pair reproduces cave_residstate_decisive.py's 9b path BIT-FOR-BIT (same machinery, same
layer 28, same union set, same decision). It is therefore a strict generalisation, gate-able against the 9b result.

readout = resid_post[AXIS_LAYER].cave-axis (NOT the emitted token); base label = realized argmax==W*; it label =
free-gen self-judge. cave-axis = unit diff-of-means(caved vs not), gated on held-out AUROC. Verdict categories
(UNCHANGED, from cave_residstate_decisive.decisive_decision): ATTENTION_CARRIES / MLP_CARRIES / BOTH_REDUNDANT /
NEITHER_LOCALIZED / CHANNEL_INERT / INSUFFICIENT.

  python controls/cave_residstate_anyscale.py --selftest
  python controls/cave_residstate_anyscale.py --base google/gemma-2-9b   --it google/gemma-2-9b-it   --device cuda --big-pool --tag 9b
  python controls/cave_residstate_anyscale.py --base google/gemma-2-2b   --it google/gemma-2-2b-it   --device cuda --big-pool --tag 2b
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))            # controls/ for sibling-control imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # latent_verify/ for the repo imports

# Import (do NOT re-implement) the whole measurement battery + the decision function from the decisive control,
# and the matched-set machinery + AUROC gate from the close. We rebind the layer constants on these modules at
# run time; their function BODIES are used verbatim.
import cave_residstate_close as close_mod          # noqa: E402
import cave_residstate_decisive as dec_mod         # noqa: E402
# Re-export the proven pure-functions so the selftest exercises the SAME objects the run uses (no shadow copies).
from cave_residstate_decisive import (             # noqa: E402,F401
    bootstrap_ci, axis_means, restoration_mean, decisive_decision, STEER_EPS,
)
from cave_residstate_diff import proj_restoration  # noqa: E402,F401


# ----------------------------------------------------------------------------- pure: the ONLY new behaviour
def pick_axis_layer(n_layers):
    """Scale-relative cave-axis layer (~0.667 depth), clamped to [1, n_layers-1]. PORTED VERBATIM from
    cave_fold_vs_listen.pick_read_layer: max(1, min(n_layers - 1, round(0.667 * n_layers))). 9b(42)->28 (the
    validated spike layer, == the decisive/close hardcoded AXIS_LAYER), 2b(26)->17. A hardcoded 28 KeyErrors on
    a 26-block model; the AUROC_THR gate backstops a poorly-chosen per-model layer. Pure (int -> int)."""
    return max(1, min(n_layers - 1, round(0.667 * n_layers)))


def pick_read_layers(n_layers):
    """The decisive/close READ_LAYERS window around the axis layer: sorted({axis-4, axis, axis+4}), each clamped to
    [1, n_layers-1] (mirrors the hardcoded [24, 28, 32] = {28-4, 28, 28+4} at n_layers=42). Deduped after clamping
    (at the edges two of the three can collapse). The AXIS_LAYER itself is always a member. Pure (int -> list)."""
    axis = pick_axis_layer(n_layers)
    def clamp(L):
        return max(1, min(n_layers - 1, L))
    return sorted({clamp(axis - 4), clamp(axis), clamp(axis + 4)})


# ----------------------------------------------------------------------------- run: rebind the layer, reuse the body
def _install_scale_relative_layer(device):
    """Monkeypatch HookedTransformer.from_pretrained_no_processing so that the instant a model is constructed we
    read model.cfg.n_layers off it and REBIND AXIS_LAYER / READ_LAYERS on BOTH sibling modules to their
    scale-relative values. Returns (restore_callable). The patch is the ONLY behavioural change vs the decisive
    control; everything downstream (the imported _measure/_decisive/_arm/decisive_decision bodies) reads the
    rebound module globals exactly as it reads its own hardcoded constants. No sibling file is edited."""
    from transformer_lens import HookedTransformer
    orig = HookedTransformer.from_pretrained_no_processing

    # remember the siblings' ORIGINAL constants so we restore them (keeps an in-process 9b run identical to the
    # decisive control even after this wrapper has run at another scale).
    saved = {"close_axis": close_mod.AXIS_LAYER, "close_read": list(close_mod.READ_LAYERS),
             "dec_axis": dec_mod.AXIS_LAYER}

    def patched(name, *a, **k):
        model = orig(name, *a, **k)
        nL = model.cfg.n_layers
        axis = pick_axis_layer(nL)
        reads = pick_read_layers(nL)
        # rebind on BOTH modules: cave_residstate_close._measure reads close_mod.AXIS_LAYER + close_mod.READ_LAYERS;
        # cave_residstate_decisive._decisive/_arm/decisive_decision read dec_mod.AXIS_LAYER (imported into dec_mod's
        # namespace). Set them every load (a base/it pair shares n_layers, so the value is stable).
        close_mod.AXIS_LAYER = axis
        close_mod.READ_LAYERS = reads
        dec_mod.AXIS_LAYER = axis
        print(f"[anyscale] n_layers={nL} -> AXIS_LAYER={axis} READ_LAYERS={reads} "
              f"(rule round(0.667*n_layers), clamp [1,{nL - 1}])", flush=True)
        return model

    HookedTransformer.from_pretrained_no_processing = staticmethod(patched)

    def restore():
        HookedTransformer.from_pretrained_no_processing = orig
        close_mod.AXIS_LAYER = saved["close_axis"]
        close_mod.READ_LAYERS = saved["close_read"]
        dec_mod.AXIS_LAYER = saved["dec_axis"]

    return restore


def run(base_name, it_name, device, big_pool, tag):
    """Delegate to cave_residstate_decisive.run UNCHANGED, with the scale-relative-layer patch installed for its
    duration. The decisive control writes out/cave_residstate_decisive.json (+ _cache.json); we additionally stamp
    out/cave_residstate_anyscale.json with the resolved layer rule so the runner sees which layer this scale used."""
    import json
    restore = _install_scale_relative_layer(device)
    try:
        dec_mod.run(base_name, it_name, device, big_pool)
    finally:
        # capture what the patch resolved (post-run module state if the run loaded a model; else the constants)
        resolved_axis, resolved_reads = dec_mod.AXIS_LAYER, list(close_mod.READ_LAYERS)
        restore()
    Path("out").mkdir(exist_ok=True)
    Path("out/cave_residstate_anyscale.json").write_text(json.dumps(
        {"base": base_name, "it": it_name, "tag": tag,
         "axis_layer_rule": "round(0.667*n_layers) clamped to [1,n_layers-1] (== cave_fold_vs_listen.pick_read_layer)",
         "read_layers_rule": "sorted({axis-4, axis, axis+4}) each clamped to [1,n_layers-1]",
         "resolved_axis_layer": resolved_axis, "resolved_read_layers": resolved_reads,
         "reuses": "cave_residstate_decisive.run (battery + decisive_decision) + cave_residstate_close._measure",
         "decision_in": "out/cave_residstate_decisive.json"}, indent=2, default=str))
    print(f"[ANYSCALE {tag}] resolved AXIS_LAYER={resolved_axis} READ_LAYERS={resolved_reads}; "
          f"decision in out/cave_residstate_decisive.json (wrote out/cave_residstate_anyscale.json)", flush=True)


# ----------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    # ---------- pick_axis_layer: scale-relative, reproduces the validated 9b layer, valid on 2b, clamps at edges
    assert pick_axis_layer(42) == 28, pick_axis_layer(42)   # 9b -> the spike layer == decisive/close AXIS_LAYER
    assert pick_axis_layer(26) == 17, pick_axis_layer(26)   # 2b -> in-range (a hardcoded 28 would KeyError)
    assert 1 <= pick_axis_layer(42) <= 41 and 1 <= pick_axis_layer(26) <= 25
    # edge clamps: tiny model can't exceed n_layers-1; degenerate small n stays >= 1.
    assert pick_axis_layer(2) == 1, pick_axis_layer(2)      # round(1.334)=1 -> within [1,1]
    assert pick_axis_layer(3) == 2, pick_axis_layer(3)      # round(2.001)=2 -> within [1,2]
    assert pick_axis_layer(4) == min(3, max(1, round(0.667 * 4))) and 1 <= pick_axis_layer(4) <= 3
    # an n where the raw value would exceed n_layers-1 is clamped down to n_layers-1.
    big = pick_axis_layer(46); assert 1 <= big <= 45 and big == round(0.667 * 46)  # 31, in-range (27b-ish)
    print(f"[selftest] pick_axis_layer: 9b(42)->{pick_axis_layer(42)} 2b(26)->{pick_axis_layer(26)} "
          f"46->{big} edges(2->1,3->2) clamped to [1,n-1]")

    # ---------- pick_read_layers: window around the axis; reproduces the hardcoded [24,28,32] at 9b; clamps at edges
    assert pick_read_layers(42) == [24, 28, 32], pick_read_layers(42)        # 9b -> EXACTLY the close READ_LAYERS
    assert pick_read_layers(26) == [13, 17, 21], pick_read_layers(26)        # 2b -> {17-4,17,17+4}, all in-range
    for n in (8, 12, 26, 42, 46):
        rl = pick_read_layers(n)
        assert rl == sorted(set(rl)), rl                                     # sorted + deduped
        assert all(1 <= L <= n - 1 for L in rl), (n, rl)                     # every member clamped in-range
        assert pick_axis_layer(n) in rl, (n, rl)                            # the axis layer is always a member
    # at a small n the +/-4 window collapses onto the clamp edges (dedupe shrinks the set, axis still present).
    assert len(pick_read_layers(6)) <= 3 and pick_axis_layer(6) in pick_read_layers(6)
    print(f"[selftest] pick_read_layers: 9b(42)->{pick_read_layers(42)} 2b(26)->{pick_read_layers(26)} "
          f"(sorted/deduped, axis-in-set, clamped)")

    # ---------- CONSTRAINT OF RECORD: at n_layers=42 the rule == the decisive/close hardcoded constants ----------
    # (so a 9b run of THIS file reproduces cave_residstate_decisive's 9b path bit-for-bit -- faithfulness-gate-able).
    assert pick_axis_layer(42) == dec_mod.AXIS_LAYER, (pick_axis_layer(42), dec_mod.AXIS_LAYER)
    assert pick_axis_layer(42) == close_mod.AXIS_LAYER, (pick_axis_layer(42), close_mod.AXIS_LAYER)
    assert pick_read_layers(42) == sorted(close_mod.READ_LAYERS), (pick_read_layers(42), close_mod.READ_LAYERS)
    print(f"[selftest] CONSTRAINT OF RECORD: 9b(42) rule reproduces hardcoded AXIS_LAYER={dec_mod.AXIS_LAYER} "
          f"READ_LAYERS={sorted(close_mod.READ_LAYERS)} bit-for-bit")

    # ---------- the IMPORTED pure-functions reproduce on synthetic arrays (reuse the siblings' selftest idiom) ----
    # bootstrap_ci (cave_residstate_decisive)
    assert bootstrap_ci([0.5, 0.5, 0.5]) == [0.5, 0.5]
    _ci = bootstrap_ci([0.0, 1.0] * 50); assert _ci[0] < 0.5 < _ci[1], _ci
    assert bootstrap_ci([]) is None
    # axis_means + restoration_mean (cave_residstate_decisive)
    rc = {"a": [2.0, 0.0], "b": [0.0, 0.0]}; lab = {"a": 1, "b": 0}; ax = [1.0, 0.0]
    cm, ncm = axis_means(rc, lab, ax); assert abs(cm - 2.0) < 1e-9 and abs(ncm) < 1e-9, (cm, ncm)
    _base = {"a": [2.0, 0.0]}; _inter = {"a": [0.0, 0.0]}
    _m, _l = restoration_mean(["a"], _base, _inter, ax, 2.0, 0.0); assert abs(_m - 1.0) < 1e-9, _m
    # proj_restoration (cave_residstate_diff, the restoration kernel)
    assert abs(proj_restoration(2.0, 0.0, 2.0, 0.0) - 1.0) < 1e-9        # back to not-caved -> 1.0
    assert proj_restoration(2.0, 2.0, 2.0, 0.0) == 0.0                   # unmoved -> 0
    assert proj_restoration(2.0, 0.0, 1.0, 1.0) == 0.0                   # zero gap -> 0
    print("[selftest] reused restoration/steer kernels: bootstrap_ci + axis_means + restoration_mean + "
          "proj_restoration reproduce")

    # ---------- the IMPORTED decision function (UNCHANGED categories) fires on synthetic arrays -----------------
    # All six branches of cave_residstate_decisive.decisive_decision, exercised on the SAME object the run uses.
    d_attn = decisive_decision(0.40, 0.02, 0.01, 0.38, 0.5, -0.4, True)
    assert d_attn["category"] == "ATTENTION_CARRIES", d_attn
    d_mlp = decisive_decision(0.02, 0.40, 0.01, 0.38, 0.5, -0.4, True)
    assert d_mlp["category"] == "MLP_CARRIES", d_mlp
    d_both = decisive_decision(0.40, 0.40, 0.01, 0.38, 0.5, -0.4, True)
    assert d_both["category"] == "BOTH_REDUNDANT", d_both
    d_none = decisive_decision(0.02, 0.02, 0.01, 0.38, 0.5, -0.4, True)
    assert d_none["category"] == "NEITHER_LOCALIZED", d_none
    d_inert = decisive_decision(0.02, 0.02, 0.01, 0.38, 0.05, 0.02, True)
    assert d_inert["category"] == "CHANNEL_INERT", d_inert
    d_ins = decisive_decision(0.40, 0.02, 0.01, 0.38, 0.5, -0.4, False)
    assert d_ins["category"] == "INSUFFICIENT", d_ins
    # channel-live boundary: +steer minus -steer exactly STEER_EPS is live (inclusive >=); just under is inert.
    assert decisive_decision(0.40, 0.02, 0.01, 0.38, STEER_EPS, 0.0, True)["channel_live"] is True
    assert decisive_decision(0.40, 0.02, 0.01, 0.38, STEER_EPS - 1e-6, 0.0, True)["category"] == "CHANNEL_INERT"
    print("[selftest] decisive_decision (UNCHANGED): ATTENTION/MLP/BOTH/NEITHER/CHANNEL_INERT/INSUFFICIENT + "
          f"STEER_EPS({STEER_EPS}) boundary fire")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--base", default="google/gemma-2-9b")
    p.add_argument("--it", default="google/gemma-2-9b-it")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--big-pool", action="store_true")
    p.add_argument("--tag", default="9b")
    a = p.parse_args()
    selftest() if a.selftest else run(a.base, a.it, a.device, a.big_pool, a.tag)


if __name__ == "__main__":
    main()
