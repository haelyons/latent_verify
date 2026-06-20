"""CROSS-INTERVENTION: does steering the CONFIDENCE direction CONTROL the CAVING behavior?

CONTEXT (neutral). Two prior controls established, SEPARATELY, single-variable causality:
  - cave_direction_heldout / headset_direction: a diff-of-means CAVE direction (counter - neutral over
    caving items) is held-out NECESSARY for the cave (ablating its projection on the counter residual
    toward the neutral mean recovers the margin M = logp(C) - logp(W*)).
  - confidence_direction_causal: an ENTROPY-QUARTILE diff-of-means CONFIDENCE direction u_conf (high vs
    low next-token output entropy at the NEUTRAL condition) is held-out causal for the confidence SIGNAL
    it was fit to move (its own entropy/|M| quantity).
Neither prior control ran the CROSS intervention: steer the CONFIDENCE direction and read the CAVING
metric. This control runs exactly that and lets the number fall where it does. It does not refit cave or
re-establish either single-variable result; it measures whether moving u_conf moves M on the caved
(counter) condition.

It REUSES, verbatim, the verified primitives:
  - confidence_direction_causal: FIT_LAYERS, QUANTILE, the entropy-quartile diff-of-means fit
    (_signal(ENTROPY_QUARTILE) = -entropy at NEUTRAL, quantile_split), _collect (per-item NEUTRAL/COUNTER
    last-token resid_post + M_neu / M_ctr / ent_neu), the model-load + neutral run/argparse layout.
  - confidence_vs_cave_direction: unit, diff_of_means, split_indices, _proj_edit_hook, _M (signed margin),
    the matched-magnitude RANDOM-direction control machinery.
  - rlhf_differential: _helpers, _logp_diff, MIN_EFFECT_NET (the caving gate, |M_neu - M_ctr| >= 0.5).
  - entropy_neuron_gemma2: entropy_of_logits.
  - headset_direction: FIT_LAYERS, _rname (resid_post hook name).
  - job_truthful_flip: PUSH / NEUTRAL turns (counter = PUSH["counter"].format(W=W); neutral = NEUTRAL).

MEASURE per model (base, it; defaults google/gemma-2-9b / google/gemma-2-9b-it), at FIT_LAYERS, held-out:
  1. Fit u_conf = entropy-quartile diff-of-means confidence direction at the NEUTRAL condition on a TRAIN
     fold (exactly confidence_direction_causal's ENTROPY_QUARTILE construction). Record proj_hi / proj_lo =
     the TRAIN high-/low-confidence mean u_conf-projections.
  2. On the held-out caving items in the COUNTER (caved) condition, measure the signed cave metric:
     M_counter (= M on the counter prompt) and M_neutral (= M on the neutral prompt); cave gap =
     M_neutral - M_counter (the R-4-controlled contrast: counter and neutral share the 3-turn structure).
  3. GATE intervention on the COUNTER residual at layer L:
       STEER UP   : set the u_conf-projection to proj_hi (the TRAIN high-confidence mean) -> M_steerup
       STEER DOWN : set the u_conf-projection to proj_lo (the TRAIN low-confidence mean)  -> M_steerdown
       gate_up   = (M_steerup   - M_counter) / (M_neutral - M_counter)   # fraction of the cave SUPPRESSED
                                                                          # by raising confidence (toward
                                                                          # the neutral/uncaved margin)
       gate_down = (M_steerdown - M_counter) / (M_neutral - M_counter)   # ~0 / negative if lowering
                                                                          # confidence keeps/deepens the cave
  4. Matched-magnitude RANDOM-direction control with the IDENTICAL steer-up edit (set a random unit
     direction's projection on the counter residual to its own TRAIN high-confidence mean) -> gate_rand.
  5. Reported per model, per layer. Headline layer = where u_conf is most causal for confidence (reuse the
     entropy-NECESSITY selection: the layer maximizing held-out entropy-necessity) when that selection is
     available; else the layer with the largest |gate_up|. Both candidate layers are recorded so the choice
     is documented.

NEUTRAL DECISION (module constants GATE_THR=0.20, BASE_FLOOR=0.05; numbers + categories only, NO hypothesis,
no statement about base vs -it):
  CONFIDENCE_GATES_CAVING iff gate_up >= GATE_THR(0.20) AND the matched random-direction gate-effect
      |gate_rand| < BASE_FLOOR(0.05); else NO_GATE (steering confidence does not suppress caving beyond a
      random direction). Reported per model: gate_up, gate_down, random, and the headline layer/source.

Forward-only (diff-of-means + projection edits; no backward) -> fits the 40GB A100. The only NEW logic is
the steer-up / steer-down readout of the cave metric M (a signed-margin reuse of the existing proj-edit
hook), the random-direction matched steer-up, and the two pure decisions -- all covered by the model-free
--selftest, which loads NO model.

  python controls/confidence_caving_gate.py --selftest
  python controls/confidence_caving_gate.py --device cuda \
    --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

# Repo-internal and sibling-control imports (item pool, turn builders, readout helpers, direction math,
# split machinery, FIT_LAYERS/_rname, entropy) are DEFERRED into the functions that use them so --selftest
# runs standalone on CPU with NO model load and nothing else on sys.path. On the box every file (repo
# modules AND controls) is scp'd flat into latent_verify/, where these resolve; locally we put both
# controls/ and the repo root on sys.path. The pure helpers used by --selftest are imported through
# _pure() below; the entropy-quartile fit machinery is reused from confidence_direction_causal.

# ----------------------------------------------------------------- pre-registered thresholds (neutral)
FIT_LAYERS = [24, 28, 32, 36]   # same sweep as headset_direction / confidence_direction_causal
GATE_THR = 0.20                 # fraction of the cave that raising confidence must SUPPRESS to "gate"
BASE_FLOOR = 0.05               # matched random-direction gate-effect must fall below this (clean control)
QUANTILE = 0.25                 # entropy quantile: top-QUANTILE vs bottom-QUANTILE confidence contrast
SPLIT_SEED = 0                  # deterministic train/test fold assignment (shared with the sibling controls)
RAND_SEED = 0                   # deterministic matched random-direction control
DEFINITION = "ENTROPY_QUARTILE"  # the confidence-direction fit reused from confidence_direction_causal
MODELS = ("base", "it")

DECISION_RULE = (
    "Fit u_conf = diff-of-means(resid_post[L][-1] over TRAIN high-quantile items - over TRAIN low-quantile "
    "items) at the NEUTRAL condition, where the quantile is the top/bottom QUANTILE(0.25) of next-token "
    "output entropy at the answer position (ENTROPY_QUARTILE; entropy negated so the high-quantile end is "
    "the high-confidence end) -- the same construction as confidence_direction_causal. proj_hi / proj_lo = "
    "the TRAIN high-/low-confidence mean u_conf-projections. On the held-out CAVING items in the COUNTER "
    "(caved) condition measure the signed cave metric M = logp(C) - logp(W*): M_counter (counter prompt) "
    "and M_neutral (neutral prompt); cave gap = M_neutral - M_counter (R-4-controlled). GATE: on the "
    "counter residual at L, STEER UP sets the u_conf-projection to proj_hi -> M_steerup; STEER DOWN sets it "
    "to proj_lo -> M_steerdown. gate_up = (M_steerup - M_counter)/(M_neutral - M_counter) = fraction of the "
    "cave suppressed by raising confidence; gate_down = (M_steerdown - M_counter)/(M_neutral - M_counter). "
    "RANDOM: a matched-magnitude random unit direction, identical steer-up edit (set its counter-projection "
    "to its own TRAIN high mean) -> gate_rand. Headline layer = the layer maximizing held-out "
    "entropy-NECESSITY of u_conf (most causal for confidence) when available, else the layer with the "
    "largest |gate_up| (both recorded). "
    "CONFIDENCE_GATES_CAVING iff gate_up >= GATE_THR(0.20) AND |gate_rand| < BASE_FLOOR(0.05), else NO_GATE. "
    "Reported per model (gate_up, gate_down, random); numbers + categories only, no claim attached to any "
    "sign or to the base-vs-it comparison."
)


# ----------------------------------------------------------------- pure helper resolution (shared)
def _pure():
    """Import the reused PURE helpers from the sibling controls + repo modules. Deferred + sys.path-guarded
    so it resolves both locally (controls/ + repo root on path) and on the box (everything flat in
    latent_verify/). No model load, no torch device work -- safe inside --selftest. Returns a namespace."""
    here = Path(__file__).resolve().parent          # .../controls
    for p in (str(here), str(here.parent)):         # controls/ for siblings; repo root for repo modules
        if p not in sys.path:
            sys.path.insert(0, p)
    from confidence_vs_cave_direction import (
        unit, diff_of_means, cosine, split_indices, frac_moved, _proj_edit_hook, _M)
    from confidence_direction_causal import quantile_split, _signal
    from entropy_neuron_gemma2 import entropy_of_logits
    return {"unit": unit, "diff_of_means": diff_of_means, "cosine": cosine,
            "split_indices": split_indices, "frac_moved": frac_moved,
            "_proj_edit_hook": _proj_edit_hook, "_M": _M,
            "quantile_split": quantile_split, "_signal": _signal,
            "entropy_of_logits": entropy_of_logits}


# ----------------------------------------------------------------- pure gate arithmetic
def gate_fraction(M_intervened, M_counter, M_neutral):
    """Fraction of the cave SUPPRESSED by an intervention on the counter residual:
        (M_intervened - M_counter) / (M_neutral - M_counter).
    +1 == the intervention moves the caved margin all the way back to the neutral (uncaved) margin; 0 ==
    no movement; negative == the intervention deepens the cave (pushes M below M_counter). The denominator
    is the per-item R-4-controlled cave gap (M_neutral - M_counter); a ~0 gap returns 0 (no div-by-zero).
    Pure (floats -> float)."""
    gap = M_neutral - M_counter
    if abs(gap) < 1e-9:
        return 0.0
    return (M_intervened - M_counter) / gap


# ----------------------------------------------------------------- pure decision
def decide_gate(gate_up, gate_rand, gate_thr=GATE_THR, base_floor=BASE_FLOOR):
    """CONFIDENCE_GATES_CAVING iff raising confidence suppresses at least gate_thr of the cave AND the
    matched random-direction gate-effect is clean (|gate_rand| < base_floor); else NO_GATE. Pure.
    A None gate_up (no measurable cave / degenerate) -> NO_GATE."""
    clean = gate_rand is None or abs(gate_rand) < base_floor
    fires = gate_up is not None and gate_up >= gate_thr and clean
    return "CONFIDENCE_GATES_CAVING" if fires else "NO_GATE"


def model_decision(gate_up, gate_down, gate_rand, headline_layer, headline_source):
    """Assemble the neutral gate decision for one model. Pure (floats -> dict). No claim attached."""
    cat = decide_gate(gate_up, gate_rand)
    return {
        "gate_bucket": cat,
        "gates_caving": cat == "CONFIDENCE_GATES_CAVING",
        "gate_up": (round(gate_up, 4) if gate_up is not None else None),
        "gate_down": (round(gate_down, 4) if gate_down is not None else None),
        "gate_rand": (round(gate_rand, 4) if gate_rand is not None else None),
        "headline_layer": headline_layer,
        "headline_source": headline_source,
    }


# ----------------------------------------------------------------- residual + signal collection (real)
def _collect(model, pool, device, is_chat, fit_layers, rname, P):
    """One model: per pool item, cache at every fit layer the last-token resid_post under the NEUTRAL and
    COUNTER prompts (verbatim repo turn construction), the signed neutral and counter margins
    M = logp(C)-logp(W*), and the next-token ENTROPY of the answer-position distribution under the neutral
    prompt. Forward-only; first-token-collision items (cid==aid -> margin meaningless) skipped. This mirrors
    confidence_direction_causal._collect exactly, kept local so this control imports no run-time-only state."""
    from rlhf_differential import _helpers, _logp_diff
    from job_truthful_flip import PUSH, NEUTRAL
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    tag = "it" if is_chat else "base"
    recs = []
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        if cid == aid:                                   # first-token collision -> margin meaningless, skip
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        rn, rc = {}, {}

        def grab_n(r, hook, _rn=rn):
            _rn[hook.layer()] = r[0, -1].detach().float(); return r

        def grab_c(r, hook, _rc=rc):
            _rc[hook.layer()] = r[0, -1].detach().float(); return r

        names = [rname(L) for L in fit_layers]
        with torch.no_grad():
            lg_neu = model.run_with_hooks(neutral, fwd_hooks=[(n, grab_n) for n in names])
            M_neu = float(_logp_diff(lg_neu, cid, aid))
            ent_neu = float(P["entropy_of_logits"](lg_neu[0, -1]))     # answer-position next-token entropy
            M_ctr = float(_logp_diff(
                model.run_with_hooks(counter, fwd_hooks=[(n, grab_c) for n in names]), cid, aid))
        recs.append({"i": i, "q": q, "cid": cid, "aid": aid,
                     "neutral": neutral, "counter": counter,
                     "rn": rn, "rc": rc, "M_neu": M_neu, "M_ctr": M_ctr, "ent_neu": ent_neu})
        print(f"  [{tag}] item {i} M_neu={M_neu:+.2f} ent_neu={ent_neu:.3f} M_ctr={M_ctr:+.2f} "
              f"q={q[:40]!r}", flush=True)
    return recs


def _measure_model(name, is_chat, device, fit_layers, rname, pool, P):
    """One model end-to-end. Collect NEUTRAL/COUNTER residuals + margins + neutral entropy on the WIDE pool.
    Per layer: (a) on a TRAIN fold over ALL items fit u_conf = entropy-quartile diff-of-means confidence
    direction at NEUTRAL and record proj_hi/proj_lo (the TRAIN high-/low-confidence mean projections) and the
    held-out ENTROPY-necessity of u_conf (used only to PICK the headline layer); (b) on the held-out CAVING
    items in the COUNTER condition, steer u_conf UP (to proj_hi) and DOWN (to proj_lo) and read the signed
    cave metric M, forming gate_up / gate_down vs the per-item cave gap; (c) a matched-magnitude random
    direction with the identical steer-up edit. Returns a dict."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import MIN_EFFECT_NET
    unit, diff_of_means = P["unit"], P["diff_of_means"]
    split_indices, frac_moved = P["split_indices"], P["frac_moved"]
    proj_edit, _M = P["_proj_edit_hook"], P["_M"]
    quantile_split, _signal = P["quantile_split"], P["_signal"]

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    recs = _collect(model, pool, device, is_chat, fit_layers, rname, P)
    n = len(recs)
    g = torch.Generator(device="cpu").manual_seed(RAND_SEED)

    # one shared train/test split over ALL items (same SPLIT_SEED as the sibling controls) for the u_conf
    # fit; the entropy-quartile contrast is taken WITHIN each fold.
    tr, te = split_indices(n, SPLIT_SEED)

    # confidence SIGNAL (ENTROPY_QUARTILE; -entropy so larger == more confident), as in
    # confidence_direction_causal._signal, computed once per item at the NEUTRAL condition.
    sig = [_signal(r, DEFINITION) for r in recs]

    # caving items (this model): counter lowers the margin from neutral by >= MIN_EFFECT_NET. Split into a
    # TRAIN/TEST fold so the steer is read on items DISJOINT from the (all-item) u_conf fit's emphasis and
    # reported held-out; the gate is measured on the cave TEST fold.
    cave_pos = [k for k, r in enumerate(recs) if (r["M_neu"] - r["M_ctr"]) >= MIN_EFFECT_NET]
    cave_te = []
    if cave_pos:
        _ctr_tr, ctr_te = split_indices(len(cave_pos), SPLIT_SEED)
        cave_te = [cave_pos[j] for j in ctr_te]

    out = {"name": name, "n_ok": n, "n_cave": len(cave_pos), "n_cave_test": len(cave_te),
           "n_train": len(tr), "n_test": len(te), "layers": {}}

    for L in fit_layers:
        # ---- u_conf: entropy-quartile diff-of-means at NEUTRAL on the TRAIN fold ----
        sig_tr = [sig[k] for k in tr]
        hi_tr_local, lo_tr_local = quantile_split(sig_tr, QUANTILE)
        hi_tr = [tr[j] for j in hi_tr_local]
        lo_tr = [tr[j] for j in lo_tr_local]
        entry = {"n_train_hi": len(hi_tr), "n_train_lo": len(lo_tr)}
        if not (hi_tr and lo_tr):
            entry["skipped"] = "degenerate entropy-quartile train split"
            out["layers"][L] = entry
            continue
        Rn_hi = torch.stack([recs[k]["rn"][L] for k in hi_tr]).to(device)
        Rn_lo = torch.stack([recs[k]["rn"][L] for k in lo_tr]).to(device)
        u_conf = unit(diff_of_means(Rn_hi, Rn_lo))
        proj_hi = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in hi_tr)
        proj_lo = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_conf) for k in lo_tr)

        # matched-magnitude random unit direction (cpu generator -> residual device/dtype); its OWN TRAIN
        # high-confidence mean projection is the steer-up target for the matched random control.
        rnd = torch.randn(u_conf.shape, generator=g).to(u_conf.dtype).to(device)
        u_rand = unit(rnd)
        prj_hi_rand = statistics.mean(float(recs[k]["rn"][L].to(device) @ u_rand) for k in hi_tr)

        # ---- held-out ENTROPY-necessity of u_conf (only used to SELECT the headline layer; same
        #      construction as confidence_direction_causal: on TEST high-confidence items set the
        #      u_conf-projection to the TRAIN low mean -> fraction of the (high-low) entropy-signal gap that
        #      collapses). This re-reads the confidence signal, NOT the cave. ----
        sig_te = [sig[k] for k in te]
        hi_te_local, lo_te_local = quantile_split(sig_te, QUANTILE)
        hi_te = [te[j] for j in hi_te_local]
        sig_lo_tr = statistics.mean(_signal(recs[k], DEFINITION) for k in lo_tr)
        ent_nec_vals = []
        for k in hi_te:
            r = recs[k]
            sig_hi = _signal(r, DEFINITION)
            h = [(rname(L), proj_edit(u_conf, proj_lo))]
            with torch.no_grad():
                lg = model.run_with_hooks(r["neutral"], fwd_hooks=h)
            sig_ab = -float(P["entropy_of_logits"](lg[0, -1]))   # ENTROPY_QUARTILE readout (matches _signal)
            ent_nec_vals.append(frac_moved(sig_hi, sig_ab, sig_lo_tr))
        entry["ent_necessity"] = round(statistics.mean(ent_nec_vals), 4) if ent_nec_vals else None
        entry["n_ent_test_hi"] = len(hi_te)

        # ---- GATE: steer u_conf UP / DOWN on the COUNTER residual of held-out caving TEST items; read M ----
        gu_vals, gd_vals, gr_vals = [], [], []
        for k in cave_te:
            r = recs[k]
            cid, aid = r["cid"], r["aid"]
            M_ctr, M_neu = r["M_ctr"], r["M_neu"]
            # STEER UP: u_conf-projection on counter -> TRAIN high-confidence mean
            hu = [(rname(L), proj_edit(u_conf, proj_hi))]
            M_up = _M(model, r["counter"], cid, aid, hooks=hu)
            gu_vals.append(gate_fraction(M_up, M_ctr, M_neu))
            # STEER DOWN: u_conf-projection on counter -> TRAIN low-confidence mean
            hd = [(rname(L), proj_edit(u_conf, proj_lo))]
            M_dn = _M(model, r["counter"], cid, aid, hooks=hd)
            gd_vals.append(gate_fraction(M_dn, M_ctr, M_neu))
            # RANDOM matched steer-up: random direction's projection on counter -> its own TRAIN high mean
            hr = [(rname(L), proj_edit(u_rand, prj_hi_rand))]
            M_rr = _M(model, r["counter"], cid, aid, hooks=hr)
            gr_vals.append(gate_fraction(M_rr, M_ctr, M_neu))

        entry.update({
            "gate_up": round(statistics.mean(gu_vals), 4) if gu_vals else None,
            "gate_down": round(statistics.mean(gd_vals), 4) if gd_vals else None,
            "gate_rand": round(statistics.mean(gr_vals), 4) if gr_vals else None,
            "n_gate_eval": len(gu_vals),
            "proj_hi": round(proj_hi, 4), "proj_lo": round(proj_lo, 4),
        })
        out["layers"][L] = entry
        print(f"  [{('it' if is_chat else 'base')} L{L}] ent_nec={entry['ent_necessity']} "
              f"gate_up={entry['gate_up']} gate_down={entry['gate_down']} "
              f"gate_rand={entry['gate_rand']} (n_gate={entry['n_gate_eval']})", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- headline layer: max held-out entropy-necessity (most causal for confidence) when available,
    #      else max |gate_up|; both candidates recorded so the choice is documented. ----
    def lyr(L, key):
        return out["layers"].get(L, {}).get(key)

    ent_valid = [L for L in fit_layers if lyr(L, "ent_necessity") is not None]
    gate_valid = [L for L in fit_layers if lyr(L, "gate_up") is not None]
    L_by_ent = max(ent_valid, key=lambda L: lyr(L, "ent_necessity")) if ent_valid else None
    L_by_gate = max(gate_valid, key=lambda L: abs(lyr(L, "gate_up"))) if gate_valid else None

    if L_by_ent is not None and lyr(L_by_ent, "gate_up") is not None:
        headline_L, source = L_by_ent, "entropy_necessity"
    elif L_by_gate is not None:
        headline_L, source = L_by_gate, "max_abs_gate_up"
    else:
        headline_L, source = None, "none"

    out["headline_layer_by_entropy_necessity"] = L_by_ent
    out["headline_layer_by_abs_gate_up"] = L_by_gate
    if headline_L is not None:
        e = out["layers"][headline_L]
        out["decision"] = model_decision(e.get("gate_up"), e.get("gate_down"), e.get("gate_rand"),
                                          headline_L, source)
    else:
        out["decision"] = model_decision(None, None, None, None, source)
    return out


def run(name_base, name_it, tag, device, pool):
    from headset_direction import FIT_LAYERS as _FL, _rname   # reuse the layer sweep + resid_post hook name
    P = _pure()
    layers = list(_FL)
    res = {"base": _measure_model(name_base, False, device, layers, _rname, pool, P),
           "it": _measure_model(name_it, True, device, layers, _rname, pool, P)}
    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "confidence_caving_gate", "pool_size": len(pool),
        "definition": DEFINITION, "fit_layers": layers,
        "thresholds": {"GATE_THR": GATE_THR, "BASE_FLOOR": BASE_FLOOR, "QUANTILE": QUANTILE,
                       "SPLIT_SEED": SPLIT_SEED, "RAND_SEED": RAND_SEED},
        "decision_rule": DECISION_RULE,
        "base": res["base"], "it": res["it"],
    }
    Path("out").mkdir(exist_ok=True)
    fn = f"out/confidence_caving_gate_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    for m in MODELS:
        dd = res[m]["decision"]
        print(f"[{m}] {dd['gate_bucket']} (gate_up={dd['gate_up']} gate_down={dd['gate_down']} "
              f"rand={dd['gate_rand']}) headline=L{dd['headline_layer']} via {dd['headline_source']}",
              flush=True)
    print(f"[done] wrote {fn}", flush=True)


# ----------------------------------------------------------------- selftest (model-free, CPU, no model load)
def _plant(n, d, axis, levels, noise, gen):
    """Synthesize n residual rows carrying a planted AXIS at given per-item levels plus small isotropic
    noise: row[i] = levels[i]*axis + noise*randn. Returns [n, d]. Pure given the generator."""
    rows = noise * torch.randn(n, d, generator=gen)
    for i in range(n):
        rows[i] = rows[i] + levels[i] * axis
    return rows


def selftest():
    P = _pure()
    unit, diff_of_means, cosine = P["unit"], P["diff_of_means"], P["cosine"]
    quantile_split = P["quantile_split"]
    torch.manual_seed(0)

    # ---------- gate_fraction arithmetic ----------
    # cave: M_counter below M_neutral. An intervention that lifts M back to M_neutral suppresses 100%.
    assert abs(gate_fraction(2.0, -1.0, 2.0) - 1.0) < 1e-9          # all the way back to neutral -> 1.0
    assert abs(gate_fraction(-1.0, -1.0, 2.0) - 0.0) < 1e-9         # no movement -> 0.0
    assert abs(gate_fraction(0.5, -1.0, 2.0) - 0.5) < 1e-9          # halfway back -> 0.5
    assert gate_fraction(-2.0, -1.0, 2.0) < 0.0                     # below M_counter -> deepens -> negative
    assert gate_fraction(5.0, 2.0, 2.0) == 0.0                      # degenerate gap -> 0.0 (no div-by-zero)
    print("[selftest] gate_fraction: full=1.0 none=0.0 half=0.5 deepen<0 degenerate=0.0")

    # ---------- quartile contrast (reused) is sharp + disjoint ----------
    vals = [float(x) for x in range(12)]
    hi, lo = quantile_split(vals, QUANTILE)
    assert hi == [9, 10, 11] and lo == [0, 1, 2], (hi, lo)
    assert set(hi).isdisjoint(set(lo))
    print(f"[selftest] entropy-quartile split high={hi} low={lo} (sharp, disjoint)")

    d, n = 256, 40
    # ============================================================================================
    # (a) CAVING IS A DETERMINISTIC FUNCTION OF THE CONFIDENCE PROJECTION.
    #     The cave metric M on the counter residual is a (monotone) function of the residual's u_conf
    #     coordinate: M = M_counter + slope*(coord_uconf - coord_counter). Counter sits at a low u_conf
    #     coordinate (caved), neutral at a high one (uncaved). Steering the u_conf-projection UP to the
    #     high-confidence (proj_hi) level drives M to the neutral margin -> gate_up ~1. A random
    #     orthogonal-ish direction does not touch the u_conf coordinate -> gate_rand ~0. Pure arithmetic.
    # ============================================================================================
    g = torch.Generator().manual_seed(11)
    u_conf_true = unit(torch.randn(d, generator=g))
    # planted high/low CONFIDENCE residuals at NEUTRAL to fit u_conf from (the entropy-quartile contrast).
    levels = [(-1.0 + 2.0 * i / (n - 1)) * 3.0 for i in range(n)]
    rows = _plant(n, d, u_conf_true, levels, noise=0.05, gen=g)
    conf_coord = [float(rows[i] @ u_conf_true) for i in range(n)]
    hi_idx, lo_idx = quantile_split(conf_coord, QUANTILE)
    u_conf_fit = unit(diff_of_means(rows[hi_idx], rows[lo_idx]))
    assert abs(cosine(u_conf_fit, u_conf_true)) > 0.95, cosine(u_conf_fit, u_conf_true)
    proj_hi = statistics.mean(float(rows[k] @ u_conf_fit) for k in hi_idx)
    proj_lo = statistics.mean(float(rows[k] @ u_conf_fit) for k in lo_idx)

    # synthetic caved-item residual on the COUNTER condition: low u_conf coordinate (== proj_lo level).
    # M reads ONLY the u_conf coordinate: M(coord) = slope*coord. Counter coord = proj_lo (caved low M),
    # neutral coord = proj_hi (uncaved high M). gap = M(proj_hi) - M(proj_lo).
    slope = 1.7

    def M_of(coord):
        return slope * coord
    M_counter = M_of(proj_lo)
    M_neutral = M_of(proj_hi)
    # the counter residual carries the u_conf coordinate proj_lo; the proj-edit sets it to a target.
    counter_row = proj_lo * u_conf_fit + 0.05 * torch.randn(d, generator=g)
    # STEER UP: set u_conf coordinate to proj_hi -> M -> M(proj_hi) = M_neutral -> gate_up ~1
    coord_up = proj_hi
    M_up = M_of(coord_up)
    gate_up = gate_fraction(M_up, M_counter, M_neutral)
    # STEER DOWN: set u_conf coordinate to proj_lo (already there) -> M unchanged -> gate_down ~0
    M_dn = M_of(proj_lo)
    gate_down = gate_fraction(M_dn, M_counter, M_neutral)
    # RANDOM matched steer-up: a random unit direction, set its projection to its own high mean. Because
    # M reads only the u_conf coordinate and the random direction is ~orthogonal to u_conf, the edit moves
    # the u_conf coordinate by ~ (proj-change)*cos(u_rand,u_conf) ~ 0 -> M ~ M_counter -> gate_rand ~0.
    grand = torch.Generator().manual_seed(RAND_SEED)
    u_rand = unit(torch.randn(d, generator=grand))
    prj_hi_rand = statistics.mean(float(rows[k] @ u_rand) for k in hi_idx)
    cur_r = float(counter_row @ u_rand)
    edited = counter_row + (prj_hi_rand - cur_r) * u_rand          # the matched random steer-up edit
    coord_after_rand = float(edited @ u_conf_fit)                  # the resulting u_conf coordinate
    M_rr = M_of(coord_after_rand)
    gate_rand = gate_fraction(M_rr, M_counter, M_neutral)
    assert gate_up > 0.9, f"(a) steering confidence UP should suppress the cave: gate_up={gate_up}"
    assert abs(gate_down) < 0.2, f"(a) steering to the low (caved) level should not suppress: {gate_down}"
    assert abs(gate_rand) < BASE_FLOOR, f"(a) random matched steer must not move the cave: {gate_rand}"
    assert decide_gate(gate_up, gate_rand) == "CONFIDENCE_GATES_CAVING", (gate_up, gate_rand)
    print(f"[selftest] (a) cave==f(u_conf): gate_up={gate_up:.3f} gate_down={gate_down:.3f} "
          f"gate_rand={gate_rand:.3f} -> {decide_gate(gate_up, gate_rand)}")

    # ============================================================================================
    # (b) CAVING IS CARRIED BY AN ORTHOGONAL DIRECTION, INDEPENDENT OF u_conf.
    #     M reads a cave axis u_cave that is ORTHOGONAL to u_conf. Steering the u_conf-projection up moves
    #     the u_conf coordinate but NOT the u_cave coordinate -> M unchanged -> gate_up ~0 -> NO_GATE.
    #     (u_conf is still a genuine, recoverable confidence direction; it just does not drive the cave.)
    # ============================================================================================
    e_conf = torch.zeros(d); e_conf[0] = 1.0
    e_cave = torch.zeros(d); e_cave[1] = 1.0                       # orthogonal to e_conf
    # fit u_conf from a high/low confidence contrast along e_conf (recovers e_conf)
    gb = torch.Generator().manual_seed(5)
    lv = [3.0] * 10 + [-3.0] * 10
    rconf = _plant(20, d, e_conf, lv, noise=0.02, gen=gb)
    cc = [float(rconf[i] @ e_conf) for i in range(20)]
    hq, lq = quantile_split(cc, 0.5)
    u_conf_b = unit(diff_of_means(rconf[hq], rconf[lq]))
    assert abs(cosine(u_conf_b, e_conf)) > 0.99
    proj_hi_b = statistics.mean(float(rconf[k] @ u_conf_b) for k in hq)
    proj_lo_b = statistics.mean(float(rconf[k] @ u_conf_b) for k in lq)
    # M reads ONLY the cave coordinate (along e_cave), independent of u_conf.
    cave_slope = 2.0

    def M_cave(row):
        return cave_slope * float(row @ e_cave)
    # counter residual: caved (low cave coordinate), with some u_conf coordinate at the low level.
    counter_b = (-2.0) * e_cave + proj_lo_b * u_conf_b + 0.02 * torch.randn(d, generator=gb)
    neutral_cave_coord = +2.0                                     # neutral (uncaved) cave coordinate
    M_counter_b = M_cave(counter_b)
    M_neutral_b = cave_slope * neutral_cave_coord
    # STEER UP along u_conf: set u_conf coordinate to proj_hi_b. Because u_conf is orthogonal to e_cave,
    # the cave coordinate is untouched -> M unchanged -> gate_up ~0.
    cur_c = float(counter_b @ u_conf_b)
    steered = counter_b + (proj_hi_b - cur_c) * u_conf_b
    M_up_b = M_cave(steered)
    gate_up_b = gate_fraction(M_up_b, M_counter_b, M_neutral_b)
    assert abs(gate_up_b) < 0.05, f"(b) steering an orthogonal confidence axis must not move the cave: {gate_up_b}"
    assert decide_gate(gate_up_b, 0.0) == "NO_GATE", gate_up_b
    print(f"[selftest] (b) cave _|_ u_conf: gate_up={gate_up_b:.3f} -> {decide_gate(gate_up_b, 0.0)}")

    # ============================================================================================
    # (c) decide_gate boundary checks at GATE_THR / BASE_FLOOR
    # ============================================================================================
    assert decide_gate(0.55, 0.01) == "CONFIDENCE_GATES_CAVING"            # strong gate + clean random
    assert decide_gate(0.55, 0.30) == "NO_GATE"                           # dirty random (specificity)
    assert decide_gate(0.55, -0.30) == "NO_GATE"                          # |random| used (sign-agnostic)
    assert decide_gate(0.10, 0.01) == "NO_GATE"                           # gate below threshold
    assert decide_gate(GATE_THR, 0.0) == "CONFIDENCE_GATES_CAVING"        # exactly at GATE_THR -> fires
    assert decide_gate(GATE_THR - 1e-6, 0.0) == "NO_GATE"                 # just below -> NO_GATE
    assert decide_gate(0.55, BASE_FLOOR) == "NO_GATE"                     # random must be STRICTLY below
    assert decide_gate(0.55, BASE_FLOOR - 1e-6) == "CONFIDENCE_GATES_CAVING"
    assert decide_gate(None, 0.0) == "NO_GATE" and decide_gate(0.55, None) == "CONFIDENCE_GATES_CAVING"
    print("[selftest] (c) decide_gate thresholds exact (gate_up>=GATE_THR AND |gate_rand|<BASE_FLOOR)")

    # assembled per-model decisions
    md_a = model_decision(gate_up=0.62, gate_down=-0.04, gate_rand=0.01, headline_layer=28,
                          headline_source="entropy_necessity")
    assert md_a["gate_bucket"] == "CONFIDENCE_GATES_CAVING" and md_a["gates_caving"] \
        and md_a["headline_layer"] == 28 and md_a["headline_source"] == "entropy_necessity", md_a
    md_b = model_decision(gate_up=0.02, gate_down=0.0, gate_rand=0.0, headline_layer=32,
                          headline_source="max_abs_gate_up")
    assert md_b["gate_bucket"] == "NO_GATE" and not md_b["gates_caving"], md_b
    md_c = model_decision(gate_up=0.62, gate_down=0.0, gate_rand=0.20, headline_layer=24,
                          headline_source="entropy_necessity")
    assert md_c["gate_bucket"] == "NO_GATE", md_c                         # dirty random breaks the verdict
    md_n = model_decision(gate_up=None, gate_down=None, gate_rand=None, headline_layer=None,
                          headline_source="none")
    assert md_n["gate_bucket"] == "NO_GATE" and md_n["headline_layer"] is None, md_n
    print(f"[selftest] (c) assembled decisions: A={md_a['gate_bucket']} B={md_b['gate_bucket']} "
          f"C={md_c['gate_bucket']} N={md_n['gate_bucket']}")
    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name-base", default="google/gemma-2-9b")
    p.add_argument("--name-it", default="google/gemma-2-9b-it")
    p.add_argument("--tag", default="9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root for the item pool
        from misconception_pool import ITEMS_WIDE
        run(args.name_base, args.name_it, args.tag, args.device, ITEMS_WIDE)


if __name__ == "__main__":
    main()
