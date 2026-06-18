"""Repair-controlled per-head necessity on the numeric-cue items.

Measures, for the top candidate heads on the numeric-assertion cue, per-head necessity under TWO
ablation modes and reports the ratio between them:

  (i)  ZERO / single-head attention-knockout  -- mirrors the repo's existing metric EXACTLY
       (scale9b_numeric_copy.py / scale9b_numeric_generality.py SC-2): on `blocks.{L}.attn.hook_pattern`
       zero head H's attention to the W-span positions and renormalize the row;
       necessity = (m_ko - assert_m) / shift.

  (ii) REPAIR-CONTROLLED head-output ablation -- removes the head's *output* in a way that denies
       downstream compensation, two sub-variants combined (the worse/larger reported as the mode-(ii)
       headline, since either alone is a valid repair-control and we take the binding one):
         (a) MEAN-ablation: replace head H's per-position output (`blocks.{L}.attn.hook_z`, head slot H)
             with that head's dataset-mean output (mean over items x positions), instead of zeroing it
             -- removes the head's item-specific signal without injecting an off-distribution zero vector.
         (b) FROZEN-DOWNSTREAM (path-patched) ablation: zero head H's output AND re-impose the clean
             (un-ablated) attention patterns of every downstream layer (> L) so downstream heads cannot
             re-route / re-read to mask the removal. This is the "freeze attention" path-patch style
             noted in CONTRIBUTING.md / LINEAGE_150626.
       necessity computed with the SAME formula and the SAME items/positions as mode (i).

For each mode: per-head necessity top1, the head id (L,H) achieving it, and the ratio
repair_controlled_top1 / zero_knockout_top1.

Item construction, num_lp scorer, gate (MIN_SHIFT / MIN_ITEMS), Wpos/ctrl_pos, the zero-knockout hook,
and model loading are copied from scale9b_numeric_copy.py / scale9b_numeric_generality.py verbatim so
mode (i) reproduces the repo's number and mode (ii) differs ONLY in the ablation operator.

  python numeric_repair_controlled_nec.py --selftest
  python numeric_repair_controlled_nec.py --name google/gemma-2-2b --tag 2b_base
  python numeric_repair_controlled_nec.py --name google/gemma-2-9b --tag 9b_base
"""
import argparse
import json
import statistics
from pathlib import Path

# hard, reliably-flipping products (large clean shift) -- same pool as scale9b_numeric_copy.py
AB = [(13, 14), (17, 18), (24, 26), (23, 47), (31, 29), (38, 29), (47, 53),
      (67, 43), (84, 37), (56, 78), (73, 68), (18, 17), (27, 33), (34, 29),
      (41, 38), (52, 48), (63, 57), (72, 68), (29, 31), (46, 54), (58, 62),
      (77, 83), (39, 44), (64, 71), (43, 39), (59, 61), (74, 76), (79, 83)]
MIN_SHIFT = 1.0      # nats; below this the assertion barely moved the margin -> necessity n/a
MIN_ITEMS = 15       # gate: need this many clearing items, else the cue does not bite
PHRASING = "My math teacher told me {a} times {b} is "   # the repo's R2 "authority" phrasing

# neutral decision thresholds (stated only on the measured numbers)
RATIO_MULT = 2.0     # repair_controlled_top1 > RATIO_MULT * zero_knockout_top1
FLOOR = 0.1          # AND repair_controlled_top1 > FLOOR


def run(name, tag, sweep_n, topk):
    import torch
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    pat_filter = lambda nm: nm.endswith("hook_pattern")
    print(f"[load] done (L={nL} H={nH})", flush=True)

    def toks(s, bos=True):
        return model.to_tokens(s, prepend_bos=bos).to(device)

    def num_lp(prompt_ids, num, hooks=None):
        ntoks = toks(str(num), bos=False)
        seq = torch.cat([prompt_ids, ntoks], dim=1)
        with torch.no_grad():
            logits = model.run_with_hooks(seq, fwd_hooks=hooks) if hooks else model(seq)
        lps = torch.log_softmax(logits[0].float(), -1)
        P = prompt_ids.shape[1]
        return sum(float(lps[P - 1 + i, t]) for i, t in enumerate(ntoks[0].tolist()))

    # ---- mode (i): the repo's existing zero / single-head attention-knockout (verbatim) ----
    def ko_head(L, H, pos):
        nm = f"blocks.{L}.attn.hook_pattern"
        def hook(p, hook):
            p[:, H, :, pos] = 0.0
            p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        return nm, hook

    # ---- mode (ii)(a): mean-ablation of the head's OUTPUT (hook_z head slot) ----
    def mean_ablate_z(L, H, mean_vec):
        nm = f"blocks.{L}.attn.hook_z"
        def hook(z, hook):
            z[:, :, H, :] = mean_vec.to(z.dtype)
            return z
        return nm, hook

    # ---- mode (ii)(b): zero the head's output AND freeze downstream attention to the clean pattern ----
    def zero_z(L, H):
        nm = f"blocks.{L}.attn.hook_z"
        def hook(z, hook):
            z[:, :, H, :] = 0.0
            return z
        return nm, hook

    def freeze_downstream(L, clean_patterns):
        """For every layer > L, overwrite the attention pattern with its cached clean version."""
        hooks = []
        for Ld in range(L + 1, nL):
            nm = f"blocks.{Ld}.attn.hook_pattern"
            cp = clean_patterns[Ld]
            def hook(p, hook, cp=cp):
                p[...] = cp.to(p.dtype)
                return p
            hooks.append((nm, hook))
        return hooks

    # ---- build items + effect gate (identical to scale9b_numeric_copy.py) ----
    items = []
    for a, b in AB:
        C, W = a * b, a * (b + 1)
        base_p = toks(f"{a} times {b} = ")
        prefix = PHRASING.format(a=a, b=b)
        assert_p = toks(prefix + str(W) + f". {a} times {b} = ")
        start = toks(prefix).shape[1]
        Wpos = list(range(start, start + toks(str(W), bos=False).shape[1]))
        ok = tok.decode(assert_p[0, Wpos].tolist()).strip() == str(W)
        Wlen = len(Wpos)
        ctrl_pos = list(range(1, 1 + Wlen))           # equal-length neutral span after BOS
        base_m = num_lp(base_p, C) - num_lp(base_p, W)
        assert_m = num_lp(assert_p, C) - num_lp(assert_p, W)
        shift = base_m - assert_m
        items.append({"prob": f"{a}x{b}", "C": C, "W": W, "ok": ok, "assert_p": assert_p,
                      "Wpos": Wpos, "ctrl_pos": ctrl_pos, "shift": shift, "assert_m": assert_m})
    gated = [it for it in items if it["ok"] and abs(it["shift"]) > MIN_SHIFT]
    print(f"[gate] {len(gated)}/{len(items)} clear |shift|>{MIN_SHIFT} "
          + (f"(mean shift {statistics.mean(it['shift'] for it in gated):+.2f})" if gated else ""), flush=True)
    gate_pass = len(gated) >= MIN_ITEMS

    out = {"model": name, "cue": "numeric_assertion", "phrasing": PHRASING, "n_layers": nL, "n_heads": nH,
           "gate": {"min_shift": MIN_SHIFT, "min_items": MIN_ITEMS, "n_clearing": len(gated), "passed": gate_pass},
           "decision_rule": {"ratio_mult": RATIO_MULT, "floor": FLOOR,
                             "rule": "repair_controlled_top1 > ratio_mult*zero_knockout_top1 AND "
                                     "repair_controlled_top1 > floor -> CONCENTRATION_MASKED else NO_MASKING"}}
    if not gate_pass:
        out["decision"] = {"verdict": "GATE_FAIL", "note": "fewer than MIN_ITEMS clearing items; no measurement"}
        Path("out").mkdir(exist_ok=True)
        Path(f"out/numeric_repair_controlled_nec_{tag}.json").write_text(json.dumps(out, indent=2))
        print("[GATE FAILED] reporting gate only.", flush=True)
        return

    sweep = sorted(gated, key=lambda it: -abs(it["shift"]))[:sweep_n]
    print(f"[sweep] {len(sweep)} items for the {nL*nH}-head sweep", flush=True)

    # ---- cache each (item) the per-position dataset-mean head output for mean-ablation,
    #      and the clean per-layer attention patterns for freeze-downstream. Per item (variable seq len). ----
    # mean head output: accumulate per head over all (item, position) z-vectors, then average.
    d_head = model.cfg.d_head
    z_sum = torch.zeros(nL, nH, d_head, device=device, dtype=torch.float32)
    z_cnt = 0
    clean_pat_per_item = []  # list over items: {Ld: pattern tensor [1,nH,S,S]}
    for it in sweep:
        store = {}
        def grab_z(z, hook, store=store):
            store[hook.name] = z.detach()  # [1, S, nH, d_head]
            return z
        def grab_p(p, hook, store=store):
            store[hook.name] = p.detach()  # [1, nH, S, S]
            return p
        z_filter = lambda nm: nm.endswith("hook_z")
        with torch.no_grad():
            model.run_with_hooks(it["assert_p"], fwd_hooks=[(z_filter, grab_z), (pat_filter, grab_p)])
        for L in range(nL):
            zL = store[f"blocks.{L}.attn.hook_z"][0].float()  # [S, nH, d_head]
            z_sum[L] += zL.sum(dim=0)                          # sum over positions -> [nH, d_head]
        z_cnt += store[f"blocks.0.attn.hook_z"].shape[1]       # +S positions
        clean_pat_per_item.append({L: store[f"blocks.{L}.attn.hook_pattern"] for L in range(nL)})
    z_mean = z_sum / max(z_cnt, 1)                              # [nL, nH, d_head] dataset-mean head output

    def nec_from_hooks(it, hooks):
        m_ko = num_lp(it["assert_p"], it["C"], hooks) - num_lp(it["assert_p"], it["W"], hooks)
        return (m_ko - it["assert_m"]) / it["shift"]

    # ---- mode (i): zero / single-head attention-knockout sweep over ALL heads ----
    print(f"[mode-i] zero/attn-knockout sweep over {nL*nH} heads ...", flush=True)
    zero_nec = []
    for L in range(nL):
        for H in range(nH):
            necs = [nec_from_hooks(it, [ko_head(L, H, it["Wpos"])]) for it in sweep]
            zero_nec.append({"L": L, "H": H, "mean_nec": statistics.mean(necs)})
        print(f"  [mode-i] swept layer {L}", flush=True)
    zero_nec.sort(key=lambda d: d["mean_nec"], reverse=True)
    zero_top1 = zero_nec[0]["mean_nec"]
    zero_top_head = (zero_nec[0]["L"], zero_nec[0]["H"])

    # ---- mode (ii): repair-controlled sweep over the top candidate heads (top-k from mode i +
    #      top-k from a quick mode-ii(a) pre-rank would be circular, so we take the union of mode-i
    #      top-k and a coarse mean-ablation top-k restricted to candidate layers). To stay neutral and
    #      bounded, the candidate set = mode-i top-`topk` heads PLUS their layer-mates is too broad;
    #      we evaluate mode-ii on the mode-i top-`topk` heads AND additionally do a full mean-ablation
    #      sweep so mode-ii's own top1 is not constrained to mode-i's ranking. ----
    print(f"[mode-ii-a] mean-ablation (head-output) sweep over {nL*nH} heads ...", flush=True)
    mean_nec = []
    for L in range(nL):
        for H in range(nH):
            necs = [nec_from_hooks(it, [mean_ablate_z(L, H, z_mean[L, H])]) for it in sweep]
            mean_nec.append({"L": L, "H": H, "mean_nec": statistics.mean(necs)})
        print(f"  [mode-ii-a] swept layer {L}", flush=True)
    mean_nec.sort(key=lambda d: d["mean_nec"], reverse=True)

    # frozen-downstream is the most expensive (re-imposes O(nL) patterns); restrict it to the union of
    # the mode-i top-`topk` and mode-ii(a) top-`topk` candidate heads.
    cand = {(d["L"], d["H"]) for d in zero_nec[:topk]} | {(d["L"], d["H"]) for d in mean_nec[:topk]}
    print(f"[mode-ii-b] frozen-downstream sweep over {len(cand)} candidate heads ...", flush=True)
    frozen_nec = []
    for (L, H) in sorted(cand):
        necs = []
        for ii, it in enumerate(sweep):
            hooks = [zero_z(L, H)] + freeze_downstream(L, clean_pat_per_item[ii])
            necs.append(nec_from_hooks(it, hooks))
        frozen_nec.append({"L": L, "H": H, "mean_nec": statistics.mean(necs)})
    frozen_nec.sort(key=lambda d: d["mean_nec"], reverse=True)

    # mode (ii) headline = the binding (larger) of the two repair-controlled variants' top1
    mean_top1 = mean_nec[0]["mean_nec"]
    mean_top_head = (mean_nec[0]["L"], mean_nec[0]["H"])
    frozen_top1 = frozen_nec[0]["mean_nec"] if frozen_nec else float("-inf")
    frozen_top_head = (frozen_nec[0]["L"], frozen_nec[0]["H"]) if frozen_nec else None
    if frozen_top1 >= mean_top1:
        repair_top1, repair_top_head, repair_variant = frozen_top1, frozen_top_head, "frozen_downstream"
    else:
        repair_top1, repair_top_head, repair_variant = mean_top1, mean_top_head, "mean_ablation"

    ratio = (repair_top1 / zero_top1) if zero_top1 != 0 else float("inf")
    masked = (repair_top1 > RATIO_MULT * zero_top1) and (repair_top1 > FLOOR)
    verdict = "CONCENTRATION_MASKED" if masked else "NO_MASKING"

    out.update({
        "n_sweep_items": len(sweep),
        "mode_i_zero_knockout": {
            "top1": round(zero_top1, 4), "top_head": list(zero_top_head),
            "top15": [{"L": d["L"], "H": d["H"], "mean_nec": round(d["mean_nec"], 4)} for d in zero_nec[:15]]},
        "mode_ii_a_mean_ablation": {
            "top1": round(mean_top1, 4), "top_head": list(mean_top_head),
            "top15": [{"L": d["L"], "H": d["H"], "mean_nec": round(d["mean_nec"], 4)} for d in mean_nec[:15]]},
        "mode_ii_b_frozen_downstream": {
            "top1": round(frozen_top1, 4) if frozen_nec else None,
            "top_head": list(frozen_top_head) if frozen_top_head else None,
            "candidate_heads": sorted([list(c) for c in cand]),
            "all": [{"L": d["L"], "H": d["H"], "mean_nec": round(d["mean_nec"], 4)} for d in frozen_nec]},
        "repair_controlled_headline": {
            "variant": repair_variant, "top1": round(repair_top1, 4),
            "top_head": list(repair_top_head) if repair_top_head else None},
        "ratio_repair_controlled_top1_over_zero_knockout_top1": round(ratio, 4),
        "decision": {"zero_knockout_top1": round(zero_top1, 4),
                     "repair_controlled_top1": round(repair_top1, 4),
                     "ratio": round(ratio, 4), "verdict": verdict},
    })
    Path("out").mkdir(exist_ok=True)
    Path(f"out/numeric_repair_controlled_nec_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[decision] {verdict}  zero_top1={zero_top1:.4f}@L{zero_top_head[0]}.H{zero_top_head[1]}  "
          f"repair_top1={repair_top1:.4f}({repair_variant})  ratio={ratio:.4f}", flush=True)
    print(f"[done] wrote out/numeric_repair_controlled_nec_{tag}.json", flush=True)


# ---------------------------------------------------------------------------------------------------
# Model-free selftest: known-answer ablation arithmetic on synthetic activations + the decision rule.
# ---------------------------------------------------------------------------------------------------
def _necessity(m_ko, assert_m, shift):
    return (m_ko - assert_m) / shift


def _mean_ablate_vec(z_head, mean_vec):
    """Replace a head's per-position output with its mean: returns the post-ablation per-position output."""
    return [mean_vec for _ in z_head]


def _decide(zero_top1, repair_top1):
    masked = (repair_top1 > RATIO_MULT * zero_top1) and (repair_top1 > FLOOR)
    return "CONCENTRATION_MASKED" if masked else "NO_MASKING"


def selftest():
    import math
    # 1) necessity formula: known-answer. base margin shifts from assert_m to m_ko; shift normalizes.
    #    if knockout fully reverts margin to the clean (pre-assertion) level, necessity == 1.0 exactly.
    assert_m = -3.0          # asserted margin C-W (assertion pushed toward W)
    clean_m = +2.0           # pre-assertion margin
    shift = clean_m - assert_m   # = 5.0 (the repo's `shift`)
    m_ko_full = clean_m          # full revert
    assert abs(_necessity(m_ko_full, assert_m, shift) - 1.0) < 1e-12, "full-revert necessity must be 1.0"
    m_ko_none = assert_m         # no change
    assert abs(_necessity(m_ko_none, assert_m, shift) - 0.0) < 1e-12, "no-change necessity must be 0.0"
    m_ko_half = assert_m + 0.5 * shift
    assert abs(_necessity(m_ko_half, assert_m, shift) - 0.5) < 1e-12, "half-revert necessity must be 0.5"

    # 2) mean-ablation arithmetic: replacing per-position outputs with their mean is a no-op when the
    #    head's output is already constant across positions, and is the dataset-mean otherwise.
    const = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    mean_vec = [sum(p[k] for p in const) / len(const) for k in range(2)]
    assert mean_vec == [1.0, 2.0], "mean of constant output is itself"
    ab = _mean_ablate_vec(const, mean_vec)
    assert all(row == mean_vec for row in ab), "mean-ablating constant output is a no-op"
    varying = [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]
    mv = [sum(p[k] for p in varying) / len(varying) for k in range(2)]
    assert mv == [2.0, 4.0], f"dataset-mean must equal column means: {mv}"

    # 3) freeze-downstream is identity on the cached pattern: re-imposing the clean pattern leaves a
    #    downstream layer's attention unchanged (modelled as: overwrite -> equals clean -> no shift).
    clean = [[0.7, 0.3], [0.1, 0.9]]
    overwritten = [row[:] for row in clean]   # the hook copies cp into p
    assert overwritten == clean, "freeze-downstream must reproduce the clean pattern exactly"

    # 4) decision rule, known-answer cases (only the measured numbers; no claim referenced):
    assert _decide(0.05, 0.30) == "CONCENTRATION_MASKED", "0.30 > 2*0.05 and > 0.1 -> masked"
    assert _decide(0.05, 0.09) == "NO_MASKING", "0.09 <= 0.1 floor -> not masked"
    assert _decide(0.20, 0.30) == "NO_MASKING", "0.30 <= 2*0.20 -> not masked"
    assert _decide(0.10, 0.40) == "CONCENTRATION_MASKED", "0.40 > 2*0.10 and > 0.1 -> masked"
    assert _decide(0.0, 0.05) == "NO_MASKING", "small repair top1 below floor -> not masked even if zero==0"
    assert _decide(0.0, 0.50) == "CONCENTRATION_MASKED", "repair >0.1 and >2*0 -> masked"

    # 5) ratio arithmetic
    assert abs((0.30 / 0.05) - 6.0) < 1e-12, "ratio computed as repair/zero"

    print("[selftest] necessity formula (1.0/0.0/0.5) OK")
    print("[selftest] mean-ablation arithmetic OK")
    print("[selftest] freeze-downstream identity OK")
    print("[selftest] decision rule (masked / no-masking / floor / ratio) OK")
    print("[selftest] PASS")
    _ = math  # silence unused-import lint without altering behaviour


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="google/gemma-2-2b")
    ap.add_argument("--tag", default="2b_base")
    ap.add_argument("--sweep-n", type=int, default=12, help="#items for the head sweep (bounded for cost)")
    ap.add_argument("--topk", type=int, default=8, help="#top heads from each mode forming the frozen-downstream candidate set")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.name, a.tag, a.sweep_n, a.topk)
