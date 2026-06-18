"""Per-head necessity NULL baseline for the sc2 single-head-ablation metric (numeric-cue items).

Companion control to scale9b_numeric_copy.py / scale9b_numeric_generality.py. Those scripts
sweep per-head W-span attention-knockout necessity (sc2) over all nL*nH heads and report a top-1.
This control answers the prior question that any top-1 needs: what does the sc2 metric read on a
SINGLE head when the cue carries no information about the asserted answer?

NULL construction (cue decoupled from the asserted answer). For each gated numeric item we keep the
SAME math problem ("{a} times {b}") and the SAME readout ("{a} times {b} = "), but replace the
asserted number in the prefix with a SHUFFLED cue: the distractor W taken from a DIFFERENT item
(deterministic derangement of the gated list). The asserted token is therefore a number that is
neither this item's correct product C nor its own distractor W, so the cue cannot move this item's
C-vs-W margin in any systematic, head-localizable way. Running the IDENTICAL single-head-ablation
sc2 metric on this decoupled prompt, over all heads, yields a null distribution of per-head
necessity values (the spread the metric produces from noise + renormalization alone). We report its
95th percentile, mean, and sd, and the null top-1 across heads.

Separately, on the UNSHUFFLED (real) numeric items we report the per-head sc2 necessity of a NAMED
head (--head L,H; default the head the repo names as the numeric SC-2 top-1, L34.H14) and the real
observed top-1 across all heads, so each can be placed against the null.

Decision (measured numbers only): if observed_top1 (or named_head_nec) is below null_95pct ->
"AT_NOISE_FLOOR", else "ABOVE_FLOOR".

  python controls/perhead_nec_null.py --selftest
  python controls/perhead_nec_null.py --name google/gemma-2-9b --tag 9b_base
  python controls/perhead_nec_null.py --name google/gemma-2-2b --tag 2b --head 18,5
"""
import argparse
import json
import statistics
from pathlib import Path

# Same products as scale9b_numeric_copy.py (hard, reliably-flipping; large clean shift).
AB = [(13, 14), (17, 18), (24, 26), (23, 47), (31, 29), (38, 29), (47, 53),
      (67, 43), (84, 37), (56, 78), (73, 68), (18, 17), (27, 33), (34, 29),
      (41, 38), (52, 48), (63, 57), (72, 68), (29, 31), (46, 54), (58, 62),
      (77, 83), (39, 44), (64, 71), (43, 39), (59, 61), (74, 76), (79, 83)]
MIN_SHIFT = 1.0      # nats; matches scale9b_numeric_copy gate
MIN_ITEMS = 15       # gate: need this many clearing items
DEFAULT_HEAD = (34, 14)   # repo-named numeric SC-2 top-1 head at 9b (FRAMING: L34.H14 mean-nec 0.127)
PCTL = 95            # null percentile reported / used by the decision rule


def derange(n):
    """Deterministic derangement of range(n): index i -> a different index (no fixed point).
    For n>=2, the cyclic shift by 1 (i -> (i+1) % n) is a fixed-point-free permutation."""
    if n < 2:
        return list(range(n))
    return [(i + 1) % n for i in range(n)]


def percentile(vals, p):
    """Nearest-rank percentile on a copy-sorted list (no numpy dependency)."""
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def run(name, tag, sweep_n, head):
    import torch
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
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

    def ko_head(L, H, pos):
        nm = f"blocks.{L}.attn.hook_pattern"
        def hook(p, hook):
            p[:, H, :, pos] = 0.0
            p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        return nm, hook

    def build_item(a, b, asserted):
        """Build one item asserting `asserted` for problem a*b. Returns dict or None if mis-tokenized.
        C = correct product, W = own distractor a*(b+1); `asserted` is what the prefix claims."""
        C, W = a * b, a * (b + 1)
        base_p = toks(f"{a} times {b} = ")
        prefix = f"My math teacher told me {a} times {b} is "
        assert_p = toks(prefix + str(asserted) + f". {a} times {b} = ")
        start = toks(prefix).shape[1]
        Apos = list(range(start, start + toks(str(asserted), bos=False).shape[1]))
        ok = tok.decode(assert_p[0, Apos].tolist()).strip() == str(asserted)
        base_m = num_lp(base_p, C) - num_lp(base_p, W)
        assert_m = num_lp(assert_p, C) - num_lp(assert_p, W)
        shift = base_m - assert_m
        return {"prob": f"{a}x{b}", "a": a, "b": b, "C": C, "W": W, "asserted": asserted,
                "ok": ok, "assert_p": assert_p, "Apos": Apos, "shift": shift, "assert_m": assert_m}

    def head_nec(it, L, H):
        """sc2 metric for a single head: fraction of the C-vs-W margin shift reverted by zeroing
        head (L,H)'s attention to the asserted-number span (then renormalizing). Same formula as
        scale9b_numeric_copy.py ko_head sweep."""
        hk = [ko_head(L, H, it["Apos"])]
        m_ko = num_lp(it["assert_p"], it["C"], hk) - num_lp(it["assert_p"], it["W"], hk)
        return (m_ko - it["assert_m"]) / it["shift"]

    # ---- REAL items: assert each item's own distractor W (gate exactly as numeric_copy) ----
    real = []
    for a, b in AB:
        it = build_item(a, b, a * (b + 1))
        real.append(it)
    real_gated = [it for it in real if it["ok"] and abs(it["shift"]) > MIN_SHIFT]
    print(f"[gate] real: {len(real_gated)}/{len(real)} clear |shift|>{MIN_SHIFT}", flush=True)
    gate_pass = len(real_gated) >= MIN_ITEMS
    if not gate_pass:
        print(f"[GATE FAILED] <{MIN_ITEMS} real items clear; null/observed not localizable.", flush=True)

    sweep = sorted(real_gated, key=lambda it: -abs(it["shift"]))[:sweep_n] if gate_pass else []

    # ---- NULL items: same problems, but assert a SHUFFLED cue = another item's distractor W ----
    # Deterministic derangement over the gated list so the asserted number is decoupled from C/W.
    perm = derange(len(real_gated))
    null_items = []
    for i, it in enumerate(real_gated):
        donor = real_gated[perm[i]]
        shuffled_assert = donor["W"]
        # avoid an accidental coupling if the donor's W equals this item's own C or W
        if shuffled_assert in (it["C"], it["W"]):
            continue
        ni = build_item(it["a"], it["b"], shuffled_assert)
        if ni["ok"] and abs(ni["shift"]) > MIN_SHIFT:
            null_items.append(ni)
    null_sweep = sorted(null_items, key=lambda it: -abs(it["shift"]))[:sweep_n]
    print(f"[null] {len(null_sweep)} decoupled items for the null sweep", flush=True)

    # ---- NULL distribution: per-head mean necessity over all heads on the decoupled items ----
    null_vals = []
    if null_sweep:
        print(f"[null] per-head sweep over {nL*nH} heads on {len(null_sweep)} decoupled items ...", flush=True)
        for L in range(nL):
            for H in range(nH):
                necs = [head_nec(it, L, H) for it in null_sweep]
                null_vals.append(statistics.mean(necs))
            print(f"  [null] swept layer {L}", flush=True)

    null_95pct = round(percentile(null_vals, PCTL), 4) if null_vals else None
    null_mean = round(statistics.mean(null_vals), 4) if null_vals else None
    null_sd = round(statistics.pstdev(null_vals), 4) if len(null_vals) > 1 else None

    # ---- REAL observed: named head + top-1 over all heads ----
    named_head = list(head)
    named_head_nec = None
    observed_top1 = None
    if sweep:
        nl, nh = head
        if 0 <= nl < nL and 0 <= nh < nH:
            named_head_nec = round(statistics.mean(head_nec(it, nl, nh) for it in sweep), 4)
        else:
            print(f"[warn] named head {head} out of range for L={nL} H={nH}; named_head_nec=None", flush=True)
        print(f"[real] per-head sweep over {nL*nH} heads on {len(sweep)} real items ...", flush=True)
        best = None
        for L in range(nL):
            for H in range(nH):
                mn = statistics.mean(head_nec(it, L, H) for it in sweep)
                if best is None or mn > best["mean_nec"]:
                    best = {"L": L, "H": H, "mean_nec": mn}
            print(f"  [real] swept layer {L}", flush=True)
        observed_top1 = {"L": best["L"], "H": best["H"], "mean_nec": round(best["mean_nec"], 4)}

    # ---- NEUTRAL decision (measured numbers only) ----
    decision = None
    if null_95pct is not None and (observed_top1 is not None or named_head_nec is not None):
        compare_val = observed_top1["mean_nec"] if observed_top1 is not None else named_head_nec
        decision = "AT_NOISE_FLOOR" if compare_val < null_95pct else "ABOVE_FLOOR"
        print(f"[decision] compare={compare_val} vs null_{PCTL}pct={null_95pct} -> {decision}", flush=True)

    out = {"model": name, "cue": "numeric_assertion_perhead_nec_null", "n_layers": nL, "n_heads": nH,
           "metric": "sc2_single_head_W_attn_knockout_necessity",
           "gate": {"min_shift": MIN_SHIFT, "min_items": MIN_ITEMS,
                    "n_real_clearing": len(real_gated), "passed": gate_pass},
           "n_null_items": len(null_sweep), "n_real_sweep_items": len(sweep),
           "null_pctl": PCTL,
           "null_95pct": null_95pct, "null_mean": null_mean, "null_sd": null_sd,
           "named_head": named_head, "named_head_nec": named_head_nec,
           "observed_top1": observed_top1,
           "decision": decision}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/perhead_nec_null_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote out/perhead_nec_null_{tag}.json", flush=True)


def selftest():
    """Model-free: derangement is fixed-point-free, percentile mechanics, and the decision-rule
    threshold on synthetic null/observed numbers (no model load)."""
    # derangement: no index maps to itself, and it is a permutation
    for n in range(2, 30):
        d = derange(n)
        assert sorted(d) == list(range(n)), f"derange({n}) not a permutation: {d}"
        assert all(d[i] != i for i in range(n)), f"derange({n}) has a fixed point: {d}"
    assert derange(0) == [] and derange(1) == [0], "trivial sizes"
    print("[selftest] derangement fixed-point-free + permutation OK")

    # percentile: nearest/linear-interp sanity
    assert percentile([1, 2, 3, 4, 5], 0) == 1
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert abs(percentile([0, 10], 95) - 9.5) < 1e-9, percentile([0, 10], 95)
    assert percentile([], 95) is None
    print("[selftest] percentile mechanics OK")

    # decision rule: tight null around 0, an observed top1 clearly above -> ABOVE_FLOOR
    null_vals = [0.0 + 0.01 * ((i % 7) - 3) for i in range(200)]  # mean ~0, small spread
    n95 = percentile(null_vals, PCTL)
    observed_high = 0.5
    observed_low = n95 - 0.001
    dec_high = "AT_NOISE_FLOOR" if observed_high < n95 else "ABOVE_FLOOR"
    dec_low = "AT_NOISE_FLOOR" if observed_low < n95 else "ABOVE_FLOOR"
    assert dec_high == "ABOVE_FLOOR", (observed_high, n95)
    assert dec_low == "AT_NOISE_FLOOR", (observed_low, n95)
    print(f"[selftest] decision: null_{PCTL}pct={round(n95,4)} obs_high={observed_high}->{dec_high} "
          f"obs_low={round(observed_low,4)}->{dec_low}")

    # a noise-floor observed exactly at the null spread -> AT_NOISE_FLOOR
    diffuse_observed = percentile(null_vals, 90)  # below the 95th pct
    assert diffuse_observed < n95, (diffuse_observed, n95)
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="google/gemma-2-9b")
    ap.add_argument("--tag", default="9b_base")
    ap.add_argument("--sweep-n", type=int, default=12, help="#items for the per-head sweeps (bounded for cost)")
    ap.add_argument("--head", default=f"{DEFAULT_HEAD[0]},{DEFAULT_HEAD[1]}",
                    help="named head 'L,H' to report real per-head necessity for (default repo SC-2 top-1)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        L, H = (int(x) for x in a.head.split(","))
        run(a.name, a.tag, a.sweep_n, (L, H))
