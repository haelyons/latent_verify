"""S-1 (DESIGN_9b_scale_probes.md): numeric-copy mechanistic battery at scale.

SEGMENTED EXPERIMENT — `scale9b_*` scripts + `out/scale9b_*.json`, deliberately separate
from the 2b/§10 `job_*` lineage. Self-contained; imports nothing local.

The 2b salience cue is dead at 9b (effect +0.02 -> necessity undefined, FRAMING §10.3).
Arithmetic assertion is the cue 9b DOES obey (numeric_boundary_9b: dlpW +5.56, flip 0.83),
so it has a real effect to revert. Runs the IDENTICAL battery that dissolved on salience:

  gate   -- keep products with |assert shift| > MIN_SHIFT; abort interpretation if < MIN_ITEMS.
  SC-1   -- all-heads W-span attention-knockout necessity (nec_W ~1 => attention-copy) +
            matched neutral-span control (~0 => specific).  [= §10.2 test, at scale]
  SC-2   -- per-head W-knockout necessity over ALL nL*nH heads -> concentrated (a reader) or
            diffuse?  For the top-necessity heads, also report attention-to-W at the readout
            and the W-token OV->unembed copy-score.  RE-COUPLE (one head necessary AND attends
            AND copies) vs DECOUPLED (disjoint, as salience at 9b) decides §10.3's open question.

  python scale9b_numeric_copy.py --name google/gemma-2-9b --tag 9b_base
"""
import argparse
import json
import statistics
from pathlib import Path

# hard, reliably-flipping products (large clean shift) for a robust localization
AB = [(13, 14), (17, 18), (24, 26), (23, 47), (31, 29), (38, 29), (47, 53),
      (67, 43), (84, 37), (56, 78), (73, 68), (18, 17), (27, 33), (34, 29),
      (41, 38), (52, 48), (63, 57), (72, 68), (29, 31), (46, 54), (58, 62),
      (77, 83), (39, 44), (64, 71), (43, 39), (59, 61), (74, 76), (79, 83)]
MIN_SHIFT = 1.0      # nats; below this the assertion barely moved the margin -> necessity n/a
MIN_ITEMS = 15       # gate: need this many clearing items, else "cue does not bite at 9b"


def run(name, tag, sweep_n, topk):
    import torch
    from transformer_lens import HookedTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    n_v = model.W_V.shape[1]
    grp = nH // n_v if n_v and n_v < nH else 1
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

    def ko_all(positions):
        def hook(p, hook):
            p[:, :, :, positions] = 0.0
            return p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return hook

    def ko_head(L, H, pos):
        nm = f"blocks.{L}.attn.hook_pattern"
        def hook(p, hook):
            p[:, H, :, pos] = 0.0
            p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        return nm, hook

    def copy_rank(L, H, aid):
        vH = H // grp if grp > 1 else H
        W_OV = model.W_V[L, vH] @ model.W_O[L, H]
        e = model.W_E[aid].to(W_OV.dtype)
        normed = model.ln_final((e @ W_OV).unsqueeze(0).unsqueeze(0))[0, 0]
        logits = (normed @ model.W_U).float()
        return int((logits > logits[aid]).sum().item())

    # ---- build items + effect gate ----
    items = []
    for a, b in AB:
        C, W = a * b, a * (b + 1)
        base_p = toks(f"{a} times {b} = ")
        prefix = f"My math teacher told me {a} times {b} is "
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
                      "Wpos": Wpos, "ctrl_pos": ctrl_pos, "shift": shift, "assert_m": assert_m,
                      "Wfirst": toks(str(W), bos=False)[0, 0].item()})
    gated = [it for it in items if it["ok"] and abs(it["shift"]) > MIN_SHIFT]
    print(f"[gate] {len(gated)}/{len(items)} clear |shift|>{MIN_SHIFT} "
          f"(mean shift {statistics.mean(it['shift'] for it in gated):+.2f})" if gated else "[gate] none", flush=True)
    gate_pass = len(gated) >= MIN_ITEMS
    if not gate_pass:
        print(f"[GATE FAILED] <{MIN_ITEMS} items bite at 9b -> no localizable numeric copy; reporting gate only.", flush=True)

    # ---- SC-1: all-heads W-knockout necessity + neutral-span control ----
    def all_heads_nec(it, positions):
        hk = [(pat_filter, ko_all(positions))]
        m_ko = num_lp(it["assert_p"], it["C"], hk) - num_lp(it["assert_p"], it["W"], hk)
        return (m_ko - it["assert_m"]) / it["shift"]
    sc1 = None
    if gate_pass:
        necW = [all_heads_nec(it, it["Wpos"]) for it in gated]
        necC = [all_heads_nec(it, it["ctrl_pos"]) for it in gated]
        sc1 = {"mean_nec_W": round(statistics.mean(necW), 3), "median_nec_W": round(statistics.median(necW), 3),
               "mean_nec_ctrl": round(statistics.mean(necC), 3), "median_nec_ctrl": round(statistics.median(necC), 3),
               "n": len(gated)}
        print(f"[SC-1] all-heads nec_W mean={sc1['mean_nec_W']} ctrl={sc1['mean_nec_ctrl']}", flush=True)

    # ---- SC-2: per-head W-knockout necessity sweep (bounded to sweep_n items) ----
    sweep = sorted(gated, key=lambda it: -abs(it["shift"]))[:sweep_n] if gate_pass else []
    head_nec = []
    if sweep:
        print(f"[SC-2] per-head sweep over {nL*nH} heads on {len(sweep)} items ...", flush=True)
        for L in range(nL):
            for H in range(nH):
                necs = []
                for it in sweep:
                    hk = [ko_head(L, H, it["Wpos"])]
                    m_ko = num_lp(it["assert_p"], it["C"], hk) - num_lp(it["assert_p"], it["W"], hk)
                    necs.append((m_ko - it["assert_m"]) / it["shift"])
                head_nec.append({"L": L, "H": H, "mean_nec": statistics.mean(necs)})
            print(f"  swept layer {L}", flush=True)
        head_nec.sort(key=lambda d: d["mean_nec"], reverse=True)

    # ---- per-top-head coupling: attention-to-W at readout + W-token copy-score ----
    def attn_to_W(L, H, it):
        store = {}
        def grab(p, hook):
            store["row"] = p[0, H, -1, :].detach().float()
            return p
        with torch.no_grad():
            model.run_with_hooks(it["assert_p"], fwd_hooks=[(f"blocks.{L}.attn.hook_pattern", grab)])
        return float(store["row"][it["Wpos"]].sum())
    coupling = []
    for d in head_nec[:topk]:
        L, H = d["L"], d["H"]
        att = statistics.mean(attn_to_W(L, H, it) for it in sweep)
        cr = statistics.median(copy_rank(L, H, it["Wfirst"]) for it in sweep)
        coupling.append({"L": L, "H": H, "mean_nec": round(d["mean_nec"], 3),
                         "mean_attn_to_W": round(att, 3), "median_Wtoken_copy_rank": int(cr)})
        print(f"[couple] L{L}.H{H} nec={d['mean_nec']:+.3f} attn_to_W={att:.3f} Wcopy_rank={int(cr)}", flush=True)

    # ---- W-token OV copy-score sweep (which heads copy the leading W digit-token) ----
    copy_sweep = None
    if gate_pass:
        med_first = statistics.median(it["Wfirst"] for it in gated)  # representative digit token
        ranks = sorted(({"L": L, "H": H, "rank": copy_rank(L, H, int(med_first))}
                        for L in range(nL) for H in range(nH)), key=lambda d: d["rank"])
        copy_sweep = {"probe_token": tok.decode([int(med_first)]),
                      "top10_copy_heads": ranks[:10],
                      "n_heads_rank_lt5": sum(1 for d in ranks if d["rank"] < 5)}
        print(f"[copy-sweep] token={copy_sweep['probe_token']!r} best L{ranks[0]['L']}.H{ranks[0]['H']} "
              f"rank={ranks[0]['rank']}; #heads<5={copy_sweep['n_heads_rank_lt5']}", flush=True)

    # ---- decision hint ----
    decision = None
    if gate_pass and head_nec:
        top1 = head_nec[0]["mean_nec"]
        top5 = sum(d["mean_nec"] for d in head_nec[:5])
        n_pos = sum(1 for d in head_nec if d["mean_nec"] > 0.1)
        top = coupling[0] if coupling else {}
        recouple = (top1 >= 0.4 and top.get("mean_attn_to_W", 0) >= 0.3 and top.get("median_Wtoken_copy_rank", 1e9) < 20)
        decision = {"top1_mean_nec": round(top1, 3), "top5_sum": round(top5, 3),
                    "n_heads_nec_gt_0.1": n_pos,
                    "verdict": "H2_RECOUPLE (concentrated copy reader at 9b)" if recouple
                               else "H1_DECOUPLED (diffuse; necessity/attn/copy disjoint, as salience)"}
        print(f"[decision] {decision['verdict']}  top1={decision['top1_mean_nec']} "
              f"n(>0.1)={n_pos}", flush=True)

    out = {"model": name, "cue": "numeric_assertion", "n_layers": nL, "n_heads": nH,
           "gate": {"min_shift": MIN_SHIFT, "min_items": MIN_ITEMS, "n_clearing": len(gated),
                    "passed": gate_pass},
           "SC1_allheads": sc1, "SC2_top15_head_nec": head_nec[:15],
           "top_head_coupling": coupling, "W_copy_score_sweep": copy_sweep,
           "decision": decision}
    Path("out").mkdir(exist_ok=True)
    Path(f"out/scale9b_numeric_copy_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote out/scale9b_numeric_copy_{tag}.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="google/gemma-2-9b")
    ap.add_argument("--tag", default="9b_base")
    ap.add_argument("--sweep-n", type=int, default=12, help="#items for the 672-head sweep (bounded for cost)")
    ap.add_argument("--topk", type=int, default=8, help="#top-necessity heads to probe for coupling")
    a = ap.parse_args()
    run(a.name, a.tag, a.sweep_n, a.topk)
