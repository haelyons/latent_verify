"""ARC2A-2 -- firmer deletion-vs-masking: ARC2A re-run at n=12 with a restoration LADDER.

ARC2A (n=5) forced -it's L18.H5 alone to attend the anchor and restored ~11% of the
base<->it gap. But a single-head QK transplant is a weak probe: it can at most restore
L18.H5's own share (~16-20% in base, §3.10 / ARC2A base-sanity), and it cannot tell
"QK gated" apart from "the anchor VALUE representation also changed in -it". This run
disambiguates with a ladder of increasingly complete transplants, on identical prompts
(shared tokenizer => aligned positions):

  L0  nothing                         = it_baseline
  L1  single-head attn  (L18.H5)      = ARC2A's probe, now at n>=12
  L2  pool attn         (top-k base anchor-readers' rows transplanted)
  L3  single attn + anchor-VALUE patch (overwrite -it resid_pre[L] at anchor with base's,
                                        then force L18.H5 to attend the anchor)

  score = logp(cap) - logp(anchor);  LOW = anchor promoted = pull engaged
  restoration(T) = (it_baseline - T) / (it_baseline - base_baseline)

Pre-committed reading (which rung first reaches ~1 localizes the RLHF change):
  - L2 >> L1, L2 -> ~1            => MASKING: the OV pool is intact, RLHF gated WHERE the
                                     pool looks (QK). Distributed, not deleted.
  - L1,L2 low but L3 -> ~1        => the anchor VALUE the reader reads also changed in -it
                                     (upstream of the head); deeper than pure QK gating,
                                     still not OV deletion.
  - all rungs ~0                  => DELETION: the OV copy pathway itself degraded.

Controls (per pair): position-control row (anchor mass removed -> should NOT restore);
symmetric base sanity (impose -it's flat row on base -> should KILL base's pull);
OV-direction proxy base vs it (does L18.H5's OV still write toward the anchor token?).

  python job_arc2a2_transplant_pool.py    # base+it -> out/arc2a2_transplant_pool.json
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat"), ("Turkey", "Istanbul", "Ankara"),
         ("China", "Shanghai", "Beijing"), ("Spain", "Barcelona", "Madrid"),
         ("Italy", "Milan", "Rome"), ("Pakistan", "Karachi", "Islamabad"),
         ("Nigeria", "Lagos", "Abuja"), ("Florida", "Miami", "Tallahassee")]
STEM = "The capital of {r} is the city of"
READER = (18, 5)
POOL_K = 4                      # top base anchor-readers to transplant for the pool rung
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    print(f"[load] {name} on {DEVICE}", flush=True)
    m = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=DEVICE)
    m.eval()
    return m


def main():
    base = load("google/gemma-2-2b")
    it = load("google/gemma-2-2b-it")
    tok = base.tokenizer
    L, H = READER
    n_heads = base.cfg.n_heads
    pat = lambda l: f"blocks.{l}.attn.hook_pattern"
    ln_name = f"blocks.{L}.ln1.hook_normalized"
    resid_name = f"blocks.{L}.hook_resid_pre"
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]

    def score(model, ids, cid, aid, hooks=None):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)
        lp = torch.log_softmax(logits[0, -1].float(), -1)
        return float(lp[cid] - lp[aid])

    def anchor_pos(model, ids_list, anchor):
        aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in aset and i > 0]

    def all_rows(model, ids):
        """readout-query attention row for every head, by layer: {layer: [head, key]}."""
        store = {}
        def mk(l):
            def grab(pattern, hook):
                store[l] = pattern[0, :, -1, :].detach().float()
                return pattern
            return (pat(l), grab)
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[mk(l) for l in range(model.cfg.n_layers)])
        return store

    def transplant(rows_by_layer):
        """rows_by_layer: {layer: {head: row[key]}}; overwrite -it readout rows (renormed)."""
        hooks = []
        for l, hr in rows_by_layer.items():
            items = {h: (r / r.sum().clamp_min(1e-9)).to(DEVICE) for h, r in hr.items()}
            def hook(pattern, hook, items=items):
                for h, r in items.items():
                    pattern[0, h, -1, :r.shape[0]] = r
                return pattern
            hooks.append((pat(l), hook))
        return hooks

    def value_patch(base_resid, apos):
        bv = base_resid.to(DEVICE)
        def hook(resid, hook):                  # [b, pos, d_model]
            resid[0, apos, :] = bv[apos, :].to(resid.dtype)
            return resid
        return (resid_name, hook)

    def ov_pref(model, ids, apos, cid, aid):
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=lambda n: n == ln_name)
            normed = cache[ln_name][0, apos[-1]].float()
            out = (normed @ model.W_V[L, H].float()) @ model.W_O[L, H].float()
            out = out / out.norm().clamp_min(1e-9)
            WU = model.W_U.float()
            return float(out @ (WU[:, aid] - WU[:, cid]))
        except Exception as e:
            print(f"    [ov-proxy skipped] {e}", flush=True)
            return None

    rows = []
    for region, anchor, cap in PAIRS:
        cid, aid = first(" " + cap), first(" " + anchor)
        framed = f"{anchor} is the most famous city in {region}. " + STEM.format(r=region)
        b_ids = base.to_tokens(framed).to(DEVICE)
        i_ids = it.to_tokens(framed).to(DEVICE)
        b_ap = anchor_pos(base, b_ids[0].tolist(), anchor)
        i_ap = anchor_pos(it, i_ids[0].tolist(), anchor)
        if not (b_ap and i_ap):
            print(f"  [skip] {region}: no anchor positions", flush=True)
            continue

        b_rows = all_rows(base, b_ids)
        it_flat_row = all_rows(it, i_ids)[L][H]
        base_base = score(base, b_ids, cid, aid)
        it_base = score(it, i_ids, cid, aid)
        gap = it_base - base_base
        if abs(gap) < 1e-6:
            print(f"  [skip] {region}: zero gap", flush=True)
            continue

        # pick the base anchor-reader pool (top-K by attention-to-anchor at readout)
        anchor_attn = {(l, h): float(b_rows[l][h][b_ap].sum())
                       for l in range(base.cfg.n_layers) for h in range(n_heads)}
        pool = sorted(anchor_attn, key=anchor_attn.get, reverse=True)[:POOL_K]

        # base residual at the reader layer input (for the value patch)
        with torch.no_grad():
            _, bcache = base.run_with_cache(b_ids, names_filter=lambda n: n == resid_name)
        base_resid = bcache[resid_name][0].float()

        # ---- ladder ----
        L1 = score(it, i_ids, cid, aid, hooks=transplant({L: {H: b_rows[L][H]}}))
        pool_rows = defaultdict(dict)
        for (l, h) in pool:
            pool_rows[l][h] = b_rows[l][h]
        L2 = score(it, i_ids, cid, aid, hooks=transplant(pool_rows))
        L3 = score(it, i_ids, cid, aid,
                   hooks=[value_patch(base_resid, i_ap)] + transplant({L: {H: b_rows[L][H]}}))

        # position control: base L18.H5 row with anchor mass removed
        ctrl = b_rows[L][H].clone()
        ctrl[i_ap] = 0.0
        Lc = score(it, i_ids, cid, aid, hooks=transplant({L: {H: ctrl}}))

        base_killed = score(base, b_ids, cid, aid, hooks=transplant({L: {H: it_flat_row}}))
        restor = lambda T: (it_base - T) / gap

        rows.append({
            "pair": f"{region}->{cap}", "anchor": anchor,
            "base_score": base_base, "it_score": it_base,
            "pool_heads": [list(p) for p in pool],
            "it_single_attn": L1, "it_pool_attn": L2, "it_attn_plus_value": L3,
            "it_position_control": Lc, "base_killed_by_it_row": base_killed,
            "restoration_single": restor(L1), "restoration_pool": restor(L2),
            "restoration_attn_value": restor(L3), "restoration_control": restor(Lc),
            "ov_pref_base": ov_pref(base, b_ids, b_ap, cid, aid),
            "ov_pref_it": ov_pref(it, i_ids, i_ap, cid, aid),
        })
        print(f"  {region:<12} base={base_base:+.2f} it={it_base:+.2f} | "
              f"R_single={restor(L1):+.2f} R_pool={restor(L2):+.2f} "
              f"R_attn+val={restor(L3):+.2f} R_ctrl={restor(Lc):+.2f}", flush=True)

    def med(k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return statistics.median(xs) if xs else None

    r_single, r_pool, r_val = med("restoration_single"), med("restoration_pool"), med("restoration_attn_value")
    if (r_pool or 0) >= 0.6 and (r_pool or 0) - (r_single or 0) > 0.2:
        verdict = "MASKING (pool): OV pool intact, RLHF gated where the pool looks (QK)"
    elif (r_val or 0) >= 0.6 and (r_val or 0) - (r_pool or 0) > 0.2:
        verdict = "VALUE-CHANGE: the anchor representation the reader reads also changed in -it (upstream of the head)"
    elif max(r_single or 0, r_pool or 0, r_val or 0) < 0.2:
        verdict = "DELETION: no rung restores; OV copy pathway degraded"
    else:
        verdict = "PARTIAL: restoration intermediate; report the ladder, no clean corner"

    summary = {
        "reader_head": list(READER), "pool_k": POOL_K, "n": len(rows),
        "median_restoration_single": r_single,
        "median_restoration_pool": r_pool,
        "median_restoration_attn_value": r_val,
        "median_restoration_control": med("restoration_control"),
        "median_ov_pref_base": med("ov_pref_base"),
        "median_ov_pref_it": med("ov_pref_it"),
        "verdict": verdict,
        "reading": "ladder L1 single-attn < L2 pool-attn < L3 attn+value; first rung reaching ~1 localizes the RLHF change",
    }
    print("\n[summary]", json.dumps(summary, indent=2), flush=True)
    Path("out").mkdir(exist_ok=True)
    Path("out/arc2a2_transplant_pool.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("[done] wrote out/arc2a2_transplant_pool.json", flush=True)


if __name__ == "__main__":
    main()
