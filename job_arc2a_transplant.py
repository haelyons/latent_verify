"""ARC2A -- salience copy: did RLHF DELETE the copy, or just stop the head LOOKING?

sec 8 showed L18.H5's attention to the anchor collapses 0.84 -> 0.01 in -it on the
identical prompt, and called it "removed from the weights". But sec 8 only measured
the attention pattern (QK). Low attention != absent capability. If the OV copy
machinery survived and RLHF only changed *where the head looks* (QK gating), then
FORCING -it's L18.H5 to attend to the anchor should route the anchor's value through
the (intact) OV and restore the pull. If the OV pathway itself degraded, it won't.

Core intervention (within-item, paired): on the identical framed prompt,
  - read base L18.H5's readout-query attention row (peaks on the anchor),
  - overwrite -it L18.H5's readout-query row with it (renormalized), run, read margin.

  score = logp(cap) - logp(anchor)   (LOW score = anchor promoted = pull engaged)
  restoration_frac = (it_baseline - it_transplant) / (it_baseline - base_baseline)
      ~1  => forcing the look restores the pull  => OV intact, RLHF gated QK (MASKING)
      ~0  => pull stays dead                     => OV pathway degraded   (DELETION)

Controls / sanities:
  - symmetric: impose -it's (flat) L18.H5 row onto BASE -> should KILL base's pull
    (confirms the transplant op works and L18.H5's look is necessary in base).
  - position control: base row with the anchor mass removed -> should NOT restore
    (isolates attending-to-the-anchor specifically, not "any change to the row").
  - OV-direction proxy (QK-free): does L18.H5's OV write toward the anchor token in
    unembed space, base vs it? (LN-free direction readout; guarded.)

  python job_arc2a_transplant.py    # base+it held together -> out/arc2a_transplant.json
"""
import json
import statistics
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
         ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
         ("Morocco", "Casablanca", "Rabat")]
STEM = "The capital of {r} is the city of"
READER = (18, 5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    print(f"[load] {name} on {DEVICE}", flush=True)
    m = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=DEVICE)
    m.eval()
    return m


def main():
    base = load("google/gemma-2-2b")
    it = load("google/gemma-2-2b-it")
    tok = base.tokenizer                         # shared tokenizer family
    L, H = READER
    pat_name = f"blocks.{L}.attn.hook_pattern"
    ln_name = f"blocks.{L}.ln1.hook_normalized"
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]

    def score(model, ids, cid, aid, hooks=None):
        with torch.no_grad():
            logits = model.run_with_hooks(ids, fwd_hooks=hooks) if hooks else model(ids)
        lp = torch.log_softmax(logits[0, -1].float(), -1)
        return float(lp[cid] - lp[aid])

    def anchor_pos(model, ids_list, anchor):
        aset = set(model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in aset and i > 0]

    def grab_reader_row(model, ids):
        store = {}
        def grab(pattern, hook):
            store["row"] = pattern[0, H, -1, :].detach().clone()    # [k]
            return pattern
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(pat_name, grab)])
        return store["row"]

    def transplant_hook(row):
        row = (row / row.sum().clamp_min(1e-9)).to(DEVICE)
        def hook(pattern, hook):                  # [b, head, q, k]
            pattern[0, H, -1, :row.shape[0]] = row
            return pattern
        return hook

    def ov_direction_pref(model, ids, apos, cid, aid):
        """LN-free: does L18.H5's OV write toward the anchor token vs the capital?"""
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(ids, names_filter=lambda n: n == ln_name)
            normed = cache[ln_name][0, apos[-1]].float()            # [d_model] at last anchor tok
            v = normed @ model.W_V[L, H].float()                    # [d_head]
            out = v @ model.W_O[L, H].float()                       # [d_model]
            out = out / out.norm().clamp_min(1e-9)
            WU = model.W_U.float()
            return float(out @ (WU[:, aid] - WU[:, cid]))           # >0 => toward anchor
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

        base_row = grab_reader_row(base, b_ids)                     # peaks on anchor
        it_row = grab_reader_row(it, i_ids)                         # ~flat / off-anchor
        base_base = score(base, b_ids, cid, aid)
        it_base = score(it, i_ids, cid, aid)

        # treatment: force -it L18.H5 to look like base
        it_T = score(it, i_ids, cid, aid, hooks=[(pat_name, transplant_hook(base_row))])
        gap = (it_base - base_base)
        restoration = (it_base - it_T) / gap if abs(gap) > 1e-6 else None

        # position control: base row with anchor mass removed
        ctrl_row = base_row.clone()
        if i_ap:
            ctrl_row[i_ap] = 0.0
        it_C = score(it, i_ids, cid, aid, hooks=[(pat_name, transplant_hook(ctrl_row))])
        restoration_ctrl = (it_base - it_C) / gap if abs(gap) > 1e-6 else None

        # symmetric sanity: kill base's pull by imposing -it's flat row
        base_killed = score(base, b_ids, cid, aid, hooks=[(pat_name, transplant_hook(it_row))])

        ov_base = ov_direction_pref(base, b_ids, b_ap, cid, aid)
        ov_it = ov_direction_pref(it, i_ids, i_ap, cid, aid)

        rows.append({
            "pair": f"{region}->{cap}", "anchor": anchor,
            "base_score": base_base, "it_score": it_base, "it_transplant": it_T,
            "it_position_control": it_C, "base_killed_by_it_row": base_killed,
            "restoration_frac": restoration, "restoration_frac_control": restoration_ctrl,
            "ov_pref_base": ov_base, "ov_pref_it": ov_it,
        })
        rf = f"{restoration:+.2f}" if restoration is not None else "n/a"
        print(f"  {region:<12} base={base_base:+.2f} it={it_base:+.2f} it+transplant={it_T:+.2f} "
              f"| restoration={rf} | base_killed={base_killed:+.2f} "
              f"| OV base={ov_base} it={ov_it}", flush=True)

    def col(k):
        return [r[k] for r in rows if r[k] is not None]
    summary = {
        "reader_head": list(READER),
        "median_restoration_frac": (statistics.median(col("restoration_frac")) if col("restoration_frac") else None),
        "median_restoration_frac_control": (statistics.median(col("restoration_frac_control")) if col("restoration_frac_control") else None),
        "median_ov_pref_base": (statistics.median(col("ov_pref_base")) if col("ov_pref_base") else None),
        "median_ov_pref_it": (statistics.median(col("ov_pref_it")) if col("ov_pref_it") else None),
        "n": len(rows),
        "reading": "restoration~1 => MASKING (OV intact, QK gated); restoration~0 => DELETION",
    }
    print("\n[summary]", json.dumps(summary, indent=2), flush=True)
    Path("out").mkdir(exist_ok=True)
    Path("out/arc2a_transplant.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("[done] wrote out/arc2a_transplant.json", flush=True)


if __name__ == "__main__":
    main()
