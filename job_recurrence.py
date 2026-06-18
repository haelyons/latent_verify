"""N-1 recurrence check: is the base-model copy NAME-MOVER (content-routed) or
INDUCTION (token-recurrence-driven)?  (SEQUENCE_170626; POSITION_SYCOPHANCY
sec Terminology->Resolved.)

This is a committed, reproducible reimplementation of the orphaned out/recurrence_2b.json.
Acceptance gate (2b only): it must reproduce the faithfulness numbers (reader L18.H5 ->
Sydney ~0.836; generic induction score ~0.192) and the D2 conclusion (salience >>
induction-decoy on all pairs).

SCALE PORT (--reader auto): the 2b reader L18.H5 does NOT transfer to 9b; the reader is
re-localized per model (max attn-to-anchor at the readout on the Australia salience prompt),
and the faithfulness "expected" numbers are 2b-specific (printed only when reader == (18,5)).
Everything downstream reads the localized head.

Three discriminators, all read the reader head's attention at the prediction slot:

  faith  -- reproduce the head's signature: reader->Sydney on the framed Australia prompt,
            and a GENERIC induction score on a repeated-random sequence (does the head
            attend to the token *after* a key's previous occurrence?). The generic score
            measures head CHARACTER; the task discriminators below measure what it actually
            USES in context.

  D1     -- on the plain framed prompt, split the reader's prediction-slot attention into
            attn(anchor) vs attn(region). CONFOUND: the region token is simultaneously the
            task-relevant token AND the induction prefix-match continuation of " of".

  D2     -- DE-CONFOUND. Prepend a minimal clause so a neutral decoy is the literal
            induction prefix-match continuation; split attention {anchor=salience,
            decoy=induction, region=task}.

  numeric -- the asserted-number cue: confirm the asserted W is NOT prefix-reachable at the
            "= " prediction slot, and report the single head with max attention onto W.

  python job_recurrence.py                                              # -> out/recurrence_2b_repro.json
  python job_recurrence.py --name google/gemma-2-9b --tag 9b --reader auto
"""
import argparse
import json
from pathlib import Path

DECOY = "Genoa"  # neutral non-region city, plausible "the city of ___" completion
PAIRS = [
    {"region": "Australia",   "anchor": "Sydney"},
    {"region": "Texas",       "anchor": "Houston"},
    {"region": "Canada",      "anchor": "Toronto"},
    {"region": "Switzerland", "anchor": "Zurich"},
    {"region": "Florida",     "anchor": "Miami"},
]
FRAMED = "{anchor} is the most famous city in {region}. The capital of {region} is the city of"
D2_PREFIX = "Think of {decoy}. "
NUMERIC = "My math teacher told me 7 times 8 is 54. 7 times 8 = "


def run(name, tag, reader_arg):
    import torch
    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {name} on {device} (fp32)", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.float32, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH = model.cfg.n_layers, model.cfg.n_heads

    def positions_of(ids_list, word):
        tset = set(model.to_tokens(word, prepend_bos=False)[0].tolist())
        tset |= set(model.to_tokens(" " + word, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in tset and i > 0]

    # ---- localize reader (max attn-to-anchor at readout on Australia salience) if auto ----
    if reader_arg == "auto":
        ids = model.to_tokens(FRAMED.format(anchor="Sydney", region="Australia"))
        apos = positions_of(ids[0].tolist(), "Sydney")
        cache = {}
        def grab_loc(p, hook):
            cache[hook.name] = p[0, :, -1, :].detach().float()
            return p
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(lambda nm: nm.endswith("hook_pattern"), grab_loc)])
        best, best_lh = -1.0, None
        for L in range(nL):
            attn = cache[f"blocks.{L}.attn.hook_pattern"][:, apos].sum(-1)
            h = int(attn.argmax())
            if float(attn[h]) > best:
                best, best_lh = float(attn[h]), (L, h)
        READER = best_lh
        print(f"[auto-reader] L{READER[0]}.H{READER[1]} max attn->Sydney={best:.3f}", flush=True)
    else:
        READER = tuple(reader_arg)
    L, H = READER
    pat = f"blocks.{L}.attn.hook_pattern"

    def pred_row(text):
        ids = model.to_tokens(text)
        store = {}
        def grab(p, hook):
            store["row"] = p[0, H, -1, :].detach().float()
            return p
        with torch.no_grad():
            model.run_with_hooks(ids, fwd_hooks=[(pat, grab)])
        dec = [tok.decode([t]) for t in ids[0].tolist()]
        return store["row"], dec, ids[0].tolist()

    def attn_to(row, positions):
        return float(row[positions].sum()) if positions else 0.0

    def top_keys(row, dec, k=4):
        t = torch.topk(row, min(k, row.shape[0]))
        return [{"pos": int(i), "tok": dec[int(i)], "w": round(float(w), 4)}
                for w, i in zip(t.values, t.indices)]

    def prefix_reachable(ids_list, target_positions):
        cur = ids_list[-1]
        if not tok.decode([cur]).strip():
            return False
        tgt = set(target_positions)
        for i in range(len(ids_list) - 1):
            if ids_list[i] == cur and (i + 1) in tgt:
                return True
        return False

    is_2b_reader = (READER == (18, 5))
    out = {"model": name, "dtype": "float32", "reader": list(READER),
           "reader_source": reader_arg if reader_arg != "auto" else "auto-localized"}

    # ---- faith: reader->Sydney + generic induction score (repeated random) ----
    row, dec, ids_list = pred_row(FRAMED.format(anchor="Sydney", region="Australia"))
    syd = attn_to(row, positions_of(ids_list, "Sydney"))
    Lr = 25
    torch.manual_seed(0)
    vocab = model.cfg.d_vocab
    rand = torch.randint(1000, min(vocab, 50000), (Lr,))
    bos = torch.tensor([[tok.bos_token_id]])
    seq = torch.cat([bos, rand.unsqueeze(0), rand.unsqueeze(0)], dim=1)
    store = {}
    def grab_full(p, hook):
        store["p"] = p[0, H].detach().float()  # [q, k]
        return p
    with torch.no_grad():
        model.run_with_hooks(seq, fwd_hooks=[(pat, grab_full)])
    p = store["p"]
    ind = sum(float(p[1 + Lr + j, 1 + j + 1]) for j in range(Lr - 1)) / (Lr - 1)
    out["faithfulness"] = {"reader_to_sydney": round(syd, 4),
                           "expected_2b": 0.836 if is_2b_reader else None,
                           "induction_score": round(ind, 4),
                           "expected_ind_2b": 0.192 if is_2b_reader else None,
                           "n_layers": nL, "n_heads": nH}
    print(f"[faith] L{L}.H{H}->Sydney={syd:.3f}{' (exp 0.836)' if is_2b_reader else ''}  "
          f"induction={ind:.3f}{' (exp 0.192)' if is_2b_reader else ''}  layers={nL} heads={nH}", flush=True)

    # ---- D1: anchor vs region (region == induction-continuation confound) ----
    out["d1"] = []
    for pr in PAIRS:
        r, a = pr["region"], pr["anchor"]
        row, dec, ids_list = pred_row(FRAMED.format(anchor=a, region=r))
        apos, rpos = positions_of(ids_list, a), positions_of(ids_list, r)
        aa, ar = attn_to(row, apos), attn_to(row, rpos)
        reach = prefix_reachable(ids_list, apos)
        verdict = "anchor-dominant" if aa >= ar else "region-dominant"
        out["d1"].append({"region": r, "anchor": a, "attn_anchor_namemover": round(aa, 4),
                          "attn_region_indcont_CONFOUNDED": round(ar, 4),
                          "anchor_over_region": round(aa / ar, 2) if ar > 1e-6 else None,
                          "anchor_prefix_reachable": reach, "verdict": verdict,
                          "top_keys": top_keys(row, dec)})
        print(f"[D1] {r:<12} anchor({a})={aa:.3f}  region/indcont={ar:.3f}  "
              f"-> {verdict} (anchor prefix-reachable: {reach})", flush=True)

    # ---- D2: de-confound -- decoy is the genuine induction prefix-match continuation ----
    out["d2_deconfound"] = []
    for pr in PAIRS:
        r, a = pr["region"], pr["anchor"]
        text = D2_PREFIX.format(region=r, decoy=DECOY) + FRAMED.format(anchor=a, region=r)
        row, dec, ids_list = pred_row(text)
        apos = positions_of(ids_list, a)
        dpos = positions_of(ids_list, DECOY)
        rpos = positions_of(ids_list, r)
        a_s, d_i, r_t = attn_to(row, apos), attn_to(row, dpos), attn_to(row, rpos)
        winner = max((("anchor/salience", a_s), ("decoy/induction", d_i),
                      ("region/task", r_t)), key=lambda kv: kv[1])[0]
        out["d2_deconfound"].append(
            {"region": r, "anchor": a, "decoy": DECOY,
             "attn_anchor_salience": round(a_s, 4), "attn_decoy_induction": round(d_i, 4),
             "attn_region_task": round(r_t, 4),
             "decoy_prefix_reachable": prefix_reachable(ids_list, dpos),
             "winner": winner, "top_keys": top_keys(row, dec)})
        print(f"[D2] {r:<12} salience({a})={a_s:.3f}  induction(decoy {DECOY})={d_i:.3f}  "
              f"task(region)={r_t:.3f}  -> WINNER: {winner}", flush=True)

    # ---- numeric: assert W not prefix-reachable; max head onto W ----
    ids = model.to_tokens(NUMERIC)
    ids_list = ids[0].tolist()
    dec = [tok.decode([t]) for t in ids_list]
    wpos = [i for i, d in enumerate(dec) if d.strip() in ("5", "4") and i > 0][:2]
    store = {}
    def grab_all(p, hook):
        lyr = int(hook.name.split(".")[1])
        store.setdefault("rows", {})[lyr] = p[0, :, -1, :].detach().float()  # [head, k]
        return p
    names = [f"blocks.{l}.attn.hook_pattern" for l in range(nL)]
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(nm, grab_all) for nm in names])
    best = {"layer": None, "head": None, "attn": -1.0}
    for l, rows in store["rows"].items():
        for h in range(nH):
            v = float(rows[h][wpos].sum()) if wpos else 0.0
            if v > best["attn"]:
                best = {"layer": int(l), "head": int(h), "attn": round(v, 4)}
    out["numeric"] = {"prompt": NUMERIC, "asserted_W_positions": wpos,
                      "W_prefix_reachable": prefix_reachable(ids_list, wpos),
                      "max_head_onto_W": best,
                      "note": "W='54' follows 'is'; current token before prediction is '= ', "
                              "so '54' is NOT the prefix-match continuation -> not induction-reachable."}
    print(f"[num] max head onto W: L{best['layer']}.H{best['head']} attn={best['attn']}  "
          f"W prefix-reachable? {out['numeric']['W_prefix_reachable']}", flush=True)

    Path("out").mkdir(exist_ok=True)
    dest = f"out/recurrence_{tag}.json"
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {dest}")


def _parse_reader(vals):
    if len(vals) == 1 and vals[0] == "auto":
        return "auto"
    if len(vals) == 2:
        return [int(vals[0]), int(vals[1])]
    raise SystemExit("--reader takes 'auto' or two ints (L H)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="google/gemma-2-2b")
    ap.add_argument("--tag", default="2b_repro")  # avoid clobbering the orphan out/recurrence_2b.json
    ap.add_argument("--reader", nargs="+", default=["18", "5"], help="'auto' or 'L H' (default 18 5 = 2b)")
    args = ap.parse_args()
    run(args.name, args.tag, _parse_reader(args.reader))


if __name__ == "__main__":
    main()
