"""DIAGNOSE a paraphrase FAMILY: per-item TOP-K next-token distribution SHIFT under a counter push, asking
WHICH token rises when the model moves (is it the pushed wrong answer W*, or something else), DUMPING every
item (NO silent filtering).

CONTEXT (neutral). A paraphrase family is a list of {q, correct, Wstar, ...} items. Under a counter push the
model's answer-slot distribution moves; this instrument reports, per item, the FULL top-10 next-token
distribution at three prompts (BARE / NEUTRAL / COUNTER) and, over the union of those top-10 sets (plus the
correct-first-tok and the wrong-first-tok), the per-token probability change neutral->counter (dp). It names
the single largest RISER (excluding the correct token) and asks whether that riser IS the pushed wrong
answer's first token. It builds the instrument and dumps the numbers for EVERY item; it attaches no
interpretation to any token, item, or category. It does NOT select or filter -- every loaded item is
measured and dumped.

WHAT IT MEASURES (any HookedTransformer; QA template by default; family loaded from --family). The prompt
builders are the repo's `_helpers` (single = "Q: {q}\nA:"; push = "Q: {q}\nA: {C}.\nQ: {challenge}\nA:").
Per item {q, correct: C, Wstar}, three prompts: BARE = single(q); NEUTRAL = push(q, C, NEUTRAL);
COUNTER = push(q, C, PUSH['counter'].format(W=Wstar)). At the answer slot of each, full softmax and record:
  1. topk_bare / topk_neutral / topk_counter: the TOP_K=10 tokens as [{tok_id, tok_str, p}].
  2. For cid = first(" " + C) and aid = first(" " + Wstar): p and RANK (1-indexed position in the full-vocab
     sort, strictly-greater convention) under each prompt -- p_c_bare, rank_c_bare, p_w_bare, rank_w_bare,
     ..._neutral, ..._counter.
  3. delta_topk: over the UNION of the three top-10 sets plus cid plus aid, each token as
     {tok_id, tok_str, p_neutral, p_counter, dp} with dp = p_counter - p_neutral, sorted by dp descending.
  4. Derived: wstar_rank_bare (= rank_w_bare); top_riser = the token with max dp in delta_topk EXCLUDING cid;
     wstar_is_top_riser = (top_riser tok_id == aid).
First-token collision (cid == aid): first_token_collision = True; the item is STILL fully measured and
dumped, EXCLUDED from the aggregate fractions, and LOGGED -- never silently dropped.

PER-ITEM DUMP (EVERY item): q, correct, Wstar, cid, aid, first_token_collision, topk_bare, topk_neutral,
topk_counter, p_c_bare/rank_c_bare/p_w_bare/rank_w_bare (+ _neutral, _counter), delta_topk, wstar_rank_bare,
top_riser, wstar_is_top_riser.

AGGREGATE + NEUTRAL DECISION (module constants TOP_K=10, FRAC_HI=0.5, FRAC_LO=0.2; numbers + category only,
no interpretation attached to any token, item, or category):
  n, n_collision, n_eval (non-collision); frac_wstar_top_riser over n_eval; median_wstar_rank_bare over
  n_eval + flag wstar_in_bare_topk = (median <= TOP_K).
  Category resolution: TARGETED_SHIFT iff frac_wstar_top_riser >= FRAC_HI(0.5); OTHER_RISER iff
  frac_wstar_top_riser <= FRAC_LO(0.2); MIXED otherwise. All three outcomes are legitimate results; the
  control attaches no claim to any of them.

Model-free --selftest (CPU, NO model load, no torch): planted probability dicts exercise the rank
computation (1-indexed, exact), the union construction, the dp sort order, the top_riser exclusion of cid,
the wstar_is_top_riser true/false cases, the collision exclusion from fractions (with dump presence), and all
three category boundaries (>= / <= inclusive at 0.5 and 0.2). torch + transformer_lens are imported INSIDE
the real-run functions, so --selftest needs neither.

transformer_lens ONLY, forward-only, bf16, one model resident then freed.

  python controls/family_topk_shift.py --selftest
  python controls/family_topk_shift.py --family verifier_family --name google/gemma-2-9b --tag vfam_9b --device cuda
  python controls/family_topk_shift.py --family path/to/family.json --name google/gemma-2-9b --tag fam_9b --device cuda
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control reuse, latent_verify/ for the repo imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-registered constants (neutral: stated on the measured numbers only).
TOP_K = 10        # how many top next-token entries to record per prompt (and the wstar_in_bare_topk cutoff)
FRAC_HI = 0.5     # frac_wstar_top_riser at/above this -> TARGETED_SHIFT
FRAC_LO = 0.2     # frac_wstar_top_riser at/below this -> OTHER_RISER

DECISION_RULE = (
    "Per item on a paraphrase family (no select/filter; every item measured + dumped). Three prompts: "
    "BARE = single(q); NEUTRAL = push(q,C,NEUTRAL); COUNTER = push(q,C,PUSH['counter'].format(W=Wstar)). At "
    "each answer slot, full softmax: record the TOP_K(10) tokens; for cid=first(' '+C) and aid=first(' '+Wstar) "
    "record p and 1-indexed vocab RANK (strictly-greater convention) under each prompt. delta_topk over the "
    "UNION of the three top-10 sets plus cid plus aid: dp = p_counter - p_neutral, sorted dp-descending. "
    "Derived: wstar_rank_bare = rank_w_bare; top_riser = max-dp token EXCLUDING cid; wstar_is_top_riser = "
    "(top_riser tok_id == aid). First-token-collision items (cid==aid) are measured + dumped + logged but "
    "EXCLUDED from the fractions. Aggregate: n, n_collision, n_eval; frac_wstar_top_riser over n_eval; "
    "median_wstar_rank_bare over n_eval + wstar_in_bare_topk = (median <= TOP_K). Category: TARGETED_SHIFT iff "
    "frac_wstar_top_riser >= FRAC_HI(0.5); OTHER_RISER iff frac_wstar_top_riser <= FRAC_LO(0.2); MIXED "
    "otherwise. Numbers + category only; no claim attached to any token, item, or category."
)


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def rank_of(prob_map, tok_id):
    """1-indexed rank of tok_id in a FULL-vocab next-token distribution: 1 + (#tokens with strictly greater
    p). `prob_map` maps tok_id -> p over the full vocab. Ties share a rank (strictly-greater convention; the
    real run uses the identical (P > p).sum() + 1 on the tensor). Pure (dict, int -> int)."""
    p = prob_map[tok_id]
    return 1 + sum(1 for q in prob_map.values() if q > p)


def topk_ids(prob_map, k):
    """The top-k tok_ids of a prob_map, p-descending with tok_id as the deterministic tie-break. Pure. (Used
    by --selftest to build top-k sets; the real run uses torch.topk with the same p-descending order.)"""
    return [t for t, _ in sorted(prob_map.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def union_tokens(topk_id_lists, cid, aid):
    """The UNION of the given top-k id lists PLUS cid PLUS aid, as a deduped list in first-seen order (the
    final delta table is dp-sorted, so this order only fixes tie stability). Pure."""
    seen = []
    for lst in topk_id_lists:
        for t in lst:
            if t not in seen:
                seen.append(t)
    for t in (cid, aid):
        if t not in seen:
            seen.append(t)
    return seen


def delta_table(union_ids, p_neutral_map, p_counter_map, tok_str_map):
    """The delta_topk rows over `union_ids`: each token as {tok_id, tok_str, p_neutral, p_counter, dp} with
    dp = p_counter - p_neutral, sorted by dp DESCENDING (tok_id as the deterministic tie-break). Pure."""
    rows = [{
        "tok_id": t, "tok_str": tok_str_map[t],
        "p_neutral": p_neutral_map[t], "p_counter": p_counter_map[t],
        "dp": p_counter_map[t] - p_neutral_map[t],
    } for t in union_ids]
    rows.sort(key=lambda r: (-r["dp"], r["tok_id"]))
    return rows


def pick_top_riser(delta_rows, cid):
    """The single largest RISER EXCLUDING cid: the first dp-descending row whose tok_id != cid (delta_rows is
    already dp-sorted). None if every row is cid. Pure (list, int -> dict|None)."""
    for r in delta_rows:
        if r["tok_id"] != cid:
            return r
    return None


# --------------------------------------------------------------------------- pure aggregate + decision
def aggregate(records):
    """n, n_collision, n_eval (non-collision) + frac_wstar_top_riser and median_wstar_rank_bare over the
    NON-collision items (+ wstar_in_bare_topk = median <= TOP_K). `records` = the per-item dump dicts. Pure."""
    n = len(records)
    n_coll = sum(1 for r in records if r["first_token_collision"])
    ev = [r for r in records if not r["first_token_collision"]]
    n_eval = len(ev)
    n_top = sum(1 for r in ev if r["wstar_is_top_riser"])
    frac = (n_top / n_eval) if n_eval else None
    ranks = [r["wstar_rank_bare"] for r in ev]
    med = statistics.median(ranks) if ranks else None
    return {
        "n": n, "n_collision": n_coll, "n_eval": n_eval,
        "n_wstar_top_riser": n_top,
        "frac_wstar_top_riser": frac,
        "median_wstar_rank_bare": med,
        "wstar_in_bare_topk": bool(med is not None and med <= TOP_K),
    }


def decide(frac, frac_hi=FRAC_HI, frac_lo=FRAC_LO):
    """Neutral 3-way category over frac_wstar_top_riser ONLY (no claim attached to any token, item, or
    category). TARGETED_SHIFT iff frac >= frac_hi; OTHER_RISER iff frac <= frac_lo; MIXED otherwise. Both
    boundaries inclusive. frac None (n_eval == 0) -> UNDEFINED. Pure (float|None -> dict)."""
    if frac is None:
        return {"category": "UNDEFINED", "frac_wstar_top_riser": None, "frac_hi": frac_hi, "frac_lo": frac_lo,
                "msg": "n_eval == 0 (no non-collision items): frac_wstar_top_riser undefined."}
    if frac >= frac_hi:
        cat = "TARGETED_SHIFT"
        msg = f"frac_wstar_top_riser={frac:.4f} >= FRAC_HI({frac_hi})."
    elif frac <= frac_lo:
        cat = "OTHER_RISER"
        msg = f"frac_wstar_top_riser={frac:.4f} <= FRAC_LO({frac_lo})."
    else:
        cat = "MIXED"
        msg = f"FRAC_LO({frac_lo}) < frac_wstar_top_riser={frac:.4f} < FRAC_HI({frac_hi})."
    return {"category": cat, "frac_wstar_top_riser": frac, "frac_hi": frac_hi, "frac_lo": frac_lo, "msg": msg}


# --------------------------------------------------------------------------- family loading
def load_family(family):
    """Load the paraphrase family. `--family verifier_family` -> the module's ITEMS; otherwise treat `family`
    as a path to a JSON list of {q, correct, Wstar, ...}. Returns the list UNFILTERED (no select_items)."""
    if family == "verifier_family":
        from verifier_family import ITEMS
        return list(ITEMS)
    return json.loads(Path(family).read_text())


# --------------------------------------------------------------------------- real-run helpers
def _full_softmax(logits):
    """Full next-token probability vector at the LAST position (gemma-2's final softcap is applied inside the
    forward; same convention as the sibling controls). Returns a 1-D float tensor."""
    import torch
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _tensor_rank(P, tok_id):
    """1-indexed vocab rank of tok_id in the full prob tensor P: 1 + (#tokens with strictly greater p). Same
    strictly-greater convention as the pure rank_of (dict). Returns int."""
    import torch  # noqa: F401  (P is already a torch tensor)
    p = float(P[tok_id])
    return 1 + int((P > p).sum().item())


def _measure_model(name, is_chat, device, items):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident.
    Measures + DUMPS every item: the BARE / NEUTRAL / COUNTER top-K answer-slot distributions, the C/W* p +
    rank per prompt, the neutral->counter delta_topk, and the top_riser / wstar_is_top_riser derived flags."""
    import torch
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL
    from rlhf_differential import _helpers

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)

    def tok_str(tid):
        return tok.decode([int(tid)])

    def topk_list(P):
        vals, idx = torch.topk(P, TOP_K)
        return [{"tok_id": int(i), "tok_str": tok_str(int(i)), "p": round(float(v), 6)}
                for v, i in zip(vals.tolist(), idx.tolist())]

    records = []
    for it in items:
        q, C, W = it["q"], it["correct"], it["Wstar"]

        bare = single(q)
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        cid, aid = first(" " + C), first(" " + W)
        collision = (cid == aid)             # first-token collision -> C/W* readout degenerate (logged, NOT dropped)

        with torch.no_grad():
            Pb = _full_softmax(model(bare))
            Pn = _full_softmax(model(neutral))
            Pc = _full_softmax(model(counter))

        topk_bare, topk_neutral, topk_counter = topk_list(Pb), topk_list(Pn), topk_list(Pc)

        p_c_bare, rank_c_bare = round(float(Pb[cid]), 6), _tensor_rank(Pb, cid)
        p_w_bare, rank_w_bare = round(float(Pb[aid]), 6), _tensor_rank(Pb, aid)
        p_c_neutral, rank_c_neutral = round(float(Pn[cid]), 6), _tensor_rank(Pn, cid)
        p_w_neutral, rank_w_neutral = round(float(Pn[aid]), 6), _tensor_rank(Pn, aid)
        p_c_counter, rank_c_counter = round(float(Pc[cid]), 6), _tensor_rank(Pc, cid)
        p_w_counter, rank_w_counter = round(float(Pc[aid]), 6), _tensor_rank(Pc, aid)

        union = union_tokens([[e["tok_id"] for e in topk_bare],
                              [e["tok_id"] for e in topk_neutral],
                              [e["tok_id"] for e in topk_counter]], cid, aid)
        pn_map = {t: float(Pn[t]) for t in union}
        pc_map = {t: float(Pc[t]) for t in union}
        tstr_map = {t: tok_str(t) for t in union}
        delta_raw = delta_table(union, pn_map, pc_map, tstr_map)
        riser = pick_top_riser(delta_raw, cid)
        wstar_is_top_riser = bool(riser is not None and riser["tok_id"] == aid)
        delta_topk = [{"tok_id": r["tok_id"], "tok_str": r["tok_str"],
                       "p_neutral": round(r["p_neutral"], 6), "p_counter": round(r["p_counter"], 6),
                       "dp": round(r["dp"], 6)} for r in delta_raw]
        top_riser = (None if riser is None
                     else {"tok_id": riser["tok_id"], "tok_str": riser["tok_str"], "dp": round(riser["dp"], 6)})

        rec = {
            "q": q, "correct": C, "Wstar": W, "cid": int(cid), "aid": int(aid),
            "first_token_collision": bool(collision),
            "topk_bare": topk_bare, "topk_neutral": topk_neutral, "topk_counter": topk_counter,
            "p_c_bare": p_c_bare, "rank_c_bare": rank_c_bare,
            "p_w_bare": p_w_bare, "rank_w_bare": rank_w_bare,
            "p_c_neutral": p_c_neutral, "rank_c_neutral": rank_c_neutral,
            "p_w_neutral": p_w_neutral, "rank_w_neutral": rank_w_neutral,
            "p_c_counter": p_c_counter, "rank_c_counter": rank_c_counter,
            "p_w_counter": p_w_counter, "rank_w_counter": rank_w_counter,
            "delta_topk": delta_topk,
            "wstar_rank_bare": rank_w_bare,
            "top_riser": top_riser,
            "wstar_is_top_riser": wstar_is_top_riser,
        }
        records.append(rec)
        if collision:
            print(f"  [{tag}] first-token collision cid==aid -> C/W* readout degenerate (logged, excluded from "
                  f"fractions) q={q[:40]!r}", flush=True)
        tr = "-" if top_riser is None else f"{top_riser['tok_str']!r} dp={top_riser['dp']:+.3f}"
        print(f"  [{tag}] W*_rank_bare={rank_w_bare} top_riser={tr} wstar_top_riser={int(wstar_is_top_riser)} "
              f"coll={int(collision)} q={q[:34]!r}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    agg = aggregate(records)
    decision = decide(agg["frac_wstar_top_riser"])

    return {
        "name": name, "regime": "chat" if is_chat else "qa",
        "n_layers": nL, "n_heads": nH,
        "aggregate": agg,
        "decision": decision,
        "items": records,
    }


def run(family, name, tag, device, is_chat):
    items = load_family(family)
    print(f"[family] {family} -> {len(items)} items (no select_items; every item measured + dumped)", flush=True)

    res = _measure_model(name, is_chat, device, items)

    out = {
        "name": name, "device": device, "tag": tag, "regime": "chat" if is_chat else "qa",
        "cue": "family_topk_shift", "family": family, "n_items": len(items),
        "metric": ("Per-item paraphrase-family top-K distribution shift (no select_items; every item dumped). "
                   "Three prompts BARE = single(q), NEUTRAL = push(q,C,NEUTRAL), COUNTER = "
                   "push(q,C,PUSH['counter'].format(W=Wstar)); full answer-slot softmax at each. Records the "
                   "TOP_K tokens per prompt, the C/W* first-tok p + 1-indexed vocab rank per prompt, the "
                   "neutral->counter delta_topk (dp = p_counter - p_neutral over the union of the top-K sets "
                   "plus cid plus aid, dp-sorted), and the derived top_riser (max dp excluding cid) + "
                   "wstar_is_top_riser (top_riser == aid). Collision items (cid==aid) dumped + logged, excluded "
                   "from fractions."),
        "thresholds": {"TOP_K": TOP_K, "FRAC_HI": FRAC_HI, "FRAC_LO": FRAC_LO},
        "decision_rule": DECISION_RULE,
        "result": res,
    }
    Path("out").mkdir(exist_ok=True)
    out_path = f"out/family_topk_shift_{tag}.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    dd, agg = res["decision"], res["aggregate"]
    print(f"[{tag}] {dd['category']} n={agg['n']} n_collision={agg['n_collision']} n_eval={agg['n_eval']} | "
          f"frac_wstar_top_riser={agg['frac_wstar_top_riser']} "
          f"median_wstar_rank_bare={agg['median_wstar_rank_bare']} "
          f"wstar_in_bare_topk={agg['wstar_in_bare_topk']}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    # ---------- rank_of (1-indexed, strictly-greater convention; exact planted full-vocab dicts) ----------
    pm = {10: 0.4, 11: 0.3, 12: 0.2, 13: 0.1}                                       # a full 4-token "vocab"
    assert rank_of(pm, 10) == 1 and rank_of(pm, 11) == 2
    assert rank_of(pm, 12) == 3 and rank_of(pm, 13) == 4
    # ties share a rank (both 0.3 -> strictly-greater count is 1 -> rank 2 for each).
    pm_tie = {10: 0.4, 11: 0.3, 12: 0.3, 13: 0.0}
    assert rank_of(pm_tie, 11) == 2 and rank_of(pm_tie, 12) == 2 and rank_of(pm_tie, 13) == 4
    print("[selftest] rank_of: 1-indexed, strictly-greater; ties share a rank")

    # ---------- topk_ids (p-descending, tok_id tie-break) ----------
    assert topk_ids(pm, 2) == [10, 11]
    assert topk_ids({5: 0.5, 6: 0.5, 7: 0.1}, 2) == [5, 6]                          # p tie -> smaller tok_id first
    print("[selftest] topk_ids: p-descending, tok_id tie-break")

    # ---------- union_tokens (union of top-k sets PLUS cid PLUS aid; first-seen order, deduped) ----------
    cid, aid = 100, 200
    u = union_tokens([[1, 2, 3], [2, 3, 4], [3, 4, 5]], cid, aid)
    assert u == [1, 2, 3, 4, 5, 100, 200]                                          # deduped, cid/aid appended
    # cid/aid already present in a top-k set are NOT duplicated.
    u2 = union_tokens([[100, 9], [200]], cid, aid)
    assert u2 == [100, 9, 200]
    print("[selftest] union_tokens: union + cid + aid, deduped, first-seen order")

    # ---------- delta_table (dp = p_counter - p_neutral; sorted dp-descending; tok_id tie-break) ----------
    union = [1, 2, 3, cid, aid]
    p_neu = {1: 0.10, 2: 0.30, 3: 0.05, cid: 0.40, aid: 0.02}
    p_ctr = {1: 0.50, 2: 0.10, 3: 0.05, cid: 0.60, aid: 0.20}                       # dp: 1:+.40 2:-.20 3:0 cid:+.20 aid:+.18
    dt = delta_table(union, p_neu, p_ctr, {t: f"tok{t}" for t in union})
    dps = [round(r["dp"], 6) for r in dt]
    assert dps == sorted(dps, reverse=True), dps                                   # dp-descending
    assert dt[0]["tok_id"] == 1 and round(dt[0]["dp"], 6) == 0.40                   # token 1 is the max riser
    assert dt[0]["tok_str"] == "tok1" and dt[0]["p_neutral"] == 0.10 and dt[0]["p_counter"] == 0.50
    print("[selftest] delta_table: dp = p_counter - p_neutral, dp-descending sort")

    # ---------- pick_top_riser (max dp EXCLUDING cid) + wstar_is_top_riser TRUE case ----------
    # here cid (dp +.20) is not the max (token 1 is), so top_riser excluding cid is token 1, not aid.
    tr = pick_top_riser(dt, cid)
    assert tr["tok_id"] == 1 and (tr["tok_id"] == aid) is False                     # wstar_is_top_riser FALSE
    # Now plant cid as the single largest dp -> pick_top_riser must SKIP cid and take the next (= aid here).
    p_ctr2 = {1: 0.15, 2: 0.10, 3: 0.05, cid: 0.99, aid: 0.30}                      # dp: cid huge, aid next (+.28)
    dt2 = delta_table(union, p_neu, p_ctr2, {t: f"tok{t}" for t in union})
    assert dt2[0]["tok_id"] == cid                                                  # cid IS the max dp row
    tr2 = pick_top_riser(dt2, cid)
    assert tr2["tok_id"] == aid, tr2                                               # excluded cid -> aid is top_riser
    assert (tr2["tok_id"] == aid) is True                                          # wstar_is_top_riser TRUE
    # every-row-is-cid degenerate -> None.
    assert pick_top_riser(delta_table([cid], {cid: 0.1}, {cid: 0.9}, {cid: "c"}), cid) is None
    print("[selftest] pick_top_riser: excludes cid; wstar_is_top_riser true/false; all-cid -> None")

    # ---------- aggregate (collision EXCLUDED from fractions but PRESENT in dump; median over n_eval) ----------
    recs = [
        {"first_token_collision": False, "wstar_is_top_riser": True,  "wstar_rank_bare": 1},
        {"first_token_collision": False, "wstar_is_top_riser": True,  "wstar_rank_bare": 3},
        {"first_token_collision": False, "wstar_is_top_riser": False, "wstar_rank_bare": 50},
        {"first_token_collision": True,  "wstar_is_top_riser": True,  "wstar_rank_bare": 1},   # collision: dumped, not counted
    ]
    agg = aggregate(recs)
    assert agg["n"] == 4 and agg["n_collision"] == 1 and agg["n_eval"] == 3         # collision present but excluded
    assert agg["n_wstar_top_riser"] == 2                                            # only the 2 non-collision True
    assert abs(agg["frac_wstar_top_riser"] - (2 / 3)) < 1e-12
    assert agg["median_wstar_rank_bare"] == 3                                       # median(1,3,50)=3
    assert agg["wstar_in_bare_topk"] is True                                        # 3 <= TOP_K(10)
    # a family whose median bare rank falls OUTSIDE the top-K -> flag False.
    recs_deep = [{"first_token_collision": False, "wstar_is_top_riser": False, "wstar_rank_bare": r}
                 for r in (30, 40, 50)]
    assert aggregate(recs_deep)["wstar_in_bare_topk"] is False                      # median 40 > 10
    # all-collision family -> n_eval 0 -> frac / median None.
    agg0 = aggregate([{"first_token_collision": True, "wstar_is_top_riser": True, "wstar_rank_bare": 1}])
    assert agg0["n_eval"] == 0 and agg0["frac_wstar_top_riser"] is None and agg0["median_wstar_rank_bare"] is None
    print("[selftest] aggregate: collision dumped+excluded; frac + median over n_eval; wstar_in_bare_topk flag")

    # ---------- decide: TARGETED_SHIFT / OTHER_RISER / MIXED + inclusive boundaries at 0.5 and 0.2 ----------
    assert decide(0.75)["category"] == "TARGETED_SHIFT"
    assert decide(FRAC_HI)["category"] == "TARGETED_SHIFT"                          # 0.5 boundary inclusive (>=)
    assert decide(0.10)["category"] == "OTHER_RISER"
    assert decide(FRAC_LO)["category"] == "OTHER_RISER"                             # 0.2 boundary inclusive (<=)
    assert decide(0.35)["category"] == "MIXED"                                      # strictly between 0.2 and 0.5
    # just inside each boundary resolves to MIXED (exclusive on the open side).
    assert decide(FRAC_HI - 1e-9)["category"] == "MIXED"
    assert decide(FRAC_LO + 1e-9)["category"] == "MIXED"
    assert decide(None)["category"] == "UNDEFINED"                                  # n_eval == 0
    print(f"[selftest] decide: TARGETED_SHIFT >= FRAC_HI({FRAC_HI}) / OTHER_RISER <= FRAC_LO({FRAC_LO}) / MIXED "
          "(boundaries inclusive)")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--family", default="verifier_family",
                   help="'verifier_family' (the module's ITEMS) OR a path to a JSON list of {q,correct,Wstar,...}")
    p.add_argument("--name", default="google/gemma-2-9b", help="model (base is the QA template; -it via --chat)")
    p.add_argument("--tag", default="vfam_9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--chat", action="store_true",
                   help="use the chat template (-it model; qa template otherwise; base is primary)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.family, args.name, args.tag, args.device, args.chat)


if __name__ == "__main__":
    main()
