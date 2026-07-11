"""EMIT a per-item CANDIDATE TABLE for a paraphrase FAMILY: from the model's BARE-arm answer-slot top-K, name
the first non-correct-variant token, greedy-expand EVERY top-K token to text, and record whether the curated
W* is in the top-K and whether the candidate expansion matches it. This is a TABLE PRODUCER, not a verdict.

CONTEXT (neutral). A paraphrase family is a list of {q, correct, Wstar, tier, category} items. This instrument
reads ONLY the bare-question arm (base-model / QA template) and, per item, dumps the full top-K next-token
distribution at the answer slot, a greedy text expansion forced from each top-K token, and a derived CANDIDATE
= the first top-K token (in rank order) that is NOT a surface variant of the correct answer's first token. It
attaches NO interpretation to any token, item, candidate, category, or count; it does NOT select or filter --
every loaded item is measured and dumped. The output is a candidate table; the decision field is a fixed
'CANDIDATES_EMITTED' with descriptive counts only (no threshold verdict).

WHAT IT MEASURES (any HookedTransformer; QA template by default; family loaded from --family). The prompt
builder is the repo's `_helpers` bare arm (single = "Q: {q}\nA:"). Per item {q, correct: C, Wstar: W}:
  1. BARE = single(q); full answer-slot softmax at the last prompt position (_full_softmax).
  2. topk: the TOP_K(10) next tokens (p-descending, tok_id tie-break), each as {tok_id, tok_str, p, rank}.
  3. is_c_variant per top-K token t: True iff t == first(" "+C), OR t is a case/leading-space first-token
     variant of C (first token of {C, C.lower(), C.upper(), C.capitalize()} with/without a leading space),
     OR decode(t).strip().lower() == decode(first(" "+C)).strip().lower().
  4. expansion per top-K token (EVERY token, not only the candidate): force t as the first answer token, then
     greedy-decode up to MAX_EXPANSION_TOKENS(8) more tokens (do_sample=False), cut at the first newline.
  5. candidate = the FIRST top-K token in rank order with is_c_variant False. Persist candidate_token
     (tok_id + tok_str), candidate_rank, candidate_expansion. If ALL top-K tokens are C-variants:
     candidate=null, reason="all_c_variants".
  6. wstar_in_topk / wstar_topk_rank: whether first(" "+W) appears in the top-K, and at what 1-indexed rank.
  7. matches_curated: whether candidate_expansion string-matches the curated W (case-insensitive substring
     either way); False when there is no candidate.

PER-ITEM DUMP (EVERY item): q, correct, Wstar, tier, category, cid (=first(" "+C)) + cid_str, the full topk
table (tok_id, tok_str, p, rank, is_c_variant, expansion), candidate_token, candidate_rank,
candidate_expansion, reason, wstar_first_id + wstar_first_str, wstar_in_topk, wstar_topk_rank, matches_curated.

DECISION (module constants TOP_K=10, MAX_EXPANSION_TOKENS=8; NO verdict -- this produces a candidate table):
  decision = "CANDIDATES_EMITTED" with counts {n_items, n_with_candidate, n_all_c_variants, n_matches_curated}.
  Counts are descriptive only; no claim is attached to any token, item, candidate, category, or count.

Model-free --selftest (CPU, NO model load, no torch): planted distributions / a stub first-token encoder
exercise the top-K rank ordering + tok_id tie-break, is_c_variant true/false (exact id, case/leading-space
variants, decoded-string variant, and a non-variant), candidate selection SKIPPING a C-variant rank-1, the
all-C-variants -> null + reason path, matches_curated true/false (either substring direction; null-candidate
False), the wstar top-K position, and the counts aggregation. torch + transformer_lens are imported INSIDE the
real-run functions, so --selftest needs neither.

transformer_lens ONLY, forward-only (softmax + greedy generate), bf16, one model resident then freed. Bare arm
only (base-model / QA template); no --chat. Writes out/modelw_candidates_{tag}.json with the full per-item table.

  python controls/modelw_candidates.py --selftest
  python controls/modelw_candidates.py --family verifier_family --name google/gemma-2-9b --tag vfam_9bbase --device cuda
  python controls/modelw_candidates.py --family path/to/family.json --name google/gemma-2-9b --tag fam_9bbase --device cuda
"""
import argparse
import json
import sys
from pathlib import Path

# FLAT-scp: controls/ for the sibling-control reuse, latent_verify/ for the repo imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Pre-registered constants (neutral: stated on the measured numbers only).
TOP_K = 10                  # how many top next-token entries to record + expand per item
MAX_EXPANSION_TOKENS = 8    # greedy tokens decoded AFTER the forced first answer token (cut at newline)

DECISION_RULE = (
    "Per item on a paraphrase family (no select/filter; every item measured + dumped). BARE = single(q) "
    "(QA template 'Q: {q}\\nA:'); full answer-slot softmax at the last prompt position. Take the TOP_K(10) "
    "next tokens (p-descending, tok_id tie-break). Per top-K token t (rank order) is_c_variant = "
    "(t == first(' '+correct)) OR (t is a case/leading-space first-token variant of correct) OR "
    "(decode(t).strip().lower() == decode(first(' '+correct)).strip().lower()). For EVERY top-K token, force t "
    "as the first answer token and greedy-decode up to MAX_EXPANSION_TOKENS(8) more tokens (do_sample=False, "
    "cut at first newline) -> expansion. candidate = the FIRST rank-order token with is_c_variant False "
    "(candidate=null, reason='all_c_variants' if every top-K token is a C-variant). Persist candidate_token/"
    "rank/expansion, the full top-K table (tok_id, tok_str, p, rank, is_c_variant, expansion), whether "
    "first(' '+Wstar) is in the top-K and at what 1-indexed rank, and matches_curated = candidate_expansion "
    "string-matches the curated Wstar (case-insensitive substring either way; False when no candidate). NO "
    "verdict: decision = 'CANDIDATES_EMITTED' with counts {n_items, n_with_candidate, n_all_c_variants, "
    "n_matches_curated}. Counts are descriptive only; no claim attached to any token, item, candidate, "
    "category, or count."
)


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def topk_ids(prob_map, k):
    """The top-k tok_ids of a prob_map, p-descending with tok_id as the deterministic tie-break. Pure. (Used
    by --selftest to exercise the rank ordering; the real run uses torch.topk with the same p-descending
    order.)"""
    return [t for t, _ in sorted(prob_map.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def _c_variants(correct):
    """Case/leading-space surface variants of the correct answer whose FIRST TOKEN counts as a C-variant: the
    answer as-is, lowercased, uppercased, and capitalized, each with and without a leading space. De-duped,
    first-seen order. Pure (str -> [str])."""
    c = (correct or "").strip()
    out = []
    for b in (c, c.lower(), c.upper(), c.capitalize()):
        for s in (" " + b, b):
            if s and s not in out:
                out.append(s)
    return out


def c_variant_ids(correct, first_fn):
    """The set of FIRST-token ids of the correct answer's case/leading-space variants (via `first_fn`, the
    first-token encoder). Encoder misses (empty/unknown) are skipped. Pure over `first_fn` (str -> int)."""
    ids = set()
    for v in _c_variants(correct):
        try:
            ids.add(int(first_fn(v)))
        except (IndexError, KeyError):
            continue
    return ids


def is_c_variant_tok(tok_id, tok_str, cvar_ids, c_first_norm):
    """Whether a top-K token is a variant of the correct answer's first token: True iff its id is in
    `cvar_ids` (exact first-token match incl. case/leading-space variants) OR its decoded string, stripped +
    lowercased, equals `c_first_norm` (= decode(first(' '+correct)).strip().lower()). Pure."""
    if int(tok_id) in cvar_ids:
        return True
    if (tok_str or "").strip().lower() == c_first_norm:
        return True
    return False


def pick_candidate(rows):
    """The candidate = the FIRST row (rows are already in rank order) whose is_c_variant is False. Returns
    (row, None). If every row is a C-variant, returns (None, 'all_c_variants'). Pure (list -> (dict|None, str|None))."""
    for r in rows:
        if not r["is_c_variant"]:
            return r, None
    return None, "all_c_variants"


def matches_curated(expansion, curated):
    """Whether the candidate expansion string-matches the curated Wstar, case-insensitively, in EITHER
    substring direction (curated in expansion OR expansion in curated). False when either side is missing or
    empty (e.g. no candidate). Pure (str|None, str|None -> bool)."""
    e = (expansion or "").strip().lower()
    w = (curated or "").strip().lower()
    if not e or not w:
        return False
    return (w in e) or (e in w)


def wstar_topk_position(topk_id_list, wstar_id):
    """Whether `wstar_id` appears in the ordered top-K id list, and at what 1-indexed rank (position). Returns
    (bool, int|None). Pure (list[int], int -> (bool, int|None))."""
    if int(wstar_id) in topk_id_list:
        return True, topk_id_list.index(int(wstar_id)) + 1
    return False, None


# --------------------------------------------------------------------------- pure aggregate + decision
def decide(records):
    """Descriptive counts over the per-item candidate table (NO verdict). `records` = the per-item dump dicts.
    decision fixed 'CANDIDATES_EMITTED' with counts {n_items, n_with_candidate, n_all_c_variants,
    n_matches_curated}. Pure (list -> dict)."""
    n_items = len(records)
    n_with = sum(1 for r in records if r["candidate_token"] is not None)
    n_all_c = sum(1 for r in records if r["reason"] == "all_c_variants")
    n_match = sum(1 for r in records if r["matches_curated"])
    return {
        "decision": "CANDIDATES_EMITTED",
        "counts": {
            "n_items": n_items,
            "n_with_candidate": n_with,
            "n_all_c_variants": n_all_c,
            "n_matches_curated": n_match,
        },
        "msg": (f"Candidate table emitted for {n_items} items: {n_with} with a candidate, {n_all_c} "
                f"all-C-variant (candidate=null), {n_match} whose candidate expansion string-matches the "
                f"curated Wstar. Descriptive counts only; no verdict."),
    }


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


def _measure_model(name, is_chat, device, items):
    """One model end-to-end (forward-only softmax + greedy generate), loaded and FREED inside this call so only
    one model is resident. Bare arm only. Measures + DUMPS every item: the BARE top-K answer-slot distribution,
    a greedy expansion forced from each top-K token, the derived candidate, the curated-W* top-K position, and
    the matches_curated flag. Returns the per-item table + descriptive counts."""
    import torch
    from transformer_lens import HookedTransformer
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

    def expand(prompt_ids, forced_id):
        """Force `forced_id` as the first answer token after `prompt_ids`, then greedy-decode up to
        MAX_EXPANSION_TOKENS more tokens (do_sample=False); decode the forced token + continuation and cut at
        the first newline. Returns the expansion string."""
        assert prompt_ids.ndim == 2 and prompt_ids.shape[0] == 1, prompt_ids.shape
        forced = torch.tensor([[int(forced_id)]], device=prompt_ids.device)
        seq = torch.cat([prompt_ids, forced], dim=1)
        with torch.no_grad():
            gen = model.generate(seq, max_new_tokens=MAX_EXPANSION_TOKENS, do_sample=False, verbose=False)
        new_ids = gen[0, prompt_ids.shape[1]:]           # forced token + greedy continuation
        text = tok.decode(new_ids, skip_special_tokens=True)
        return text.split("\n", 1)[0].strip()

    records = []
    for it in items:
        q, C, W = it["q"], it["correct"], it["Wstar"]
        tier, category = it.get("tier"), it.get("category")

        bare = single(q)
        cid = int(first(" " + C))
        c_first_norm = tok_str(cid).strip().lower()
        cvar_ids = c_variant_ids(C, first)
        wstar_first_id = int(first(" " + W))

        with torch.no_grad():
            Pb = _full_softmax(model(bare))
        vals, idx = torch.topk(Pb, TOP_K)                # p-descending
        topk_pairs = [(int(i), float(v)) for v, i in zip(vals.tolist(), idx.tolist())]
        topk_pairs.sort(key=lambda kv: (-kv[1], kv[0]))  # torch.topk breaks ties arbitrarily; enforce tok_id tie-break (matches topk_ids helper)
        topk_id_list = [tid for tid, _ in topk_pairs]

        wstar_in_topk, wstar_topk_rank = wstar_topk_position(topk_id_list, wstar_first_id)

        rows = []
        for rk, (tid, p) in enumerate(topk_pairs, start=1):
            ts = tok_str(tid)
            rows.append({
                "tok_id": tid, "tok_str": ts, "p": round(p, 6), "rank": rk,
                "is_c_variant": bool(is_c_variant_tok(tid, ts, cvar_ids, c_first_norm)),
                "expansion": expand(bare, tid),
            })

        cand_row, reason = pick_candidate(rows)
        if cand_row is None:
            candidate_token = None
            candidate_rank = None
            candidate_expansion = None
        else:
            candidate_token = {"tok_id": cand_row["tok_id"], "tok_str": cand_row["tok_str"]}
            candidate_rank = cand_row["rank"]
            candidate_expansion = cand_row["expansion"]
        mc = matches_curated(candidate_expansion, W)

        rec = {
            "q": q, "correct": C, "Wstar": W, "tier": tier, "category": category,
            "cid": cid, "cid_str": tok_str(cid),
            "topk": rows,
            "candidate_token": candidate_token,
            "candidate_rank": candidate_rank,
            "candidate_expansion": candidate_expansion,
            "reason": reason,
            "wstar_first_id": wstar_first_id, "wstar_first_str": tok_str(wstar_first_id),
            "wstar_in_topk": bool(wstar_in_topk), "wstar_topk_rank": wstar_topk_rank,
            "matches_curated": bool(mc),
        }
        records.append(rec)
        ct = "-" if candidate_token is None else f"{candidate_token['tok_str']!r}@{candidate_rank}"
        print(f"  [{tag}] cand={ct} exp={candidate_expansion!r} wstar_in_topk={int(wstar_in_topk)} "
              f"wstar_rank={wstar_topk_rank} matches_curated={int(mc)} reason={reason} q={q[:34]!r}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    decision = decide(records)

    return {
        "name": name, "regime": "chat" if is_chat else "qa",
        "n_layers": nL, "n_heads": nH,
        "decision": decision,
        "items": records,
    }


def run(family, name, tag, device):
    items = load_family(family)
    print(f"[family] {family} -> {len(items)} items (no select_items; every item measured + dumped)", flush=True)

    res = _measure_model(name, is_chat=False, device=device, items=items)

    out = {
        "name": name, "device": device, "tag": tag, "regime": "qa",
        "cue": "modelw_candidates", "family": family, "n_items": len(items),
        "metric": ("Per-item BARE-arm candidate table on a paraphrase family (bare arm only; no select_items; "
                   "every item dumped). BARE = single(q); full answer-slot softmax at the last prompt position. "
                   "Records the TOP_K next tokens (p-descending, tok_id tie-break) as {tok_id, tok_str, p, "
                   "rank}, per-token is_c_variant (first token of ' '+correct / case+leading-space variant / "
                   "decoded-string variant), and a greedy expansion for EVERY top-K token (force t as the first "
                   "answer token, greedy up to max_expansion_tokens more, cut at newline). Derives candidate = "
                   "the first rank-order non-C-variant token (null + reason='all_c_variants' if all are "
                   "C-variants), records whether first(' '+Wstar) is in the top-K and at what rank, and "
                   "matches_curated (candidate expansion vs curated Wstar, case-insensitive substring either "
                   "way). Table producer; the decision is a fixed CANDIDATES_EMITTED with descriptive counts."),
        "params": {"K": TOP_K, "max_expansion_tokens": MAX_EXPANSION_TOKENS},
        "decision_rule": DECISION_RULE,
        "result": res,
    }
    Path("out").mkdir(exist_ok=True)
    out_path = f"out/modelw_candidates_{tag}.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    c = res["decision"]["counts"]
    print(f"[{tag}] CANDIDATES_EMITTED n_items={c['n_items']} n_with_candidate={c['n_with_candidate']} "
          f"n_all_c_variants={c['n_all_c_variants']} n_matches_curated={c['n_matches_curated']}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    # ---------- topk_ids (p-descending rank ordering + tok_id tie-break) ----------
    assert topk_ids({10: 0.4, 11: 0.3, 12: 0.2, 13: 0.1}, 3) == [10, 11, 12]
    assert topk_ids({5: 0.5, 6: 0.5, 7: 0.1}, 2) == [5, 6]                          # p tie -> smaller tok_id first
    print("[selftest] topk_ids: p-descending rank ordering, tok_id tie-break")

    # ---------- _c_variants / c_variant_ids (case + leading-space first-token variants) ----------
    variants = _c_variants("Nile")
    # as-is / lower / upper / capitalize, each with + without a leading space (capitalize('Nile')=='Nile' dedups).
    assert " Nile" in variants and "Nile" in variants
    assert " nile" in variants and "nile" in variants
    assert " NILE" in variants and "NILE" in variants
    # stub first-token encoder over a synthetic case/space-sensitive vocab.
    STUB_IDS = {" Nile": 10, "Nile": 11, " nile": 12, "nile": 13, " NILE": 14, "NILE": 15,
                " Amazon": 20, "Amazon": 21}

    def stub_first(s):
        if s in STUB_IDS:
            return STUB_IDS[s]
        raise KeyError(s)

    cvar = c_variant_ids("Nile", stub_first)
    assert cvar == {10, 11, 12, 13, 14, 15}, cvar                                    # all Nile case/space variants
    assert 20 not in cvar and 21 not in cvar                                         # Amazon is NOT a C-variant
    print("[selftest] c_variant_ids: case + leading-space first-token variants collected; non-C excluded")

    # ---------- is_c_variant_tok: exact id / case+space variant / decoded-string variant / non-variant ----------
    assert is_c_variant_tok(10, " Nile", cvar, "nile") is True                       # exact id match
    assert is_c_variant_tok(15, "NILE", cvar, "nile") is True                        # uppercase variant (by id)
    assert is_c_variant_tok(999, " NiLe", cvar, "nile") is True                      # unseen id, decoded 'nile' matches
    assert is_c_variant_tok(20, " Amazon", cvar, "nile") is False                    # different token -> not a variant
    assert is_c_variant_tok(21, "Amazon", cvar, "nile") is False
    print("[selftest] is_c_variant_tok: exact-id / case+space / decoded-string TRUE; non-variant FALSE")

    # ---------- pick_candidate: skips a C-variant rank-1, takes the first non-C-variant ----------
    rows = [
        {"tok_id": 10, "tok_str": " Nile", "rank": 1, "is_c_variant": True,  "expansion": "Nile"},
        {"tok_id": 20, "tok_str": " Amazon", "rank": 2, "is_c_variant": False, "expansion": "Amazon River"},
        {"tok_id": 30, "tok_str": " The", "rank": 3, "is_c_variant": False, "expansion": "The Nile"},
    ]
    cand, reason = pick_candidate(rows)
    assert cand is not None and cand["tok_id"] == 20 and cand["rank"] == 2 and reason is None
    print("[selftest] pick_candidate: skips C-variant rank-1, returns first non-C-variant (rank 2)")

    # ---------- pick_candidate: ALL top-K are C-variants -> null + reason ----------
    rows_all_c = [
        {"tok_id": 10, "tok_str": " Nile", "rank": 1, "is_c_variant": True, "expansion": "Nile"},
        {"tok_id": 13, "tok_str": "nile",  "rank": 2, "is_c_variant": True, "expansion": "nile"},
    ]
    cand2, reason2 = pick_candidate(rows_all_c)
    assert cand2 is None and reason2 == "all_c_variants"
    print("[selftest] pick_candidate: all-C-variants -> candidate null, reason='all_c_variants'")

    # ---------- matches_curated: either substring direction; null-candidate / empty -> False ----------
    assert matches_curated("Amazon River", "Amazon") is True                         # curated in expansion
    assert matches_curated("Ama", "Amazon") is True                                  # expansion in curated
    assert matches_curated("AMAZON, the river", "amazon") is True                    # case-insensitive
    assert matches_curated("The Nile", "Amazon") is False                            # no overlap
    assert matches_curated(None, "Amazon") is False                                  # no candidate expansion
    assert matches_curated("Amazon", None) is False                                  # no curated
    assert matches_curated("", "Amazon") is False and matches_curated("Amazon", "") is False
    print("[selftest] matches_curated: either substring direction TRUE; null/empty FALSE")

    # ---------- wstar_topk_position: present -> (True, rank); absent -> (False, None) ----------
    assert wstar_topk_position([10, 20, 30], 20) == (True, 2)
    assert wstar_topk_position([10, 20, 30], 999) == (False, None)
    print("[selftest] wstar_topk_position: present -> (True, 1-indexed rank); absent -> (False, None)")

    # ---------- decide: descriptive counts aggregation (no verdict) ----------
    recs = [
        {"candidate_token": {"tok_id": 20}, "reason": None, "matches_curated": True},   # candidate + matches curated
        {"candidate_token": {"tok_id": 30}, "reason": None, "matches_curated": False},  # candidate, no match
        {"candidate_token": None, "reason": "all_c_variants", "matches_curated": False},  # all-C-variants
    ]
    dec = decide(recs)
    assert dec["decision"] == "CANDIDATES_EMITTED"
    assert dec["counts"] == {"n_items": 3, "n_with_candidate": 2, "n_all_c_variants": 1, "n_matches_curated": 1}, dec["counts"]
    # empty family -> all-zero counts, still CANDIDATES_EMITTED (no verdict).
    dec0 = decide([])
    assert dec0["decision"] == "CANDIDATES_EMITTED"
    assert dec0["counts"] == {"n_items": 0, "n_with_candidate": 0, "n_all_c_variants": 0, "n_matches_curated": 0}
    print("[selftest] decide: CANDIDATES_EMITTED + exact counts {n_items, n_with_candidate, n_all_c_variants, n_matches_curated}")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--family", default="verifier_family",
                   help="'verifier_family' (the module's ITEMS) OR a path to a JSON list of {q,correct,Wstar,...}")
    p.add_argument("--name", default="google/gemma-2-9b", help="model (bare arm only; QA template)")
    p.add_argument("--tag", default="vfam_9bbase")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.family, args.name, args.tag, args.device)


if __name__ == "__main__":
    main()
