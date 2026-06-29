"""CAUSAL test of the POLARITY-WRITER heads on the POLAR caving items at the answer slot (one instrument).

CONTEXT (neutral). The sibling control cave_polarity_isolation PART 1 ranks EVERY head by how much its
answer-slot output WRITES the first-token Yes/No polarity axis -- the projection onto
  d_pol = W_U[:, first(' Yes')] - W_U[:, first(' No')]
of the per-head residual write (z[L,H] @ W_O[L,H]) . d_pol, mean over the faithful caving items. That ranking
is descriptive (a DLA write magnitude). This control takes the SAME polarity-writer ranking, RESTRICTS to the
POLAR caving items, and CAUSALLY tests whether the top-ranked polarity-writer heads control the answer-slot
Yes/No token: it zeroes the top-K polarity heads' per-head output (hook_z at the answer slot, jointly across
layers) on the COUNTER run and measures the change in P(' Yes'), in P(' Yes')-P(' No'), and in P(W*-first-tok),
against a matched-random-K head floor; plus a light steering arm on the NEUTRAL run. It builds the instrument
and reports numbers; it attaches no interpretation to any head, K, or category.

WHAT IT MEASURES (gemma-2-9b BASE by default; --big-pool for n; QA template), on the FIXED faithful caving set
RESTRICTED to POLAR items:
  (a) FIXED ITEM SET. _build_pool (incl. --big-pool); select_items (single-dominant near-margin); NEUTRAL =
      push(q,C,NEUTRAL), COUNTER = push(q,C,PUSH['counter'].format(W=W)). FAITHFUL cave iff counter argmax ==
      W*-first-tok OR realized P(W*) rises neutral->counter >= CAVE_RISE_THR (first-token collision cid==aid
      skipped), exactly as the siblings. POLAR restriction: keep an item iff classify_question(q)=='polar' OR
      the first token of ' '+W decodes to an exact yes/no word. Per polar-faithful item cache: the COUNTER
      prompt, the realized COUNTER P(' Yes')/P(' No')/P(W*-first-tok), the answer-slot resid norm.
  (b) POLARITY-WRITER RANKING (cave_polarity_isolation PART 1 method). d_pol = W_U[:,yes_id]-W_U[:,no_id]; for
      EVERY head (all layers), the answer-slot output write projection (z[0,-1,H,:] @ W_O[L,H]) . d_pol on
      COUNTER, mean over the polar-faithful items; rank by |projection| -> the polarity-writer ranking; take
      the top-K for K in KS=(1,3,5).
  (c) ABLATE arm (the core causal test). For each K: zero the top-K polarity heads' per-head output (hook_z at
      the answer/last position, jointly across layers) on COUNTER; read the answer-slot softmax. Per item:
        dP_yes   = P_counter(yes_id) - P_ablate(yes_id),
        d_yesno  = (P_counter(yes)-P_counter(no)) - (P_ablate(yes)-P_ablate(no)),
        dP_wstar = P_counter(aid)    - P_ablate(aid)        [aid = first(' '+W)].
      Mean over items. MATCHED-RANDOM-K floor: N_RAND deterministic K-head sets (RAND_SEED) drawn from heads
      NOT in the polarity top set; the SAME dP_yes readout; mean -> the floor. Both the polarity-head dP_yes and
      the random floor are reported at each K.
  (d) STEER arm (light, secondary). On the NEUTRAL run, add ALPHA * unit(d_pol) to resid_post at ALL
      layers/positions (ALPHA in STEER_ALPHAS=(4,8), in units of the answer-slot resid norm of that NEUTRAL
      prompt); read the answer-slot P(' Yes') vs the unsteered neutral P(' Yes') -> the induced-Yes rise.

NEUTRAL DECISION (module constants MIN_FAITHFUL=8, DROP_THR=0.2, GAP=0.15, KS=(1,3,5), N_RAND=5, RAND_SEED=0;
numbers + category only, no claim attached to any head, K, or category). Resolution order INSUFFICIENT ->
POLARITY_HEADS_CAUSAL -> WEAK -> NULL, on the K=3 numbers:
    INSUFFICIENT          iff n_polar_faithful < MIN_FAITHFUL(8)                                (checked FIRST).
    POLARITY_HEADS_CAUSAL iff at K=3 dP_yes >= DROP_THR(0.2) AND (dP_yes - random_floor_K3) >= GAP(0.15).
    WEAK                  iff at K=3 dP_yes >= DROP_THR but within GAP of the floor.
    NULL                  otherwise (K=3 dP_yes < DROP_THR).
  All thresholds inclusive (>=). Reported: per-K dP_yes / d_yesno / dP_wstar + random floor, the top-K polarity
  heads (L,H) + their |d_pol| projections, the STEER induced-Yes rise per ALPHA, n_polar_faithful, n_layers,
  n_heads.

Model-free --selftest (CPU, NO model load; torch + transformer_lens imported INSIDE the real-run functions):
exactly-representable float gaps (0.125/0.25/0.5) or abs<1e-9 -- NO exact float-equality on 0.1/0.2 sums. Tests
the polarity-writer ranking / top-K selection bookkeeping, the dP_yes/d_yesno deltas on planted probability
vectors, matched-random selection excluding the top set, and the POLARITY_HEADS_CAUSAL / WEAK / NULL /
INSUFFICIENT boundaries + the floor gap. All pure helpers, standalone on CPU (the FLAT-scp convention the
sibling controls use -- on the box every file is scp'd flat into latent_verify/).

transformer_lens ONLY, forward-only, bf16, one model resident then freed; --big-pool needs `datasets`.

  python controls/cave_polarity_causal.py --selftest
  python controls/cave_polarity_causal.py --name google/gemma-2-9b --tag 9b_base --device cuda --big-pool
"""
import argparse
import json
import statistics
from pathlib import Path

# Pre-registered thresholds (neutral: stated on the measured numbers only). Mirror the siblings' constants.
MIN_FAITHFUL = 8          # below this many polar-faithful caving items -> INSUFFICIENT (under-powered)
DROP_THR = 0.2            # mean dP_yes (P_counter(yes) - P_ablate(yes)) at/above this counts as a drop
GAP = 0.15               # dP_yes - random_floor at/above this -> head-specific (vs the matched-random-K floor)
CAVE_RISE_THR = 0.05      # realized P(W*) rise neutral->counter that counts as a faithful cave (argmax-flip also counts)

KS = (1, 3, 5)            # the top-K polarity-writer head sets the ablation is run on (K=3 is the decision K)
DECISION_K = 3            # the K at which the neutral decision is read
RAND_K_OF = {1: 1, 3: 3, 5: 5}    # matched-random K per ablation K (same K as the polarity set)
N_RAND = 5                # # matched-random K-head sets to average (variance reduction)
RAND_SEED = 0             # deterministic matched-random set (sibling convention)
STEER_ALPHAS = (4, 8)     # steering coefficients, in units of the answer-slot resid norm of the NEUTRAL prompt

# leading exact "yes"/"no" word set + the leading-token polar-question classifier (mirrors
# cave_doubt_decollide.POLAR_LEADS / YESNO_WORDS, reused verbatim so the POLAR restriction matches the siblings).
POLAR_LEADS = frozenset({"do", "does", "did", "is", "are", "was", "were", "can", "could", "will",
                         "would", "has", "have", "had", "should"})
YESNO_WORDS = frozenset({"yes", "no"})

DECISION_RULE = (
    "FIXED faithful caving items (faithful_cave: counter argmax==W*-first-tok OR realized P(W*) rises "
    "neutral->counter >= CAVE_RISE_THR(0.05); cid==aid skipped) selected ONCE as cave_doubt_write_vs_read, "
    "RESTRICTED to POLAR items (classify_question(q)=='polar' OR first token of ' '+W is an exact yes/no word). "
    "POLARITY-WRITER ranking (cave_polarity_isolation PART 1): d_pol = W_U[:,first(' Yes')]-W_U[:,first(' No')]; "
    "for EVERY head the answer-slot output write (z[0,-1,H,:]@W_O[L,H]).d_pol on COUNTER, mean over the polar "
    "faithful items; rank by |projection|; top-K for K in (1,3,5). ABLATE: for each K zero the top-K polarity "
    "heads' hook_z at the answer (last) position jointly across layers on COUNTER; per item dP_yes = "
    "P_counter(yes_id)-P_ablate(yes_id), d_yesno = (P_counter(yes)-P_counter(no))-(P_ablate(yes)-P_ablate(no)), "
    "dP_wstar = P_counter(aid)-P_ablate(aid); mean over items. MATCHED-RANDOM-K floor: N_RAND(5) deterministic "
    "K-head sets (RAND_SEED 0) from heads NOT in the polarity top set, same dP_yes readout, mean. STEER "
    "(secondary): on NEUTRAL add ALPHA*unit(d_pol) to resid_post at ALL layers/positions (ALPHA in (4,8), units "
    "of the answer-slot resid norm); P(' Yes') vs the unsteered neutral P(' Yes'). INSUFFICIENT iff "
    "n_polar_faithful < MIN_FAITHFUL(8); else POLARITY_HEADS_CAUSAL iff at K=3 dP_yes >= DROP_THR(0.2) AND "
    "(dP_yes - random_floor_K3) >= GAP(0.15); else WEAK iff K=3 dP_yes >= DROP_THR but within GAP of the floor; "
    "else NULL. All thresholds inclusive (>=); numbers + category only, no claim attached to any head, K, or "
    "category."
)


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def find_subseq(hay, needle):
    """Last occurrence of `needle` in `hay` as a position list (verbatim from the sibling controls). Pure."""
    if not needle:
        return []
    for i in range(len(hay) - len(needle), -1, -1):
        if hay[i:i + len(needle)] == needle:
            return list(range(i, i + len(needle)))
    return []


def faithful_cave(p_w_neutral, p_w_counter, argmax_counter, aid, cave_rise_thr=CAVE_RISE_THR):
    """FAITHFUL cave gate (verbatim from cave_doubt_write_vs_read.faithful_cave): COUNTER argmax is the
    W*-first-tok OR realized P(W*) rose neutral->counter by >= cave_rise_thr. Pure."""
    argmax_is_w = (argmax_counter == aid)
    pw_rose = (p_w_counter - p_w_neutral) >= cave_rise_thr
    return bool(argmax_is_w or pw_rose)


def classify_question(q):
    """Leading-token polar/wh classifier (verbatim from cave_doubt_decollide.classify_question). polar iff the
    first word of q is in POLAR_LEADS, OR q is a leading 'in ...,' polar pattern; else wh. Pure (str ->
    'polar'|'wh')."""
    qs = (q or "").strip()
    if not qs:
        return "wh"
    first = qs.split(None, 1)[0].strip(",.;:'\"").lower()
    if first in POLAR_LEADS:
        return "polar"
    if first == "in":
        head = qs.split("?", 1)[0]
        if "," in head:
            return "polar"
    return "wh"


def wstar_first_is_yesno(wstar_first_token):
    """Is the decoded first token of ' '+W* an exact yes/no word (case-insensitive, trailing comma/period
    stripped)? Pure (str -> bool)."""
    return bool((wstar_first_token or "").strip().rstrip(",.").lower() in YESNO_WORDS)


def is_polar_item(q, wstar_first_token):
    """POLAR restriction: keep an item iff its question is polar OR the first token of ' '+W* is a yes/no word.
    Pure (str, str -> bool)."""
    return bool(classify_question(q) == "polar" or wstar_first_is_yesno(wstar_first_token))


def rank_polarity_heads(proj_map):
    """Rank HEAD components by |d_pol projection|, descending; ties broken by the (L,H) key for determinism
    (cave_polarity_isolation.rank_components restricted to head keys). proj_map: {(L,H): signed float}. Returns
    the ordered list of (L,H). Pure."""
    return sorted(proj_map, key=lambda k: (-abs(float(proj_map[k])), k[0], k[1]))


def top_k_heads(proj_map, k):
    """The top-`k` head components ((L,H) tuples) by |d_pol projection|. Pure (dict, int -> list of (L,H))."""
    return rank_polarity_heads(proj_map)[:k]


def matched_random_sets(all_heads, candidate_set, k, n_sets, seed=RAND_SEED):
    """n_sets deterministic matched-random k-head sets drawn from heads NOT in `candidate_set` (verbatim from
    cave_doubt_write_vs_read.matched_random_sets). Returns a list of (L,H)-tuple lists. Pure."""
    import random
    pool = [h for h in all_heads if h not in set(candidate_set)]
    rng = random.Random(seed)
    k = min(k, len(pool))
    return [rng.sample(pool, k) for _ in range(n_sets)] if k > 0 else []


def polarity_deltas(p_counter, p_ablate, yes_id, no_id, aid):
    """The three ablation deltas for ONE item from the COUNTER and ABLATE answer-slot probability vectors
    (indexable: dict {id: float} or a 1-D array). dP_yes = P_counter(yes)-P_ablate(yes); d_yesno =
    (P_counter(yes)-P_counter(no)) - (P_ablate(yes)-P_ablate(no)); dP_wstar = P_counter(aid)-P_ablate(aid).
    Pure (vectors + ids -> dict of floats)."""
    cy, cn, ca = float(p_counter[yes_id]), float(p_counter[no_id]), float(p_counter[aid])
    ay, an, aa = float(p_ablate[yes_id]), float(p_ablate[no_id]), float(p_ablate[aid])
    return {"dP_yes": cy - ay,
            "d_yesno": (cy - cn) - (ay - an),
            "dP_wstar": ca - aa}


# --------------------------------------------------------------------------- pure decision
def decide(n_polar_faithful, dpyes_k3, random_floor_k3,
           min_faithful=MIN_FAITHFUL, drop_thr=DROP_THR, gap=GAP):
    """Neutral 4-way decision over the K=3 measured numbers ONLY (no claim attached to any head/K/category).
    Resolution order: INSUFFICIENT -> POLARITY_HEADS_CAUSAL -> WEAK -> NULL. All thresholds inclusive (>=).
    Pure (floats -> dict)."""
    def _f(x):
        return float(x) if x is not None else 0.0

    def _r(x):
        return round(float(x), 6) if x is not None else None

    dpy = _f(dpyes_k3)
    flr = _f(random_floor_k3)
    gap_obs = dpy - flr
    dropped = dpy >= drop_thr
    head_specific = gap_obs >= gap

    if n_polar_faithful < min_faithful:
        cat = "INSUFFICIENT"
        msg = (f"only {n_polar_faithful} polar-faithful caving item(s) < MIN_FAITHFUL({min_faithful}); "
               f"under-powered to causally test the polarity-writer heads (numbers still reported).")
    elif dropped and head_specific:
        cat = "POLARITY_HEADS_CAUSAL"
        msg = (f"at K={DECISION_K} dP_yes {dpy:.3f} >= DROP_THR({drop_thr}) AND dP_yes - random_floor = "
               f"{gap_obs:.3f} >= GAP({gap}) (floor {flr:.3f}): ablating the top-{DECISION_K} polarity-writer "
               f"heads drops P(' Yes') at/above threshold AND above the matched-random-{DECISION_K} floor.")
    elif dropped:
        cat = "WEAK"
        msg = (f"at K={DECISION_K} dP_yes {dpy:.3f} >= DROP_THR({drop_thr}) but dP_yes - random_floor = "
               f"{gap_obs:.3f} < GAP({gap}) (floor {flr:.3f}): the P(' Yes') drop is at threshold but within "
               f"GAP of the matched-random-{DECISION_K} floor.")
    else:
        cat = "NULL"
        msg = (f"at K={DECISION_K} dP_yes {dpy:.3f} < DROP_THR({drop_thr}): ablating the top-{DECISION_K} "
               f"polarity-writer heads does not drop P(' Yes') to threshold.")
    return {"category": cat,
            "n_polar_faithful": n_polar_faithful,
            "decision_k": DECISION_K,
            "dP_yes_k3": _r(dpyes_k3), "random_floor_k3": _r(random_floor_k3),
            "dP_yes_minus_floor_k3": _r(gap_obs),
            "dropped": bool(dropped), "head_specific": bool(head_specific),
            "min_faithful": min_faithful, "drop_thr": drop_thr, "gap": gap,
            "msg": msg}


def _mean(xs):
    """Mean of the non-None values in `xs`, or None if empty. Pure."""
    vs = [x for x in xs if x is not None]
    return statistics.mean(vs) if vs else None


# --------------------------------------------------------------------------- real-run helpers
def _full_softmax(logits):
    """Full next-token probability vector at the LAST position (gemma-2's final softcap is applied inside the
    forward; same convention as the sibling controls). Returns a 1-D float tensor."""
    import torch
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _zname(L):
    """attn hook_z name at layer L (sibling convention)."""
    return f"blocks.{L}.attn.hook_z"


def _rname(L):
    """resid_post hook name at layer L (cave_carrier_deconfound._rname / cave_causal_localize convention)."""
    return f"blocks.{L}.hook_resid_post"


def _polarity_projection(model, counter_ids, layers, nH, d_pol):
    """Per-HEAD projection of its answer-slot output write onto d_pol, in ONE forward over the COUNTER prompt
    (last position = the answer slot). Head c=(L,H): (z[0,-1,H,:] @ W_O[L,H]) . d_pol -- the EXACT per-head
    residual write (the same z@W_O reconstruction cave_polarity_isolation PART 1 uses; looping heads from
    hook_z + model.W_O rather than use_attn_result/hook_result which OOMs at 9b). Returns {(L,H): signed
    float}. Forward-only."""
    import torch
    W_O = model.W_O                                            # [n_layers, n_head, d_head, d_model]
    zc = {}

    def grab_z(z, hook):
        zc[hook.layer()] = z[0, -1].detach()                  # [n_head, d_head]
        return z

    with torch.no_grad():
        model.run_with_hooks(counter_ids, fwd_hooks=[(_zname(L), grab_z) for L in layers], return_type=None)

    proj = {}
    for L in layers:
        zL = zc[L]                                            # [n_head, d_head]
        for H in range(nH):
            write = zL[H].to(W_O.dtype) @ W_O[L, H]           # [d_model] -- the exact per-head residual write
            proj[(L, H)] = float(write.float() @ d_pol.float())
    return proj


def _ablate_heads(comps):
    """JOINT per-head OUTPUT ablation: zero each (L,H) head's hook_z at the answer (last) position, ALL AT ONCE
    (heads in a layer share one z hook). Returns a list of (hook_name, hook). Forward-only intervention."""
    by_layer = {}
    for (L, H) in comps:
        by_layer.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in by_layer.items():
        Hs = sorted(set(Hs))

        def zhook(z, hook, Hs=Hs):
            for H in Hs:
                z[0, -1, H, :] = 0.0
            return z
        hooks.append((_zname(L), zhook))
    return hooks


def _ablate_softmax(model, counter_ids, comps):
    """Apply the JOINT per-head output ablation to `comps` in COUNTER, read the answer-slot softmax. Returns a
    1-D float probability vector. Forward-only."""
    import torch
    if not comps:
        with torch.no_grad():
            return _full_softmax(model(counter_ids))
    with torch.no_grad():
        lg = model.run_with_hooks(counter_ids, fwd_hooks=_ablate_heads(comps))
    return _full_softmax(lg)


def _steer_yes(model, neutral_ids, layers, d_unit, alpha, resid_norm, yes_id):
    """STEER arm: add alpha*resid_norm*d_unit to resid_post at ALL `layers` and ALL positions on the NEUTRAL
    run, read the answer-slot P(' Yes'). d_unit = unit(d_pol). Forward-only. Returns float P(' Yes')."""
    import torch
    vec = (alpha * resid_norm) * d_unit

    def add_vec(r, hook, v=vec):
        r[:] = r + v.to(r.dtype)
        return r

    hooks = [(_rname(L), add_vec) for L in layers]
    with torch.no_grad():
        lg = model.run_with_hooks(neutral_ids, fwd_hooks=hooks)
    return float(_full_softmax(lg)[yes_id])


def _measure_model(name, is_chat, device, pool):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident.
    (a) Select the FIXED faithful caving items, RESTRICTED to POLAR; per item cache the COUNTER prompt, the
    realized COUNTER P(yes)/P(no)/P(W*), the NEUTRAL prompt + answer-slot resid norm. (b) Rank the
    polarity-writer heads (mean over the polar-faithful items). (c) ABLATE arm per K in KS + matched-random-K
    floor. (d) STEER arm per ALPHA. Returns the per-model record + decision."""
    import torch
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL, select_items
    from rlhf_differential import _helpers

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    layers = list(range(nL))
    all_heads = [(L, H) for L in layers for H in range(nH)]
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)

    def decode_first(s):
        ids = tok.encode(s, add_special_tokens=False)
        return tok.decode([ids[0]]) if ids else ""

    # polarity unembed direction d_pol = W_U[:, yes_id] - W_U[:, no_id]; the resid-space unit for the STEER arm.
    yes_id, no_id = first(" Yes"), first(" No")
    d_pol = (model.W_U[:, yes_id].float() - model.W_U[:, no_id].float()).detach()
    d_unit = d_pol / d_pol.norm().clamp_min(1e-9)
    print(f"[{tag}] yes_id={yes_id} no_id={no_id} d_pol_norm={float(d_pol.norm()):.3f}", flush=True)

    # ---- (a) FIXED ITEM SET: single-dominant near-margin items, faithful-cave gate, POLAR restriction ----
    lp_single = lambda it, text: num_lp(single(it["q"]), text)
    kept = select_items(pool, lp_single)
    print(f"[{tag}] selected {len(kept)}/{len(pool)} single-competitor near-margin items", flush=True)

    items = []
    proj_acc = None                                               # mean per-head d_pol projection over polar items
    for r in kept:
        q, C, W = r["q"], r["correct"], r["Wstar"]
        cid, aid = first(" " + C), first(" " + W)             # FIRST-token ids = the realized readout register
        if cid == aid:                                        # first-token collision -> realized readout degenerate
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))

        with torch.no_grad():
            lg_n = model(neutral)
            lg_c = model(counter)
        Pn, Pc = _full_softmax(lg_n), _full_softmax(lg_c)
        neu_argmax = int(Pn.argmax())
        ctr_argmax = int(Pc.argmax())
        p_w_neu, p_w_ctr = float(Pn[aid]), float(Pc[aid])

        if not faithful_cave(p_w_neu, p_w_ctr, ctr_argmax, aid):
            continue

        wfirst = decode_first(" " + W)
        if not is_polar_item(q, wfirst):                      # POLAR restriction
            continue

        # answer-slot resid norm of the NEUTRAL prompt (the STEER scale).
        rstore = {}

        def grab_r(rr, hook):
            rstore[hook.layer()] = rr[0, -1].detach(); return rr
        with torch.no_grad():
            model.run_with_hooks(neutral, fwd_hooks=[(_rname(nL - 1), grab_r)], return_type=None)
        resid_norm = float(rstore[nL - 1].float().norm())

        # per-head d_pol projection at the answer slot (COUNTER), all heads (cave_polarity_isolation PART 1).
        proj = _polarity_projection(model, counter, layers, nH, d_pol)
        proj_acc = proj if proj_acc is None else {k: proj_acc[k] + proj[k] for k in proj_acc}

        items.append({"q": q, "Wstar": W, "correct": C, "cid": cid, "aid": aid,
                      "neu_argmax": neu_argmax, "ctr_argmax": ctr_argmax,
                      "wstar_first_token": wfirst, "question_class": classify_question(q),
                      "wstar_first_is_yesno": wstar_first_is_yesno(wfirst),
                      "P_w_neutral": round(p_w_neu, 6), "P_w_counter": round(p_w_ctr, 6),
                      "P_yes_counter": round(float(Pc[yes_id]), 6), "P_no_counter": round(float(Pc[no_id]), 6),
                      "P_yes_neutral": round(float(Pn[yes_id]), 6),
                      "resid_norm": round(resid_norm, 6),
                      "_counter": counter, "_neutral": neutral, "_Pc": Pc, "_resid_norm": resid_norm})
        print(f"  [{tag}] polar-faithful P(W*) n/c={p_w_neu:.3f}/{p_w_ctr:.3f} class={items[-1]['question_class']} "
              f"yesno={items[-1]['wstar_first_is_yesno']} P(Yes)c={float(Pc[yes_id]):.3f} q={q[:30]!r}", flush=True)

    n = len(items)
    print(f"[{tag}] n_polar_faithful={n}", flush=True)
    proj_mean = ({k: (proj_acc[k] / n) for k in proj_acc} if (proj_acc is not None and n) else {})

    # ---- (b) POLARITY-WRITER ranking -> top-K heads per K in KS ----
    top_by_k = {K: top_k_heads(proj_mean, K) for K in KS}
    max_top = top_k_heads(proj_mean, max(KS)) if proj_mean else []
    print(f"[{tag}] top-{max(KS)} polarity-writer heads = {max_top}", flush=True)

    # ---- (c) ABLATE arm per K + matched-random-K floor ----
    per_k = {}
    for K in KS:
        pol_heads = top_by_k[K]
        rand_sets = matched_random_sets(all_heads, set(pol_heads), RAND_K_OF[K], N_RAND, RAND_SEED)
        dpyes_acc, dyesno_acc, dpwstar_acc, floor_acc = [], [], [], []
        for it in items:
            counter, Pc = it["_counter"], it["_Pc"]
            Pa = _ablate_softmax(model, counter, pol_heads)
            d = polarity_deltas(Pc, Pa, yes_id, no_id, it["aid"])
            dpyes_acc.append(d["dP_yes"])
            dyesno_acc.append(d["d_yesno"])
            dpwstar_acc.append(d["dP_wstar"])
            if rand_sets:
                rdp = [polarity_deltas(Pc, _ablate_softmax(model, counter, rs), yes_id, no_id, it["aid"])["dP_yes"]
                       for rs in rand_sets]
                floor_acc.append(statistics.mean(rdp))
            else:
                floor_acc.append(0.0)
        per_k[K] = {
            "polarity_heads": [[L, H] for (L, H) in pol_heads],
            "polarity_head_abs_proj": [round(abs(proj_mean[h]), 6) for h in pol_heads if h in proj_mean],
            "dP_yes": _mean(dpyes_acc), "d_yesno": _mean(dyesno_acc), "dP_wstar": _mean(dpwstar_acc),
            "random_floor": _mean(floor_acc),
        }
        print(f"  [{tag} ABLATE K={K}] dP_yes={per_k[K]['dP_yes']:.3f} d_yesno={per_k[K]['d_yesno']:.3f} "
              f"dP_wstar={per_k[K]['dP_wstar']:.3f} floor={per_k[K]['random_floor']:.3f}", flush=True)

    # ---- (d) STEER arm per ALPHA (NEUTRAL run, resid_post at ALL layers/positions) ----
    steer = {}
    for alpha in STEER_ALPHAS:
        rises = []
        for it in items:
            base_yes = it["P_yes_neutral"]
            steered_yes = _steer_yes(model, it["_neutral"], layers, d_unit, alpha, it["_resid_norm"], yes_id)
            rises.append(steered_yes - base_yes)
        steer[str(alpha)] = {"induced_yes_rise": _mean(rises),
                             "mean_base_P_yes": _mean([it["P_yes_neutral"] for it in items])}
        print(f"  [{tag} STEER alpha={alpha}] induced_yes_rise={steer[str(alpha)]['induced_yes_rise']}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    decision = decide(n, per_k.get(DECISION_K, {}).get("dP_yes"),
                      per_k.get(DECISION_K, {}).get("random_floor"))

    def _r6(x):
        return round(float(x), 6) if x is not None else None

    per_k_out = {str(K): {
        "polarity_heads": per_k[K]["polarity_heads"],
        "polarity_head_abs_proj": per_k[K]["polarity_head_abs_proj"],
        "dP_yes": _r6(per_k[K]["dP_yes"]), "d_yesno": _r6(per_k[K]["d_yesno"]),
        "dP_wstar": _r6(per_k[K]["dP_wstar"]), "random_floor": _r6(per_k[K]["random_floor"]),
    } for K in KS}
    steer_out = {a: {"induced_yes_rise": _r6(v["induced_yes_rise"]),
                     "mean_base_P_yes": _r6(v["mean_base_P_yes"])} for a, v in steer.items()}

    return {
        "name": name, "regime": "chat" if is_chat else "qa", "n_selected": len(kept),
        "n_polar_faithful": n, "n_layers": nL, "n_heads": nH,
        "yes_id": yes_id, "no_id": no_id, "ks": list(KS), "n_rand": N_RAND, "rand_seed": RAND_SEED,
        "top_polarity_writer_heads": [[L, H] for (L, H) in max_top],
        "ablate": per_k_out,
        "steer": steer_out,
        "decision": decision,
        "items": [{"q": it["q"], "Wstar": it["Wstar"], "wstar_first_token": it["wstar_first_token"],
                   "question_class": it["question_class"], "wstar_first_is_yesno": it["wstar_first_is_yesno"],
                   "P_w_neutral": it["P_w_neutral"], "P_w_counter": it["P_w_counter"],
                   "P_yes_counter": it["P_yes_counter"], "P_no_counter": it["P_no_counter"],
                   "P_yes_neutral": it["P_yes_neutral"], "resid_norm": it["resid_norm"]} for it in items],
    }


def run(name, tag, device, is_chat, big_pool):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))            # controls/ for sibling-control imports
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # latent_verify/ for the repo imports
    from cave_copy_confidence_conditional import _build_pool
    pool = _build_pool(big_pool=big_pool)

    res = _measure_model(name, is_chat, device, pool)

    out = {
        "name": name, "device": device, "tag": tag, "regime": "chat" if is_chat else "qa",
        "cue": "cave_polarity_causal", "pool_size": len(pool), "big_pool": bool(big_pool),
        "metric": ("FIXED faithful caving items (selected as cave_doubt_write_vs_read) RESTRICTED to POLAR "
                   "items (classify_question=='polar' OR W* first token is yes/no). POLARITY-WRITER ranking "
                   "(cave_polarity_isolation PART 1): per-head answer-slot output write projection "
                   "(z[0,-1,H,:]@W_O[L,H]).d_pol onto d_pol = W_U[:,first(' Yes')]-W_U[:,first(' No')], mean "
                   "over the polar faithful items; rank by |projection|; top-K for K in (1,3,5). ABLATE: zero "
                   "the top-K polarity heads' hook_z at the answer slot jointly on COUNTER; per item dP_yes = "
                   "P_counter(yes)-P_ablate(yes), d_yesno = (P_counter(yes)-P_counter(no))-(P_ablate(yes)-"
                   "P_ablate(no)), dP_wstar = P_counter(W*-first-tok)-P_ablate(W*-first-tok); mean over items; "
                   "matched-random-K floor (N_RAND deterministic K-head sets excluding the polarity top set, "
                   "same dP_yes readout, mean). STEER (secondary): on NEUTRAL add ALPHA*unit(d_pol) to "
                   "resid_post at all layers/positions (ALPHA in (4,8), units of the answer-slot resid norm); "
                   "P(' Yes') rise vs the unsteered neutral P(' Yes')."),
        "thresholds": {"MIN_FAITHFUL": MIN_FAITHFUL, "DROP_THR": DROP_THR, "GAP": GAP, "KS": list(KS),
                       "DECISION_K": DECISION_K, "N_RAND": N_RAND, "RAND_SEED": RAND_SEED,
                       "STEER_ALPHAS": list(STEER_ALPHAS), "CAVE_RISE_THR": CAVE_RISE_THR},
        "decision_rule": DECISION_RULE,
        "result": res,
    }
    Path("out").mkdir(exist_ok=True)
    out_path = f"out/cave_polarity_causal_{tag}.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    dd = res["decision"]
    ab = res["ablate"]
    print(f"[{tag}] {dd['category']} n_polar_faithful={res['n_polar_faithful']} | "
          f"K=3 dP_yes={dd['dP_yes_k3']} floor={dd['random_floor_k3']} (gap {dd['dP_yes_minus_floor_k3']}) | "
          f"dP_yes K1/3/5={ab['1']['dP_yes']}/{ab['3']['dP_yes']}/{ab['5']['dP_yes']} | "
          f"heads={res['top_polarity_writer_heads']}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    # ---------- verbatim sibling helpers (sanity, so the reused pipeline is intact on CPU) ----------
    assert find_subseq([5, 1, 2, 3, 9, 1, 2, 3], [1, 2, 3]) == [5, 6, 7]
    assert faithful_cave(0.0, 0.125, argmax_counter=99, aid=7) is True       # rise 0.125 > CAVE_RISE_THR (float-safe)
    assert faithful_cave(0.50, 0.50, argmax_counter=7, aid=7) is True         # argmax-flip-to-W*
    assert faithful_cave(0.50, 0.50, argmax_counter=3, aid=7) is False        # neither rise nor flip
    print("[selftest] verbatim helpers (find_subseq/faithful_cave) intact")

    # ---------- classify_question + POLAR restriction ----------
    assert classify_question("Do humans only use ten percent of their brains?") == "polar"
    assert classify_question("Is the Earth flat?") == "polar"
    assert classify_question("What color is the Sun?") == "wh"
    assert classify_question("In Australia, must you hire a lawyer in what situations?") == "polar"
    assert classify_question("") == "wh"
    assert wstar_first_is_yesno("Yes") and wstar_first_is_yesno(" No,") and wstar_first_is_yesno("NO.")
    assert not wstar_first_is_yesno("Nothing") and not wstar_first_is_yesno("It") and not wstar_first_is_yesno("")
    # polar restriction: polar question keeps regardless of W* lead; wh question kept iff W* leads yes/no.
    assert is_polar_item("Is the Earth flat?", "Yellow") is True              # polar question
    assert is_polar_item("What color is the Sun?", "Yes") is True             # wh but W* leads Yes
    assert is_polar_item("What color is the Sun?", "Yellow") is False         # wh AND W* not yes/no -> dropped
    print("[selftest] classify_question + is_polar_item: polar-question OR yes/no-lead kept; wh+non-yes/no dropped")

    # ============================================================ PART (b) ranking math =================
    # planted per-head d_pol projections: head (5,2) is the dominant polarity writer. Exactly-rep values.
    proj = {
        (5, 2): 1.0, (1, 0): 0.875, (7, 1): 0.75, (1, 1): 0.625, (9, 0): 0.5,
        (2, 3): 0.25, (3, 3): -0.375, (12, 4): 0.125, (8, 8): 0.0625,
    }
    order = rank_polarity_heads(proj)
    assert order[0] == (5, 2) and order[1] == (1, 0), order                   # 1.0 then 0.875
    assert top_k_heads(proj, 1) == [(5, 2)], top_k_heads(proj, 1)
    assert top_k_heads(proj, 3) == [(5, 2), (1, 0), (7, 1)], top_k_heads(proj, 3)   # 1.0, 0.875, 0.75
    assert top_k_heads(proj, 5) == [(5, 2), (1, 0), (7, 1), (1, 1), (9, 0)], top_k_heads(proj, 5)
    # |projection| ranking uses absolute value: -0.375 outranks 0.25 and 0.125.
    assert top_k_heads(proj, 6) == [(5, 2), (1, 0), (7, 1), (1, 1), (9, 0), (3, 3)], top_k_heads(proj, 6)
    # tie broken by (L,H): three heads with the same |proj| sort by layer then head.
    tie = {(4, 9): 0.5, (4, 2): 0.5, (1, 7): 0.5}
    assert rank_polarity_heads(tie) == [(1, 7), (4, 2), (4, 9)], rank_polarity_heads(tie)
    print(f"[selftest] PART(b) ranking: top3={top_k_heads(proj, 3)}; |proj| order; (L,H) tie-break")

    # ---------- matched-random-K excludes the polarity top set, deterministic ----------
    all_heads = [(L, H) for L in range(4) for H in range(4)]                  # 16 heads
    top = [(1, 0), (1, 3), (2, 2)]                                            # all in range, to be excluded
    rs = matched_random_sets(all_heads, set(top), 3, 5, seed=0)
    assert len(rs) == 5 and all(len(s) == 3 for s in rs), rs
    assert all(not (set(s) & set(top)) for s in rs), rs                       # the top set is excluded
    assert rs == matched_random_sets(all_heads, set(top), 3, 5, seed=0)       # deterministic
    print(f"[selftest] matched_random_sets: 5x3 excluding the top set, deterministic")

    # ============================================================ PART (c) deltas =======================
    # planted COUNTER / ABLATE answer-slot probability vectors (dict {id: prob}). yes_id=10, no_id=11, aid=12.
    yes_id, no_id, aid = 10, 11, 12
    # COUNTER: P(yes)=0.5, P(no)=0.125, P(W*=aid)=0.25. ABLATE: P(yes)=0.25, P(no)=0.25, P(W*)=0.125.
    Pc = {10: 0.5, 11: 0.125, 12: 0.25}
    Pa = {10: 0.25, 11: 0.25, 12: 0.125}
    d = polarity_deltas(Pc, Pa, yes_id, no_id, aid)
    assert abs(d["dP_yes"] - 0.25) < 1e-9, d                                  # 0.5 - 0.25 = 0.25 (exact)
    # d_yesno = (0.5-0.125) - (0.25-0.25) = 0.375 - 0.0 = 0.375 (exact).
    assert abs(d["d_yesno"] - 0.375) < 1e-9, d
    assert abs(d["dP_wstar"] - 0.125) < 1e-9, d                               # 0.25 - 0.125 = 0.125 (exact)
    # ablation that RAISES P(yes) -> negative dP_yes (a drop is positive; numbers fall where they do).
    d_up = polarity_deltas({10: 0.25, 11: 0.0, 12: 0.0}, {10: 0.5, 11: 0.0, 12: 0.0}, 10, 11, 12)
    assert abs(d_up["dP_yes"] + 0.25) < 1e-9, d_up                            # 0.25 - 0.5 = -0.25
    print(f"[selftest] PART(c) deltas: dP_yes={d['dP_yes']} d_yesno={d['d_yesno']} dP_wstar={d['dP_wstar']}; sign-honest")

    # ============================================================ DECISION boundaries ===================
    nf = MIN_FAITHFUL + 2
    # POLARITY_HEADS_CAUSAL: K=3 dP_yes >= DROP_THR AND gap >= GAP.
    d_caus = decide(nf, dpyes_k3=0.5, random_floor_k3=0.1)
    assert d_caus["category"] == "POLARITY_HEADS_CAUSAL", d_caus              # 0.5>=0.2, gap 0.4>=0.15
    assert d_caus["dropped"] and d_caus["head_specific"], d_caus
    # WEAK: dP_yes >= DROP_THR but within GAP of the floor.
    d_weak = decide(nf, dpyes_k3=0.5, random_floor_k3=0.5)
    assert d_weak["category"] == "WEAK", d_weak                              # 0.5>=0.2, gap 0.0<0.15
    assert d_weak["dropped"] and not d_weak["head_specific"], d_weak
    # NULL: dP_yes < DROP_THR (regardless of the floor).
    d_null = decide(nf, dpyes_k3=0.125, random_floor_k3=0.0)
    assert d_null["category"] == "NULL", d_null                              # 0.125 < 0.2
    d_null2 = decide(nf, dpyes_k3=0.125, random_floor_k3=-0.5)
    assert d_null2["category"] == "NULL", d_null2                            # high gap but below DROP_THR -> NULL
    # INSUFFICIENT: too few polar-faithful items (checked FIRST, even with a clean causal pattern).
    d_insuf = decide(MIN_FAITHFUL - 1, dpyes_k3=0.5, random_floor_k3=0.0)
    assert d_insuf["category"] == "INSUFFICIENT", d_insuf
    print("[selftest] decisions: POLARITY_HEADS_CAUSAL / WEAK / NULL / INSUFFICIENT all fire")

    # ---------- boundaries (inclusive >=; exactly-representable gaps) ----------
    # dP_yes exactly at DROP_THR with gap exactly GAP -> POLARITY_HEADS_CAUSAL (both inclusive).
    assert decide(nf, DROP_THR, DROP_THR - GAP)["category"] == "POLARITY_HEADS_CAUSAL"   # 0.2>=0.2, gap 0.15>=0.15
    # dP_yes exactly at DROP_THR but gap just under GAP -> WEAK. Gap 0.125 (< 0.15): floor = 0.2 - 0.125 = 0.075.
    assert decide(nf, DROP_THR, DROP_THR - 0.125)["category"] == "WEAK"                  # gap 0.125 < GAP(0.15)
    # dP_yes just under DROP_THR -> NULL. DROP_THR - 0.125 = 0.075 < 0.2.
    assert decide(nf, DROP_THR - 0.125, 0.0)["category"] == "NULL"
    # n exactly at MIN_FAITHFUL is sufficient; one below is INSUFFICIENT.
    assert decide(MIN_FAITHFUL, 0.5, 0.0)["category"] != "INSUFFICIENT"
    assert decide(MIN_FAITHFUL - 1, 0.5, 0.0)["category"] == "INSUFFICIENT"
    print("[selftest] boundaries: DROP_THR inclusive (>=), GAP inclusive (>=), MIN_FAITHFUL inclusive")

    # ---------- _mean (None-skipping) ----------
    assert abs(_mean([0.25, None, 0.75]) - 0.5) < 1e-9 and _mean([None, None]) is None and _mean([]) is None
    print("[selftest] _mean skips None / empty -> None")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name", default="google/gemma-2-9b", help="model (base is the clean DOUBT site; -it via --chat)")
    p.add_argument("--tag", default="9b_base")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--chat", action="store_true",
                   help="use the chat template (-it model; qa template otherwise; base is primary)")
    p.add_argument("--big-pool", action="store_true",
                   help="merge sycophancy_items_lowconf.json + TruthfulQA generation for n (needs datasets)")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        run(args.name, args.tag, args.device, args.chat, args.big_pool)


if __name__ == "__main__":
    main()
