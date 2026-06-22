"""ROUTE / MEDIATION test: does the SPAN-ranked DOUBT heads' attention-knockout restoration ROUTE THROUGH the
downstream MLP output-carriers, or reach the logits without them?

CONTEXT (neutral). Two sibling numbers, on the SAME fixed faithful-caving items + the SAME span-ranked top-5
doubt heads, do not obviously agree about how much of the cave those heads carry:
  - cave_doubt_cue_attention / cave_headset_specificity / cave_doubt_write_vs_read: the span-ranked top-5 doubt
    heads' JOINT attention-to-doubt-span KNOCKOUT (zero + renormalize) restores the cave ~0.59 (the READ
    intervention -- it removes what those heads read off the doubt span).
  - cave_doubt_write_vs_read: those same heads' direct OUTPUT-patch (replace their COUNTER z with NEUTRAL z)
    restores only ~0.44. The gap (~0.59 - ~0.44) is carried DOWNSTREAM of the heads' own write.
  - cave_circuit_patch: a shape-agnostic ATP finder flagged the top downstream OUTPUT-carriers of the realized
    cave as distributed MLPs (the mlp-class confirmed set), NOT the doubt heads themselves.
This control tests the obvious bridge: is the doubt-heads' restoration MEDIATED BY those MLPs? It attaches no
hypothesis to any head, MLP, layer, or sign; it measures three restorations and lets the numbers decide whether
the doubt effect ROUTES THROUGH the MLPs (freezing them blocks it) or bypasses them (it survives the freeze).

WHAT IT MEASURES (gemma-2-9b BASE -- the clean DOUBT site; QA template; --big-pool for n), on a FIXED faithful
caving-item set selected ONCE under the DOUBT framing:
  (a) FIXED ITEM SET. _build_pool (incl. --big-pool); select_items (single-dominant near-margin); build
      NEUTRAL = push(q,C,NEUTRAL) and COUNTER = push(q,C,PUSH['counter'].format(W=W)) (the DOUBT framing
      'Actually, I think the answer is {W}. Are you sure?'). FAITHFUL cave iff the model realizes a shift toward
      W* under pushback: counter argmax == W*-first-tok OR realized P(W*) rises neutral->counter >=
      CAVE_RISE_THR (cave_circuit_patch.faithful_cave; first-token-collision items cid==aid skipped). Per
      faithful item cache, ONCE: the COUNTER prompt + the DOUBT span (challenge turn minus the W* span); the
      NEUTRAL per-layer MLP output mlp_out[0,-1] (the RESTORED MLP value, for ranking MLPs); and the COUNTER
      per-layer MLP output mlp_out[0,-1] (the UN-RESTORED MLP value, what the freeze PINS to).
  (b) SPAN RANKING (NOT ATP). Per head, answer-query attention TO the doubt span in COUNTER, mean over the
      fixed items (_answer_attn_to_span); rank_heads -> the SPAN-ranked top-5 doubt heads (L,H). The head set
      both doubt controls use, ranked the same way.
  (c) MLP RANKING. Per LAYER, the individual counter->neutral MLP-output-patch restore (patch that layer's
      NEUTRAL mlp_out into the COUNTER run at the answer slot, read the faithful cave_restoration), mean over
      the fixed items; rank descending -> the top-K(5) MLP carriers (the carriers the ATP finder flagged). A
      matched-random-K set of LAYERS not in the top-K is the carrier-identity floor (rank-irrelevant MLPs).
  (d) THREE faithful restores on the SAME 5 doubt heads + SAME items (mean over items, faithful
      cave_restoration -- relative drop in realized P(W*) OR argmax restored to the item's NEUTRAL answer;
      NEVER the logp-difference metric M):
        1. BASELINE: doubt-head ATTENTION-KO restore (zero the 5 heads' attention to the doubt span +
           renormalize jointly, _ko_heads_to) -> reproduces the ~0.59 read-gate result.
        2. ROUTE TEST (mediation): the SAME doubt-head ATTENTION-KO WHILE FREEZING the top-K MLPs at their
           COUNTER (un-restored) mlp_out -- run COUNTER with the doubt heads' attention-to-doubt zeroed AND a
           hook PINNING each top-K MLP's output to its normal-COUNTER value (so the MLPs CANNOT respond to the
           doubt-KO). If this is BLOCKED (restore drops toward 0), the doubt effect ROUTES THROUGH those MLPs;
           if it survives (~0.59), it bypasses them.
        3. CONTROL: the same attention-KO + freeze, but freezing a matched-random-K set of MLP LAYERS not in
           the top-K (does freezing ANY K MLPs block it, or specifically the top-K carriers?).
      Single K=5 (no per-K sweep): the spec fixes the carrier set at the top-K.

NEUTRAL DECISION (module constants MIN_FAITHFUL=8, RESTORE_THR=0.2, BLOCK_FRAC=0.5; numbers + categories only,
no hypothesis named, nothing said about which model/sign/component supports any claim):
  Let block_topk  = 1 - restore_with_topk_mlp_frozen  / baseline_restore   (fraction of the baseline blocked),
      block_rand  = 1 - restore_with_random_mlp_frozen / baseline_restore   (fraction blocked by the floor),
  with baseline_restore as the denominator (baseline ~0 -> NO_BASELINE before any block math).
    INSUFFICIENT       iff n_faithful < MIN_FAITHFUL(8)                                       (checked FIRST).
    NO_BASELINE        iff baseline_restore < RESTORE_THR(0.2)  (the ~0.59 read-gate did not reproduce; nothing
                          to mediate -- reported before the block tests).
    ROUTES_THROUGH_MLPS iff baseline_restore >= RESTORE_THR AND freezing the top-K MLPs reduces the restoration
                          by >= BLOCK_FRAC (restore_with_topk_mlp_frozen <= (1-BLOCK_FRAC)*baseline_restore)
                          AND the random-MLP freeze does NOT (block_rand < BLOCK_FRAC) -- the doubt effect is
                          mediated specifically by those MLPs.
    NONSPECIFIC        iff baseline_restore >= RESTORE_THR AND BOTH the top-K and the random freeze block by
                          >= BLOCK_FRAC -- freezing any K MLPs disrupts (not carrier-specific).
    DIRECT_OR_OTHER    iff baseline_restore >= RESTORE_THR AND freezing the top-K MLPs does NOT block (block_topk
                          < BLOCK_FRAC; restoration survives) -- the doubt effect reaches the logits without
                          routing through those MLPs.
  Resolution order: INSUFFICIENT -> NO_BASELINE -> NONSPECIFIC -> ROUTES_THROUGH_MLPS -> DIRECT_OR_OTHER
  (NONSPECIFIC checked before ROUTES so a both-block result is never mislabeled carrier-specific). All
  thresholds inclusive (>=). Reported: baseline_restore, restore_with_topk_mlp_frozen,
  restore_with_random_mlp_frozen, block_topk, block_rand, the top-K MLP layers, the top-5 doubt heads,
  n_faithful, the category.

Model-free --selftest (CPU, NO model load): synthetic (baseline, topk_frozen, random_frozen, n) tuples
verifying: top-K freeze blocks + random doesn't -> ROUTES_THROUGH_MLPS; top-K freeze doesn't block ->
DIRECT_OR_OTHER; both block -> NONSPECIFIC; baseline<thr -> NO_BASELINE; n<8 -> INSUFFICIENT; the inclusive >=
boundaries + the block-fraction math. find_subseq, doubt_span, faithful_cave, cave_restoration, rank_heads,
rank_layers, matched_random_layers, the answer-query attention readout, the JOINT attention-to-doubt-span
knockout (_ko_heads_to), the per-layer MLP-output patch (hook_mlp_out), and the MLP-output FREEZE are
RE-IMPLEMENTED below verbatim so --selftest is standalone on CPU (the same FLAT-scp convention the sibling
controls use -- on the box every file is scp'd flat into latent_verify/).

transformer_lens ONLY (NO circuit-tracer): every intervention is forward-only (a JOINT attention-pattern
knockout + per-layer hook_mlp_out PATCH/FREEZE hooks + full-softmax readouts; no backward). 9b fits an A100
40GB (one model resident); --big-pool needs `datasets`.

  python controls/cave_doubt_route.py --selftest
  python controls/cave_doubt_route.py --name google/gemma-2-9b --tag 9b_base --device cuda --big-pool
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured numbers only.
MIN_FAITHFUL = 8          # below this many faithful caving items -> INSUFFICIENT (under-powered)
RESTORE_THR = 0.2         # baseline attention-KO restore at/above this -> a reproduced read-gate (else NO_BASELINE)
BLOCK_FRAC = 0.5          # a freeze that reduces the restoration by >= this fraction of baseline counts as BLOCKING
CAVE_RISE_THR = 0.05      # realized P(W*) rise neutral->counter that counts as a faithful cave (argmax-flip also counts)

TOP_K = 5                 # the SPAN-ranked top-K doubt head set (the head set the doubt controls use)
MLP_K = 5                 # the top-K MLP carriers (ranked by their individual counter->neutral mlp_out patch restore)
RAND_K = 5                # matched-random-K MLP-layer set for the carrier-identity floor
RAND_SEED = 0             # deterministic matched-random set (headset_joint_patch convention)
N_RAND = 5                # # matched-random RAND_K-layer sets to average (variance reduction)

DECISION_RULE = (
    "FIXED faithful caving items (faithful_cave: counter argmax==W*-first-tok OR realized P(W*) rises "
    "neutral->counter >= CAVE_RISE_THR(0.05)) selected ONCE under the DOUBT framing COUNTER=push(q,C,"
    "PUSH['counter'].format(W=W)). SPAN ranking: per head answer-query attention TO the doubt span (challenge "
    "turn minus the W* span) in COUNTER, mean over the fixed items; rank_heads -> the span-ranked top-5 doubt "
    "heads. MLP ranking: per LAYER the individual counter->neutral mlp_out-patch restore (patch that layer's "
    "NEUTRAL mlp_out into COUNTER at the answer slot, read the faithful cave_restoration), mean over items; "
    "rank descending -> the top-MLP_K(5) carriers; a matched-random-K set of LAYERS not in the top-K is the "
    "carrier floor. THREE faithful restores on the SAME 5 doubt heads + SAME items (mean over items, faithful "
    "cave_restoration: restore_pw = max(0,(P_counter(W*)-P_int(W*))/P_counter(W*)) OR argmax restored to the "
    "item's NEUTRAL argmax; never M): (1) BASELINE = zero the 5 heads' attention to the doubt span + "
    "renormalize jointly (_ko_heads_to). (2) ROUTE TEST = the SAME attention-KO WHILE FREEZING the top-K MLPs' "
    "mlp_out at their normal-COUNTER (un-restored) value (a pin hook so the MLPs cannot respond to the KO). "
    "(3) CONTROL = the same KO + freeze on a matched-random-K MLP-layer set. block_topk = 1 - "
    "restore_with_topk_mlp_frozen/baseline_restore; block_rand = 1 - restore_with_random_mlp_frozen/"
    "baseline_restore. INSUFFICIENT iff n_faithful < MIN_FAITHFUL(8); else NO_BASELINE iff baseline_restore < "
    "RESTORE_THR(0.2); else NONSPECIFIC iff block_topk >= BLOCK_FRAC(0.5) AND block_rand >= BLOCK_FRAC; else "
    "ROUTES_THROUGH_MLPS iff block_topk >= BLOCK_FRAC AND block_rand < BLOCK_FRAC; else DIRECT_OR_OTHER. All "
    "thresholds inclusive (>=); numbers + categories only, no claim attached to any model, sign, head, MLP, or "
    "intervention."
)


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def find_subseq(hay, needle):
    """Last occurrence of `needle` in `hay` as a position list (verbatim from job_truthful_flip.find_subseq /
    cave_doubt_write_vs_read.find_subseq / cave_circuit_patch.find_subseq). Pure (selftest-able)."""
    if not needle:
        return []
    for i in range(len(hay) - len(needle), -1, -1):
        if hay[i:i + len(needle)] == needle:
            return list(range(i, i + len(needle)))
    return []


def doubt_span(challenge_pos, wstar_pos):
    """The DOUBT/CHALLENGE token span = the challenge-turn position list MINUS the W* answer-token positions
    (verbatim from cave_doubt_cue_attention.doubt_span / cave_circuit_patch.doubt_span). The pushback FRAMING
    tokens that express the user's doubt, EXCLUDING the asserted answer (so it is NOT the copy source). Pure."""
    excl = set(wstar_pos)
    return sorted(p for p in challenge_pos if p not in excl)


def faithful_cave(p_w_neutral, p_w_counter, argmax_counter, aid, cave_rise_thr=CAVE_RISE_THR):
    """Is this a FAITHFUL cave? The model realizes a shift toward W* under pushback iff the COUNTER argmax is
    the W*-first-tok OR the realized P(W*) rose neutral->counter by >= cave_rise_thr (verbatim from
    cave_circuit_patch.faithful_cave / cave_doubt_write_vs_read.faithful_cave). Pure."""
    argmax_is_w = (argmax_counter == aid)
    pw_rose = (p_w_counter - p_w_neutral) >= cave_rise_thr
    return bool(argmax_is_w or pw_rose)


def cave_restoration(p_w_counter, p_w_int, argmax_counter, argmax_int, aid, neu_argmax):
    """FAITHFUL per-item restoration from an intervention applied to COUNTER (verbatim readout from
    cave_circuit_patch.cave_restoration / cave_doubt_write_vs_read.cave_restoration; the only difference between
    interventions is the hook set, not this readout):
      restore_pw       = max(0, (P_counter(W*) - P_int(W*)) / P_counter(W*))  -- relative drop in realized
                         P(W*) (clamped at 0; a RISE in P(W*) is no restoration; P_counter~0 -> 0.0),
      argmax_restored  = (counter argmax == W*) AND (int argmax == the item's NEUTRAL-condition argmax),
      cave_restoration = max(restore_pw, argmax_restored).
    Pure (floats + ids -> dict). Never touches the logp-difference metric M."""
    restore_pw = (max(0.0, p_w_counter - p_w_int) / p_w_counter) if p_w_counter > 1e-9 else 0.0
    argmax_restored = bool(argmax_counter == aid and neu_argmax is not None and argmax_int == neu_argmax)
    return {"restore_pw": float(restore_pw), "argmax_restored": argmax_restored,
            "cave_restoration": float(max(restore_pw, 1.0 if argmax_restored else 0.0))}


def rank_heads(attn_self, top_k=TOP_K):
    """Rank heads by their attention TO the doubt span IN THIS MODEL (`attn_self`, descending), returning the
    top-k (L,H) tuples (verbatim from cave_headset_specificity.rank_heads / cave_doubt_write_vs_read.rank_heads
    -- this model's own attention only, the SPAN ranking; ties broken by (L,H) for determinism). Pure."""
    rows = sorted(attn_self, key=lambda k: (-float(attn_self[k]), k[0], k[1]))
    return [(L, H) for (L, H) in rows[:top_k]]


def rank_layers(mlp_restore, top_k=MLP_K):
    """Rank MLP LAYERS by their individual counter->neutral mlp_out-patch restore (descending), returning the
    top-k layer ints (the carrier ranking the ATP finder's mlp-class confirmed set motivates). Ties broken by
    layer index for determinism. Pure (dict layer->float -> list of ints)."""
    layers = sorted(mlp_restore, key=lambda L: (-float(mlp_restore[L]), L))
    return [int(L) for L in layers[:top_k]]


def matched_random_layers(all_layers, candidate_set, k, n_sets, seed=RAND_SEED):
    """n_sets deterministic matched-random k-LAYER sets drawn from layers NOT in `candidate_set` (the
    headset_joint_patch matched-random-K convention adapted from heads to MLP layers: a fixed-seed random
    control that excludes the candidate carriers). Returns a list of int-layer lists. Pure (deterministic
    RNG)."""
    pool = [L for L in all_layers if L not in set(candidate_set)]
    rng = random.Random(seed)
    k = min(k, len(pool))
    return [rng.sample(pool, k) for _ in range(n_sets)] if k > 0 else []


def block_fraction(baseline, frozen):
    """Fraction of the baseline restoration that a freeze removes: 1 - frozen/baseline, clamped to [0,1]
    (baseline <= 0 -> 0.0, no baseline to block). Pure (floats -> float)."""
    b = float(baseline)
    if b <= 1e-9:
        return 0.0
    frac = 1.0 - (float(frozen) / b)
    return float(max(0.0, min(1.0, frac)))


# --------------------------------------------------------------------------- pure decision
def decide(n_faithful, baseline_restore, topk_frozen_restore, random_frozen_restore,
           min_faithful=MIN_FAITHFUL, restore_thr=RESTORE_THR, block_frac=BLOCK_FRAC):
    """Neutral 5-way decision over the measured numbers only (no hypothesis attached to any model/sign/comp).
      n_faithful            : # faithful caving items (the fixed set).
      baseline_restore      : mean faithful restore from the doubt-head attention-KO (the ~0.59 read gate).
      topk_frozen_restore   : mean faithful restore from the SAME KO while freezing the top-K MLP carriers.
      random_frozen_restore : mean faithful restore from the SAME KO while freezing a matched-random-K MLP set.
    block_topk = block_fraction(baseline, topk_frozen); block_rand = block_fraction(baseline, random_frozen);
    a freeze BLOCKS iff its block fraction >= block_frac. Resolution order: INSUFFICIENT -> NO_BASELINE ->
    NONSPECIFIC -> ROUTES_THROUGH_MLPS -> DIRECT_OR_OTHER. All thresholds inclusive (>=). Pure."""
    def _f(x):
        return float(x) if x is not None else 0.0

    def _r(x):
        return round(float(x), 6) if x is not None else None

    base = _f(baseline_restore)
    topk = _f(topk_frozen_restore)
    rand = _f(random_frozen_restore)
    block_topk = block_fraction(base, topk)
    block_rand = block_fraction(base, rand)
    topk_blocks = block_topk >= block_frac
    rand_blocks = block_rand >= block_frac
    have_baseline = base >= restore_thr

    if n_faithful < min_faithful:
        cat = "INSUFFICIENT"
        msg = (f"only {n_faithful} faithful caving item(s) < MIN_FAITHFUL({min_faithful}); under-powered to "
               f"trace the doubt-head route through the MLPs (numbers still reported).")
    elif not have_baseline:
        cat = "NO_BASELINE"
        msg = (f"baseline doubt-head attention-KO restore {base:.3f} < RESTORE_THR({restore_thr}): the ~0.59 "
               f"read-gate restoration did not reproduce on this set -- nothing to mediate (numbers reported).")
    elif topk_blocks and rand_blocks:
        cat = "NONSPECIFIC"
        msg = (f"freezing the top-{MLP_K} MLP carriers blocks the restoration (block_topk={block_topk:.3f} >= "
               f"BLOCK_FRAC({block_frac})) AND freezing a matched-random-{RAND_K} MLP set blocks it too "
               f"(block_rand={block_rand:.3f} >= BLOCK_FRAC): freezing ANY K MLPs disrupts the doubt-KO -- not "
               f"carrier-specific.")
    elif topk_blocks and not rand_blocks:
        cat = "ROUTES_THROUGH_MLPS"
        msg = (f"freezing the top-{MLP_K} MLP carriers BLOCKS the restoration (restore_with_topk_mlp_frozen "
               f"{topk:.3f} <= (1-BLOCK_FRAC)*baseline = {(1 - block_frac) * base:.3f}; block_topk="
               f"{block_topk:.3f} >= BLOCK_FRAC({block_frac})) while the matched-random-{RAND_K} MLP freeze does "
               f"NOT (block_rand={block_rand:.3f} < BLOCK_FRAC): the doubt-head effect is mediated specifically "
               f"by those MLPs.")
    else:
        cat = "DIRECT_OR_OTHER"
        msg = (f"freezing the top-{MLP_K} MLP carriers does NOT block the restoration (restore_with_topk_mlp_"
               f"frozen {topk:.3f}; block_topk={block_topk:.3f} < BLOCK_FRAC({block_frac})): the doubt-head "
               f"effect reaches the logits without routing through those MLPs.")
    return {"category": cat,
            "n_faithful": n_faithful,
            "baseline_restore": _r(baseline_restore),
            "restore_with_topk_mlp_frozen": _r(topk_frozen_restore),
            "restore_with_random_mlp_frozen": _r(random_frozen_restore),
            "block_topk": round(block_topk, 6), "block_rand": round(block_rand, 6),
            "topk_blocks": bool(topk_blocks), "rand_blocks": bool(rand_blocks),
            "have_baseline": bool(have_baseline),
            "min_faithful": min_faithful, "restore_thr": restore_thr, "block_frac": block_frac,
            "top_k": TOP_K, "mlp_k": MLP_K, "rand_k": RAND_K,
            "msg": msg}


# --------------------------------------------------------------------------- real-run helpers
def _full_softmax(logits):
    """Full next-token probability vector at the LAST position (gemma-2's final softcap is applied inside the
    forward, so softmax(logits[0,-1]) is the realized distribution; same convention as the sibling controls).
    Returns a 1-D float tensor."""
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _patname(L):
    """attention-pattern hook name at layer L (job_truthful_flip / cave_doubt_cue_attention convention)."""
    return f"blocks.{L}.attn.hook_pattern"


def _mname(L):
    """MLP output hook name at layer L (cave_circuit_patch._mname / cave_direction_dla._mname convention)."""
    return f"blocks.{L}.hook_mlp_out"


def _answer_attn_to_span(model, ids, positions, layers, nH):
    """Per-head attention mass FROM the answer/last position TO the key `positions`, at each layer in `layers`,
    in ONE forward (verbatim from cave_doubt_cue_attention._answer_attn_to_span / cave_doubt_write_vs_read.
    _answer_attn_to_span: grab the [head, query, key] pattern, take the last-query row, sum over the span key
    positions). Returns {(L,H): float}; positions empty -> all 0.0. Forward-only."""
    store = {}

    def grab(p, hook):
        store[hook.layer()] = p[0, :, -1, :].detach().float()      # [head, key] at the answer query
        return p

    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(_patname(L), grab) for L in layers])
    return {(L, H): (float(store[L][H, positions].sum()) if positions else 0.0)
            for L in layers for H in range(nH)}


def _ko_heads_to(head_positions, span_positions):
    """JOINT attention-pattern knockout TO `span_positions` + per-head renormalize, for a SET of (L,H) heads
    (verbatim from cave_doubt_write_vs_read._ko_heads_to / cave_doubt_cue_attention._ko_heads_to: the per-head
    ko_head mechanics from job_truthful_flip.ko_head, grouped by layer so a top-K set spanning multiple layers
    is knocked out jointly in ONE forward). Each hook zeroes its layer's listed heads' attention to the span and
    renormalizes only those heads' rows. Returns a list of (hook_name, hook). This is the READ intervention."""
    by_layer = {}
    for (L, H) in head_positions:
        by_layer.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in by_layer.items():
        Hs = sorted(set(Hs))

        def hook(p, hook, Hs=Hs, span=span_positions):
            for H in Hs:
                p[:, H, :, span] = 0.0
                p[:, H] = p[:, H] / p[:, H].sum(-1, keepdim=True).clamp_min(1e-9)
            return p
        hooks.append((_patname(L), hook))
    return hooks


def _mlp_freeze_hooks(counter_mlp_out, layers):
    """Per-layer hook_mlp_out FREEZE hooks: PIN each listed layer's MLP output at the answer (last) position to
    its cached normal-COUNTER (un-restored) value (the mediation freeze -- the MLP cannot respond to the
    upstream attention knockout). counter_mlp_out = {L: [d_model]} cached on the plain COUNTER prompt for THIS
    item. This is the per-layer hook_mlp_out patch form of cave_circuit_patch._confirm_set's MLP branch, but
    pinned to the COUNTER value rather than the NEUTRAL value. Returns a list of (hook_name, hook)."""
    hooks = []
    for L in layers:
        mval = counter_mlp_out[L]

        def mfreeze(m, hook, mval=mval):
            m[0, -1, :] = mval.to(m.dtype)
            return m
        hooks.append((_mname(L), mfreeze))
    return hooks


def _restore_from_hooks(model, counter_ids, fwd_hooks, aid, ctr_argmax, neu_argmax, p_w_ctr):
    """Run COUNTER with the given fwd_hooks, read the realized answer-slot softmax, and return the FAITHFUL
    cave_restoration scalar for ONE item (forward-only). Empty hooks -> 0.0 (no intervention)."""
    if not fwd_hooks:
        return 0.0
    with torch.no_grad():
        lg = model.run_with_hooks(counter_ids, fwd_hooks=fwd_hooks)
    P = _full_softmax(lg)
    cr = cave_restoration(p_w_ctr, float(P[aid]), ctr_argmax, int(P.argmax()), aid, neu_argmax)
    return cr["cave_restoration"]


def _mlp_patch_restore(model, counter_ids, neutral_mlp_out, L, aid, ctr_argmax, neu_argmax, p_w_ctr):
    """Activation-patch ONE layer's MLP OUTPUT: write its cached NEUTRAL mlp_out into the COUNTER run at the
    answer (last) position, read the realized answer-slot softmax, return the FAITHFUL cave_restoration for ONE
    item (the per-layer counter->neutral mlp_out patch the MLP ranking uses -- the MLP branch of
    cave_circuit_patch._confirm_component). Forward-only."""
    mval = neutral_mlp_out[L]

    def mpatch(m, hook, mval=mval):
        m[0, -1, :] = mval.to(m.dtype)
        return m
    with torch.no_grad():
        lg = model.run_with_hooks(counter_ids, fwd_hooks=[(_mname(L), mpatch)])
    P = _full_softmax(lg)
    cr = cave_restoration(p_w_ctr, float(P[aid]), ctr_argmax, int(P.argmax()), aid, neu_argmax)
    return cr["cave_restoration"]


def _measure_model(name, is_chat, device, pool):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident.
    (a) Select the FIXED faithful caving items; per item cache the COUNTER prompt, the doubt span, the NEUTRAL
    per-layer mlp_out (the RESTORED MLP value, for ranking), and the COUNTER per-layer mlp_out (the UN-RESTORED
    value, what the freeze pins to). (b) Rank the span-ranked top-5 doubt heads. (c) Rank the top-K MLP carriers
    by their individual counter->neutral mlp_out-patch restore + draw the matched-random-K layer floor. (d)
    Compute the three restores (BASELINE attention-KO, ROUTE TEST KO+top-K-MLP-freeze, CONTROL KO+random-MLP-
    freeze). Returns the per-model record + decision."""
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL, select_items
    from rlhf_differential import _helpers

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    layers = list(range(nL))
    all_layers = list(range(nL))
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)

    # ---- (a) FIXED ITEM SET: single-dominant near-margin items, then the faithful-cave gate (selected ONCE) ----
    lp_single = lambda it, text: num_lp(single(it["q"]), text)
    kept = select_items(pool, lp_single)
    print(f"[{tag}] selected {len(kept)}/{len(pool)} single-competitor near-margin items", flush=True)

    items = []
    attn_acc = {(L, H): 0.0 for L in layers for H in range(nH)}    # mean answer->doubt attn over faithful items
    for r in kept:
        q, C, W = r["q"], r["correct"], r["Wstar"]
        cid, aid = first(" " + C), first(" " + W)             # FIRST-token ids = the realized readout register
        if cid == aid:                                        # first-token collision -> realized readout degenerate
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))

        # NEUTRAL per-layer mlp_out (RESTORED value, cached ONCE for the MLP ranking patch) + realized readouts;
        # COUNTER per-layer mlp_out (UN-RESTORED value, what the freeze pins to) + realized readout.
        mneu, mctr = {}, {}

        def grab_mn(m, hook):
            mneu[hook.layer()] = m[0, -1].detach().clone(); return m       # [d_model]

        def grab_mc(m, hook):
            mctr[hook.layer()] = m[0, -1].detach().clone(); return m       # [d_model]
        with torch.no_grad():
            lg_n = model.run_with_hooks(neutral, fwd_hooks=[(_mname(L), grab_mn) for L in range(nL)])
            lg_c = model.run_with_hooks(counter, fwd_hooks=[(_mname(L), grab_mc) for L in range(nL)])
        Pn, Pc = _full_softmax(lg_n), _full_softmax(lg_c)
        neu_argmax = int(Pn.argmax())
        ctr_argmax = int(Pc.argmax())
        p_w_neu, p_w_ctr = float(Pn[aid]), float(Pc[aid])

        if not faithful_cave(p_w_neu, p_w_ctr, ctr_argmax, aid):
            continue

        # DOUBT span = challenge-turn tokens MINUS the W* answer-token span.
        ctoks = counter[0].tolist()
        challenge_text = PUSH["counter"].format(W=W)
        chal_pos = (find_subseq(ctoks, raw(" " + challenge_text.strip(), bos=False)[0].tolist())
                    or find_subseq(ctoks, raw(challenge_text.strip(), bos=False)[0].tolist()))
        Wpos = (find_subseq(ctoks, raw(" " + W.strip(), bos=False)[0].tolist())
                or find_subseq(ctoks, raw(W.strip(), bos=False)[0].tolist()))
        dpos = doubt_span(chal_pos, Wpos)
        if not dpos:                                          # could not isolate a doubt span -> skip (logged)
            print(f"  [{tag}] no doubt span isolated (chal={len(chal_pos)} W*={len(Wpos)}) q={q[:34]!r}",
                  flush=True)
            continue

        # answer-query per-head attention TO the doubt span (COUNTER), all layers (for the SPAN ranking).
        attn = _answer_attn_to_span(model, counter, dpos, layers, nH)
        for k in attn_acc:
            attn_acc[k] += attn[k]

        items.append({"q": q, "cid": cid, "aid": aid, "neu_argmax": neu_argmax, "ctr_argmax": ctr_argmax,
                      "P_w_neutral": round(p_w_neu, 6), "P_w_counter": round(p_w_ctr, 6),
                      "doubt_span_len": len(dpos), "wstar_span_len": len(Wpos),
                      "_counter": counter, "_mneu": mneu, "_mctr": mctr, "_dpos": dpos})
        print(f"  [{tag}] faithful P(W*) n/c={p_w_neu:.3f}/{p_w_ctr:.3f} doubt_len={len(dpos)} "
              f"W*_len={len(Wpos)} q={q[:34]!r}", flush=True)

    n = len(items)
    print(f"[{tag}] n_faithful={n}", flush=True)
    attn_mean = {(L, H): (attn_acc[(L, H)] / n if n else 0.0) for L in layers for H in range(nH)}

    # ---- (b) SPAN ranking (NOT ATP): top-5 doubt heads by this model's own answer->doubt attention ----
    doubt_heads = rank_heads(attn_mean, TOP_K)
    print(f"[{tag}] span-ranked top-{TOP_K} doubt heads = {doubt_heads}", flush=True)

    # ---- (c) MLP ranking: per-layer individual counter->neutral mlp_out-patch restore (mean over items) ----
    mlp_restore_acc = {L: [] for L in layers}
    for it in items:
        counter, mneu = it["_counter"], it["_mneu"]
        aid, ctr_argmax, neu_argmax, p_w_ctr = it["aid"], it["ctr_argmax"], it["neu_argmax"], it["P_w_counter"]
        for L in layers:
            mlp_restore_acc[L].append(
                _mlp_patch_restore(model, counter, mneu, L, aid, ctr_argmax, neu_argmax, p_w_ctr))
    mlp_restore = {L: (statistics.mean(mlp_restore_acc[L]) if mlp_restore_acc[L] else 0.0) for L in layers}
    topk_mlps = rank_layers(mlp_restore, MLP_K)
    rand_layer_sets = matched_random_layers(all_layers, set(topk_mlps), RAND_K, N_RAND, RAND_SEED)
    print(f"[{tag}] top-{MLP_K} MLP carriers = {topk_mlps} "
          f"(restores {[round(mlp_restore[L], 3) for L in topk_mlps]})", flush=True)

    # ---- (d) three restores on the SAME doubt heads + SAME items (mean over items) ----
    baseline_acc, topk_frozen_acc, random_frozen_acc = [], [], []
    for it in items:
        counter, dpos, mctr = it["_counter"], it["_dpos"], it["_mctr"]
        aid, ctr_argmax, neu_argmax, p_w_ctr = it["aid"], it["ctr_argmax"], it["neu_argmax"], it["P_w_counter"]
        ko_hooks = _ko_heads_to(doubt_heads, dpos)
        # 1. BASELINE: doubt-head attention-KO only (the ~0.59 read gate).
        base = _restore_from_hooks(model, counter, ko_hooks, aid, ctr_argmax, neu_argmax, p_w_ctr)
        # 2. ROUTE TEST: the SAME KO + freeze the top-K MLPs at their normal-COUNTER (un-restored) mlp_out.
        topk_hooks = ko_hooks + _mlp_freeze_hooks(mctr, topk_mlps)
        topk_frozen = _restore_from_hooks(model, counter, topk_hooks, aid, ctr_argmax, neu_argmax, p_w_ctr)
        # 3. CONTROL: the SAME KO + freeze a matched-random-K MLP-layer set (mean over N_RAND), the floor.
        if rand_layer_sets:
            rk = [_restore_from_hooks(model, counter, ko_hooks + _mlp_freeze_hooks(mctr, rs),
                                      aid, ctr_argmax, neu_argmax, p_w_ctr) for rs in rand_layer_sets]
            random_frozen = statistics.mean(rk)
        else:
            random_frozen = 0.0
        baseline_acc.append(base)
        topk_frozen_acc.append(topk_frozen)
        random_frozen_acc.append(random_frozen)
        print(f"  [{tag} INT] baseline={base:.3f} topk_frozen={topk_frozen:.3f} "
              f"random_frozen={random_frozen:.3f}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    baseline_restore = (statistics.mean(baseline_acc) if baseline_acc else None)
    topk_frozen_restore = (statistics.mean(topk_frozen_acc) if topk_frozen_acc else None)
    random_frozen_restore = (statistics.mean(random_frozen_acc) if random_frozen_acc else None)
    decision = decide(n, baseline_restore, topk_frozen_restore, random_frozen_restore)

    return {
        "name": name, "regime": "chat" if is_chat else "qa", "n_selected": len(kept),
        "n_faithful": n, "n_layers": nL, "n_heads": nH,
        "top_k": TOP_K, "mlp_k": MLP_K, "rand_k": RAND_K, "n_rand": N_RAND,
        "span_ranked_doubt_heads": [[L, H] for (L, H) in doubt_heads],
        "topk_mlp_layers": list(topk_mlps),
        "topk_mlp_individual_restore": {int(L): round(mlp_restore[L], 6) for L in topk_mlps},
        "baseline_restore": (round(baseline_restore, 6) if baseline_restore is not None else None),
        "restore_with_topk_mlp_frozen": (round(topk_frozen_restore, 6) if topk_frozen_restore is not None else None),
        "restore_with_random_mlp_frozen": (round(random_frozen_restore, 6) if random_frozen_restore is not None else None),
        "decision": decision,
        "items": [{"q": it["q"], "P_w_neutral": it["P_w_neutral"], "P_w_counter": it["P_w_counter"],
                   "doubt_span_len": it["doubt_span_len"], "wstar_span_len": it["wstar_span_len"]}
                  for it in items],
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
        "cue": "cave_doubt_route", "pool_size": len(pool), "big_pool": bool(big_pool),
        "metric": ("FIXED faithful caving-item set (selected once under the DOUBT framing); SPAN-ranked top-5 "
                   "doubt heads (answer->doubt-span attention, mean over the fixed items); top-K MLP carriers "
                   "ranked by their individual counter->neutral mlp_out-patch restore; three JOINT faithful "
                   "restores on the SAME 5 doubt heads + SAME items (faithful cave_restoration: relative drop "
                   "in realized P(W*) OR argmax restored to the neutral answer): BASELINE (doubt-head "
                   "attention-to-doubt-span KO + renormalize), ROUTE TEST (the SAME KO while FREEZING the top-K "
                   "MLPs' mlp_out at their normal-COUNTER value -- a mediation freeze), CONTROL (the SAME KO + "
                   "freeze on a matched-random-K MLP-layer set, the carrier-identity floor)"),
        "thresholds": {"MIN_FAITHFUL": MIN_FAITHFUL, "RESTORE_THR": RESTORE_THR, "BLOCK_FRAC": BLOCK_FRAC,
                       "CAVE_RISE_THR": CAVE_RISE_THR, "TOP_K": TOP_K, "MLP_K": MLP_K, "RAND_K": RAND_K,
                       "RAND_SEED": RAND_SEED, "N_RAND": N_RAND},
        "decision_rule": DECISION_RULE,
        "result": res,
    }
    Path("out").mkdir(exist_ok=True)
    out_path = f"out/cave_doubt_route_{tag}.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    dd = res["decision"]
    print(f"[{tag}] {dd['category']} n_faithful={res['n_faithful']} "
          f"baseline={dd['baseline_restore']} topk_frozen={dd['restore_with_topk_mlp_frozen']} "
          f"random_frozen={dd['restore_with_random_mlp_frozen']} "
          f"(block_topk={dd['block_topk']} block_rand={dd['block_rand']}) | "
          f"doubt_heads={res['span_ranked_doubt_heads']} topk_mlps={res['topk_mlp_layers']}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    # ---------- find_subseq + doubt_span (verbatim mirrors of the sibling controls) ----------
    assert find_subseq([5, 1, 2, 3, 9, 1, 2, 3], [1, 2, 3]) == [5, 6, 7]   # LAST occurrence
    assert find_subseq([1, 2, 3], [9]) == [] and find_subseq([1, 2], []) == []
    chal = list(range(4, 14))                  # "Actually, I think the answer is {W}. Are you sure?"
    wstar = [9, 10]                            # the asserted W* answer-token span inside the challenge turn
    dsp = doubt_span(chal, wstar)
    assert dsp == [4, 5, 6, 7, 8, 11, 12, 13], dsp
    assert not (set(dsp) & set(wstar)), "doubt span must EXCLUDE the W* answer-token span (copy source)"
    assert doubt_span(chal, [99, 100]) == chal and doubt_span([], [9, 10]) == []
    print(f"[selftest] doubt_span = challenge MINUS W* -> {dsp} (W* {wstar} excluded)")

    # ---------- faithful_cave gate (verbatim mirror) ----------
    cid, aid = 3, 7
    assert faithful_cave(0.05, 0.06, argmax_counter=aid, aid=aid) is True            # argmax-flip-to-W*
    assert faithful_cave(0.05, 0.05 + CAVE_RISE_THR, argmax_counter=cid, aid=aid) is True   # P(W*) rise
    assert faithful_cave(0.05, 0.06, argmax_counter=cid, aid=aid) is False           # neither
    assert faithful_cave(0.10, 0.10 + CAVE_RISE_THR, argmax_counter=cid, aid=aid) is True   # boundary >=
    assert faithful_cave(0.10, 0.10 + CAVE_RISE_THR - 1e-4, argmax_counter=cid, aid=aid) is False
    print("[selftest] faithful_cave: argmax-flip OR P(W*) rise >= CAVE_RISE_THR (boundary inclusive)")

    # ---------- cave_restoration (the shared faithful readout for ALL interventions) ----------
    cr = cave_restoration(p_w_counter=0.60, p_w_int=0.15, argmax_counter=aid, argmax_int=cid, aid=aid,
                          neu_argmax=cid)
    assert abs(cr["restore_pw"] - 0.75) < 1e-9 and cr["argmax_restored"] is True, cr
    assert cr["cave_restoration"] == 1.0, cr                              # argmax restored dominates (max channel)
    cr_rise = cave_restoration(0.60, 0.70, argmax_counter=aid, argmax_int=aid, aid=aid, neu_argmax=cid)
    assert cr_rise["restore_pw"] == 0.0 and cr_rise["cave_restoration"] == 0.0, cr_rise
    cr_drop = cave_restoration(0.60, 0.30, argmax_counter=aid, argmax_int=99, aid=aid, neu_argmax=cid)
    assert abs(cr_drop["restore_pw"] - 0.5) < 1e-9 and cr_drop["argmax_restored"] is False, cr_drop
    assert abs(cr_drop["cave_restoration"] - 0.5) < 1e-9, cr_drop
    assert cave_restoration(0.0, 0.0, cid, cid, aid, cid)["restore_pw"] == 0.0      # P_counter~0 -> no div-by-zero
    print(f"[selftest] cave_restoration: drop+argmax={cr['cave_restoration']} rise->{cr_rise['cave_restoration']} "
          f"drop-only={cr_drop['cave_restoration']:.3f}")

    # ---------- rank_heads (the SPAN ranking, ties by (L,H)) ----------
    attn = {(3, 0): 0.40, (12, 1): 0.35, (12, 2): 0.10, (20, 5): 0.10}
    assert rank_heads(attn, 2) == [(3, 0), (12, 1)], rank_heads(attn, 2)             # desc by attn
    assert rank_heads(attn, 4)[2:] == [(12, 2), (20, 5)], rank_heads(attn, 4)        # tie 0.10 -> (L,H) order
    assert rank_heads(attn, 10) == rank_heads(attn, 10)                              # deterministic
    print(f"[selftest] rank_heads top2={rank_heads(attn, 2)} (desc attn, ties by (L,H))")

    # ---------- rank_layers (MLP carriers by individual mlp_out-patch restore, ties by layer) ----------
    mlp_r = {10: 0.40, 22: 0.55, 30: 0.10, 31: 0.10, 5: 0.02}
    assert rank_layers(mlp_r, 2) == [22, 10], rank_layers(mlp_r, 2)                   # desc by restore
    assert rank_layers(mlp_r, 4)[2:] == [30, 31], rank_layers(mlp_r, 4)              # tie 0.10 -> layer order
    assert rank_layers(mlp_r, 10) == rank_layers(mlp_r, 10)                           # deterministic
    print(f"[selftest] rank_layers top2={rank_layers(mlp_r, 2)} (desc restore, ties by layer)")

    # ---------- matched_random_layers (deterministic, excludes the candidate carriers) ----------
    all_layers = list(range(42))
    rs = matched_random_layers(all_layers, set([22, 10, 30, 31, 5]), 5, 3, seed=0)
    assert len(rs) == 3 and all(len(s) == 5 for s in rs), rs
    assert all(all(L not in (22, 10, 30, 31, 5) for L in s) for s in rs), rs         # excludes the carrier set
    assert rs == matched_random_layers(all_layers, set([22, 10, 30, 31, 5]), 5, 3, seed=0)   # deterministic
    print(f"[selftest] matched_random_layers: 3 sets of 5 MLP layers, exclude carriers, deterministic")

    # ---------- block_fraction (the mediation block math) ----------
    assert abs(block_fraction(0.60, 0.30) - 0.5) < 1e-9                               # half blocked
    assert abs(block_fraction(0.60, 0.06) - 0.9) < 1e-9                               # 90% blocked
    assert block_fraction(0.60, 0.60) == 0.0                                          # fully survives -> 0 blocked
    assert block_fraction(0.60, 0.90) == 0.0                                          # rose -> clamp to 0
    assert block_fraction(0.0, 0.0) == 0.0 and block_fraction(0.0, 0.5) == 0.0        # no baseline -> 0
    assert block_fraction(0.60, 0.0) == 1.0                                           # fully blocked -> 1
    print("[selftest] block_fraction: 1 - frozen/baseline, clamped to [0,1]")

    # ============================================================ DECISION scenarios ===================
    nf = MIN_FAITHFUL + 3
    # (i) ROUTES_THROUGH_MLPS: top-K freeze blocks (~0.59 -> ~0.10) AND the random freeze does not (~0.55).
    d_route = decide(nf, baseline_restore=0.59, topk_frozen_restore=0.10, random_frozen_restore=0.55)
    assert d_route["category"] == "ROUTES_THROUGH_MLPS", d_route
    assert d_route["topk_blocks"] and not d_route["rand_blocks"], d_route
    # (ii) DIRECT_OR_OTHER: top-K freeze does NOT block (restoration survives at ~0.55).
    d_direct = decide(nf, baseline_restore=0.59, topk_frozen_restore=0.55, random_frozen_restore=0.55)
    assert d_direct["category"] == "DIRECT_OR_OTHER", d_direct
    assert not d_direct["topk_blocks"], d_direct
    # (iii) NONSPECIFIC: BOTH the top-K and the random freeze block similarly (freezing any K MLPs disrupts).
    d_nonspec = decide(nf, baseline_restore=0.59, topk_frozen_restore=0.10, random_frozen_restore=0.12)
    assert d_nonspec["category"] == "NONSPECIFIC", d_nonspec
    assert d_nonspec["topk_blocks"] and d_nonspec["rand_blocks"], d_nonspec
    # (iv) NO_BASELINE: the ~0.59 read gate did not reproduce (baseline < RESTORE_THR), checked before blocks.
    d_nobase = decide(nf, baseline_restore=0.10, topk_frozen_restore=0.0, random_frozen_restore=0.0)
    assert d_nobase["category"] == "NO_BASELINE", d_nobase
    assert not d_nobase["have_baseline"], d_nobase
    # (v) INSUFFICIENT: too few faithful items (checked FIRST, even with a clean routes-through result).
    d_insuf = decide(MIN_FAITHFUL - 1, baseline_restore=0.59, topk_frozen_restore=0.10, random_frozen_restore=0.55)
    assert d_insuf["category"] == "INSUFFICIENT", d_insuf
    print("[selftest] decisions: ROUTES_THROUGH_MLPS / DIRECT_OR_OTHER / NONSPECIFIC / NO_BASELINE / "
          "INSUFFICIENT all fire")

    # ---------- block-fraction math inside decide (block_topk / block_rand reported) ----------
    assert abs(d_route["block_topk"] - block_fraction(0.59, 0.10)) < 1e-6, d_route  # decision rounds to 6dp
    assert abs(d_route["block_rand"] - block_fraction(0.59, 0.55)) < 1e-6, d_route
    # the (1-BLOCK_FRAC)*baseline phrasing in the spec matches the block_topk >= BLOCK_FRAC test:
    # frozen <= (1-BLOCK_FRAC)*baseline  <=>  1 - frozen/baseline >= BLOCK_FRAC.
    base = 0.59
    frozen_at_edge = (1 - BLOCK_FRAC) * base
    assert abs(block_fraction(base, frozen_at_edge) - BLOCK_FRAC) < 1e-9
    print(f"[selftest] block math: block_topk={d_route['block_topk']:.3f} block_rand={d_route['block_rand']:.3f}; "
          f"frozen<=(1-BLOCK_FRAC)*baseline <=> block_frac>=BLOCK_FRAC")

    # ---------- threshold boundaries (inclusive >=) ----------
    # n exactly at MIN_FAITHFUL is sufficient; one below is INSUFFICIENT.
    assert decide(MIN_FAITHFUL, 0.59, 0.10, 0.55)["category"] != "INSUFFICIENT"
    assert decide(MIN_FAITHFUL - 1, 0.59, 0.10, 0.55)["category"] == "INSUFFICIENT"
    # RESTORE_THR boundary: baseline exactly at THR escapes NO_BASELINE; just under -> NO_BASELINE.
    # at baseline==RESTORE_THR with topk fully blocked + random surviving -> ROUTES_THROUGH_MLPS.
    assert decide(nf, RESTORE_THR, 0.0, RESTORE_THR)["category"] == "ROUTES_THROUGH_MLPS"
    assert decide(nf, RESTORE_THR - 1e-6, 0.0, RESTORE_THR)["category"] == "NO_BASELINE"
    # BLOCK_FRAC boundary (top-K): frozen exactly at (1-BLOCK_FRAC)*baseline -> block_topk == BLOCK_FRAC -> blocks
    # (inclusive >=); a hair above -> does not block -> DIRECT_OR_OTHER (random kept surviving).
    base = 0.60
    edge = (1 - BLOCK_FRAC) * base                                       # block_topk == BLOCK_FRAC exactly
    assert decide(nf, base, edge, base)["category"] == "ROUTES_THROUGH_MLPS", decide(nf, base, edge, base)
    assert decide(nf, base, edge + 1e-3, base)["category"] == "DIRECT_OR_OTHER", decide(nf, base, edge + 1e-3, base)
    # BLOCK_FRAC boundary (random): both at the edge -> NONSPECIFIC (both block, inclusive >=).
    assert decide(nf, base, edge, edge)["category"] == "NONSPECIFIC", decide(nf, base, edge, edge)
    # random a hair above the edge (does not block) while top-K blocks -> ROUTES_THROUGH_MLPS.
    assert decide(nf, base, edge, edge + 1e-3)["category"] == "ROUTES_THROUGH_MLPS", decide(nf, base, edge, edge + 1e-3)
    print("[selftest] boundaries (MIN_FAITHFUL, RESTORE_THR, BLOCK_FRAC top-K & random) inclusive-OK")

    # ============================================================ END-TO-END synthetic pipeline =========
    # Build synthetic per-item (baseline, topk_frozen, random_frozen) tuples and aggregate exactly as
    # _measure_model does (mean over items), then decide.
    def e2e(per_item, n_faithful):
        base = statistics.mean(x[0] for x in per_item)
        topk = statistics.mean(x[1] for x in per_item)
        rand = statistics.mean(x[2] for x in per_item)
        return decide(n_faithful, base, topk, rand)

    # (i) ROUTES_THROUGH_MLPS: every item's top-K freeze collapses the restore; the random freeze leaves it.
    route_items = [(0.59, 0.08, 0.55)] * (MIN_FAITHFUL + 2)
    de1 = e2e(route_items, len(route_items))
    assert de1["category"] == "ROUTES_THROUGH_MLPS", de1
    # (ii) DIRECT_OR_OTHER: the doubt-KO restore survives the top-K MLP freeze (it bypasses those MLPs).
    direct_items = [(0.59, 0.56, 0.55)] * (MIN_FAITHFUL + 2)
    de2 = e2e(direct_items, len(direct_items))
    assert de2["category"] == "DIRECT_OR_OTHER", de2
    # (iii) NONSPECIFIC: both freezes block similarly (freezing any K MLPs disrupts the KO).
    nonspec_items = [(0.59, 0.08, 0.10)] * (MIN_FAITHFUL + 2)
    de3 = e2e(nonspec_items, len(nonspec_items))
    assert de3["category"] == "NONSPECIFIC", de3
    # (iv) NO_BASELINE: the read-gate restore did not reproduce.
    nobase_items = [(0.05, 0.01, 0.02)] * (MIN_FAITHFUL + 2)
    de4 = e2e(nobase_items, len(nobase_items))
    assert de4["category"] == "NO_BASELINE", de4
    # (v) INSUFFICIENT: too few faithful items.
    de5 = e2e(route_items[:MIN_FAITHFUL - 1], MIN_FAITHFUL - 1)
    assert de5["category"] == "INSUFFICIENT", de5
    print(f"[selftest] end-to-end: ROUTES_THROUGH_MLPS / DIRECT_OR_OTHER / NONSPECIFIC / NO_BASELINE / "
          f"INSUFFICIENT (baseline {de1['baseline_restore']}/{de2['baseline_restore']}; "
          f"block_topk {de1['block_topk']}/{de2['block_topk']})")

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
