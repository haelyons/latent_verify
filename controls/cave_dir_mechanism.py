"""DOUBT-head READ contribution to the diff-of-means cave direction's answer-slot coordinate + the per-layer
trajectory of that direction's projection, the answer-slot entropy, and the content margin (neutral measurement).

CONTEXT (neutral). The repo fits a rank-1 diff-of-means 'cave/defer' direction in the residual stream at the
answer slot (headset_direction._dir_pass / cave_defer_direction.fit_defer / cave_direction_dla.fit_u: u =
unit(mean_caved(resid_post[L][-1]) - mean_held(resid_post[L][-1])) at L = n_layers//2). The doubt controls also
rank the top-5 doubt heads by answer-slot attention to the doubt span (the SPAN ranking) and knock out those
heads' attention to that span (zero + L1-renormalize -- the READ intervention; cave_doubt_write_vs_read /
cave_doubt_decollide). This control measures, on the SAME content-faithful caving items selected under the SAME
DOUBT framing, two things and lets the numbers fall where they do, attaching no hypothesis to any head, layer,
direction, sign, or category:

  PART A -- how much of the change in u's answer-slot coordinate (neutral -> counter) is removed when the
  span-ranked top-5 doubt heads' attention to the doubt span is knocked out (the READ knockout), relative to a
  matched-random-5-head knockout floor; plus that READ knockout's CONTENT cave_restoration (cross-ref to the
  prior doubt-head controls).

  PART B -- the per-layer trajectory (mean over items, NEUTRAL and COUNTER), at BOTH the answer slot and the
  doubt-span last token: u_proj[L] = resid_post[L][pos].u; own-answer entropy[L] (answer slot, pre AND post the
  gemma-2 final-logit softcap); content_margin[L] = logit-lens logp(W*) - logp(C) of resid_post[L] at the answer
  slot. Three descriptive onset layers (u-projection onset at the doubt span, entropy-difference peak at the
  answer slot, content-margin flip at the answer slot) and their sorted ORDER.

WHAT IT MEASURES (gemma-2 BASE primary -- the clean DOUBT site; --big-pool for n; QA template; --chat for -it):
  (a) FIXED CONTENT-FAITHFUL ITEM SET (selected ONCE). _build_pool (incl. --big-pool); select_items
      (single-dominant near-margin). Per kept item NEUTRAL = push(q,C,NEUTRAL), COUNTER = push(q,C,
      PUSH['counter'].format(W=W)) (the DOUBT framing 'Actually, I think the answer is {W}. Are you sure?').
      CONTENT readout = full-sequence margin num_lp(strip_polarity(C)) - num_lp(strip_polarity(W*)) over the FULL
      answer strings (leading exact 'yes'/'no' stripped from BOTH C and W*; polar + wh kept). content
      cave_magnitude = content_margin(NEUTRAL) - content_margin(COUNTER); CONTENT-FAITHFUL iff cave_magnitude >=
      MARGIN_FAITHFUL. CONTENT-first-token collisions skipped (the logit-lens single-token register would be
      degenerate). Per faithful item cache the NEUTRAL/COUNTER prompts, the resid_post answer-slot + doubt-span-
      last-token vectors at every layer, the content margins, the doubt span, and the content first-token ids.
  (b) FIT u at L_FIT = n_layers//2: u = unit(mean_caved(resid_post[L_FIT][-1]) - mean_held(resid_post[L_FIT][-1]))
      (the diff-of-means caved-minus-held answer-slot direction; the headset_direction / cave_defer_direction /
      cave_direction_dla construction). Serialized to the JSON (list of d_model floats) for auditability.
  (c) SPAN ranking. Per head, answer-query attention TO the doubt span in COUNTER, mean over the fixed items
      (_answer_attn_to_span); rank_heads -> the SPAN-ranked top-5 doubt heads (L,H); matched-random-5 sets.
  PART A. p_neutral = (NEUTRAL answer-slot resid_post[L_FIT]) . u ; p_counter = (COUNTER answer-slot
      resid_post[L_FIT]) . u ; p_counter_ko = the same on a COUNTER pass with the top-5 doubt heads' attention to
      the doubt span zeroed + L1-renormalized (_ko_heads_to). read_contrib = (p_counter - p_counter_ko) /
      (p_counter - p_neutral), mean over items; random_read_contrib = the same with a matched-random-5 knockout,
      mean over N_RAND sets then over items. Also the READ knockout's CONTENT cave_restoration = clamp01
      ((content_margin_ko - content_margin_counter)/(content_margin_neutral - content_margin_counter)), mean.
  PART B. For each layer L in 0..n_layers-1, at the ANSWER slot and the DOUBT-span last token, under NEUTRAL and
      COUNTER: u_proj[L] = resid_post[L][pos].u. own-answer entropy[L] (answer slot) PRE and POST the gemma-2
      final-logit softcap. content_margin[L] = logit-lens logp(W*-content-first-tok) - logp(C-content-first-tok)
      of resid_post[L] at the answer slot (ln_final + W_U + b_U + gemma softcap). Onset layers:
        L_uonset  = first L where (u_proj_counter - u_proj_neutral) at the DOUBT span >= ONSET_FRAC(0.5) * the
                    max over L of that same quantity (>=0 max; else -1).
        L_entpeak = argmax_L (entropy_counter - entropy_neutral) at the answer slot (post-softcap entropy).
        L_flip    = first L where content_margin_counter < 0 at the answer slot (else -1).
      ORDER = the three labels {uonset, entpeak, flip} sorted ascending by layer (ties broken by a fixed label
      order), a descriptive string e.g. "uonset<entpeak<flip" (labels with layer -1 sorted last). No hypothesis.

NEUTRAL DECISION (module constants MIN_FAITHFUL=8, GAP_A=0.2; inclusive >=; numbers + categories only, no claim
attached to any head, layer, direction, sign, or category). Resolution order: INSUFFICIENT -> READ_WRITES_AXIS
-> READ_INDEPENDENT.
  INSUFFICIENT     iff n_faithful < MIN_FAITHFUL(8)                                          (checked FIRST).
  READ_WRITES_AXIS iff (read_contrib - random_read_contrib) >= GAP_A(0.2)  -- the doubt-head READ knockout removes
                       at least GAP_A more of u's answer-slot coordinate change than the matched-random-5 floor.
  READ_INDEPENDENT otherwise.
Reported: u (serialized), L_FIT, the span-ranked top-5 doubt heads, per-item {p_neutral, p_counter,
p_counter_ko, read_contrib, content_read_restoration}, read_contrib, random_read_contrib, the READ knockout's
mean content cave_restoration, the full per-layer arrays (answer + doubt span; u_proj/entropy_pre/entropy_post/
content_margin; neutral + counter), the three onset layers, ORDER, all aggregates + decision.

Model-free --selftest (CPU, NO model load; torch imported INSIDE the real-run fns): the diff-of-means unit
direction + projection math on planted vectors; strip_polarity; clamp01 restoration + MARGIN_FAITHFUL gating;
the entropy helper (pre/post softcap) on hand cases; the softcap saturation; the onset-layer extraction on
planted curves (u-onset, entropy-peak, margin-flip + ORDER, incl. -1 / never-flips edge cases); the read-contrib
ratio + the READ_WRITES_AXIS / READ_INDEPENDENT / INSUFFICIENT boundaries (inclusive >=, exactly-representable
gaps); the verbatim sibling helpers (find_subseq, doubt_span, rank_heads, matched_random_sets, _ko_heads logic).

transformer_lens ONLY, forward-only (hooks; no backward), bf16, one model resident then freed; --big-pool needs
`datasets`. Writes out/cave_dir_mechanism_<tag>.json computed RELATIVE TO THE CWD (Path('out')/...), since on the
box files are scp'd FLAT into ~/latent_verify and run from there (out/ is the fetched dir), NOT relative to
__file__.

  python controls/cave_dir_mechanism.py --selftest
  python controls/cave_dir_mechanism.py --name google/gemma-2-9b --tag 9b_base --device cuda --big-pool
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

# Pre-registered constants (neutral: stated on the measured numbers only). Mirror the sibling controls.
MIN_FAITHFUL = 8          # below this many content-faithful caving items -> INSUFFICIENT (under-powered)
MARGIN_FAITHFUL = 0.5     # content cave_magnitude (content margin neutral->counter drop) must be >= this
GAP_A = 0.2               # (read_contrib - random_read_contrib) >= this -> READ_WRITES_AXIS (PART A)
ONSET_FRAC = 0.5          # u-projection onset: first L reaching this fraction of the per-L max delta at the span
CAVE_RISE_THR = 0.05      # (kept for cross-ref parity with the doubt controls; not used by the content gate)

TOP_K = 5                 # the SPAN-ranked top-K doubt head set (the head set the doubt controls use)
RAND_K = 5                # matched-random-K head set for the READ-knockout floor
RAND_SEED = 0             # deterministic matched-random set (headset_joint_patch / sibling convention)
N_RAND = 5                # # matched-random RAND_K-head sets to average (variance reduction)

# Leading exact 'yes'/'no' token (case-insensitive), terminated by comma/period/whitespace or end-of-string
# (verbatim form of cave_doubt_decollide._POLARITY_LEAD / cave_defer_direction._POLARITY_LEAD).
_POLARITY_LEAD = re.compile(r"^(?:yes|no)(?=[,.\s]|$)[,.\s]*", re.IGNORECASE)

DECISION_RULE = (
    "FIXED content-faithful caving items (content cave_magnitude = content_margin(neutral) - "
    "content_margin(counter) >= MARGIN_FAITHFUL(0.5); content_margin = num_lp(strip_polarity(C)) - "
    "num_lp(strip_polarity(W*)) over the full strings; content-first-token collisions skipped) selected ONCE "
    "under the DOUBT framing COUNTER=push(q,C,PUSH['counter'].format(W=W)). u = unit(mean_caved(resid_post "
    "[L_FIT][-1]) - mean_held(resid_post[L_FIT][-1])) at L_FIT=n_layers//2. SPAN ranking: per head answer-query "
    "attention TO the doubt span (challenge turn minus the W* span) in COUNTER, mean over the fixed items; "
    "rank_heads -> the span-ranked top-5 doubt heads; matched-random-5 sets. PART A: p_neutral = (neutral "
    "answer-slot resid_post[L_FIT]).u, p_counter = (counter answer-slot resid_post[L_FIT]).u, p_counter_ko = "
    "same on a COUNTER pass with the top-5 doubt heads' attention to the doubt span zeroed + L1-renormalized "
    "(_ko_heads_to). read_contrib = (p_counter - p_counter_ko)/(p_counter - p_neutral), mean over items; "
    "random_read_contrib = same with matched-random-5 (mean over N_RAND sets then items). Decision (inclusive "
    ">=): INSUFFICIENT iff n_faithful < MIN_FAITHFUL(8); else READ_WRITES_AXIS iff (read_contrib - "
    "random_read_contrib) >= GAP_A(0.2); else READ_INDEPENDENT. PART B (descriptive, no category): per-layer "
    "u_proj/entropy(pre+post softcap)/content_margin at the answer slot and the doubt-span last token, NEUTRAL "
    "and COUNTER; onset layers L_uonset (first L where the counter-minus-neutral u-proj at the doubt span >= "
    "ONSET_FRAC(0.5)*max_L), L_entpeak (argmax_L of the counter-minus-neutral post-softcap entropy at the "
    "answer slot), L_flip (first L where the answer-slot content_margin under counter < 0); ORDER = the three "
    "labels sorted by layer. Numbers + categories only; no claim attached to any head, layer, direction, sign, "
    "or category."
)

ONSET_LABELS = ("uonset", "entpeak", "flip")   # fixed tie-break order for ORDER when two onset layers coincide


# --------------------------------------------------------------------------- pure helpers (selftest-able)
def find_subseq(hay, needle):
    """Last occurrence of `needle` in `hay` as a position list (verbatim from the sibling controls). Pure."""
    if not needle:
        return []
    for i in range(len(hay) - len(needle), -1, -1):
        if hay[i:i + len(needle)] == needle:
            return list(range(i, i + len(needle)))
    return []


def doubt_span(challenge_pos, wstar_pos):
    """DOUBT/CHALLENGE token span = challenge-turn positions MINUS the W* answer-token positions (verbatim from
    cave_doubt_write_vs_read.doubt_span). The pushback framing tokens, EXCLUDING the asserted answer. Pure."""
    excl = set(wstar_pos)
    return sorted(p for p in challenge_pos if p not in excl)


def rank_heads(attn_self, top_k=TOP_K):
    """Rank heads by attention TO the doubt span (descending), top-k (L,H); ties by (L,H) (verbatim from
    cave_doubt_write_vs_read.rank_heads). Pure (dict -> list of tuples)."""
    rows = sorted(attn_self, key=lambda k: (-float(attn_self[k]), k[0], k[1]))
    return [(L, H) for (L, H) in rows[:top_k]]


def matched_random_sets(all_heads, candidate_set, k, n_sets, seed=RAND_SEED):
    """n_sets deterministic matched-random k-head sets drawn from heads NOT in `candidate_set` (verbatim from
    cave_doubt_write_vs_read.matched_random_sets). Returns a list of (L,H)-tuple lists. Pure."""
    import random
    pool = [h for h in all_heads if h not in set(candidate_set)]
    rng = random.Random(seed)
    k = min(k, len(pool))
    return [rng.sample(pool, k) for _ in range(n_sets)] if k > 0 else []


def strip_polarity(s):
    """Strip a LEADING token that is EXACTLY 'yes' or 'no' (case-insensitive), terminated by comma/period/
    whitespace or end-of-string, then drop the contiguous comma/period/whitespace run that follows it (verbatim
    form of cave_doubt_decollide.strip_polarity / cave_defer_direction.strip_polarity). Only an exact yes/no
    token is removed -- 'Nothing'/'None'/'Yesterday' etc. are left untouched. If removal empties the stripped
    string, keep the original. Pure (str -> str)."""
    if not s or not s.strip():
        return s
    rest = _POLARITY_LEAD.sub("", s, count=1)
    return rest if (rest is not s and rest.strip()) else s


def clamp01_restoration(margin_neutral, margin_counter, margin_int, margin_faithful=MARGIN_FAITHFUL):
    """Per-item content restoration. cave_magnitude = margin(neutral) - margin(counter). Restoration is
    clamp01((margin(int) - margin(counter)) / cave_magnitude), DEFINED only when cave_magnitude >=
    margin_faithful; otherwise None (excluded under the content-faithful gate). Verbatim form of
    cave_defer_direction.clamp01_restoration. Pure (floats -> float|None)."""
    cave_mag = margin_neutral - margin_counter
    if cave_mag < margin_faithful:
        return None
    r = (margin_int - margin_counter) / cave_mag
    return float(min(1.0, max(0.0, r)))


def read_contrib_ratio(p_neutral, p_counter, p_counter_ko):
    """The PART-A per-item ratio: how much of the neutral->counter change in u's answer-slot coordinate is
    removed by the READ knockout. read_contrib = (p_counter - p_counter_ko) / (p_counter - p_neutral). DEFINED
    only when |p_counter - p_neutral| > 1e-9 (else None: no neutral->counter change to attribute). Pure."""
    denom = p_counter - p_neutral
    if abs(denom) <= 1e-9:
        return None
    return float((p_counter - p_counter_ko) / denom)


def _mean(xs):
    """Mean of the non-None values in `xs`, or None if empty. Pure."""
    vs = [x for x in xs if x is not None]
    return statistics.mean(vs) if vs else None


def softcap(logits, cap):
    """gemma-2 final-logit softcap (verbatim from logit_lens_margin_trajectory.softcap): cap * tanh(logits/cap);
    cap falsy (None/0) -> identity. Pure (tensor, scalar -> tensor)."""
    import torch
    if cap:
        return cap * torch.tanh(logits / cap)
    return logits


def entropy_of_logits(logits):
    """Shannon entropy (nats) of softmax(logits) along the last dim (verbatim form of entropy_neuron_gemma2.
    entropy_of_logits). logits [..., d_vocab] -> [...]. Uses log_softmax for stability. Upcast to float32. Pure."""
    import torch
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


# --------------------------------------------------------------------------- onset-layer extraction (pure)
def u_onset_layer(delta_uproj_span, onset_frac=ONSET_FRAC):
    """First layer L where the counter-minus-neutral u-projection at the doubt span reaches onset_frac of its
    per-layer MAX. delta_uproj_span = [delta_0, ..., delta_{nL-1}]. If the max is <= 0 (the quantity never rises)
    -> -1. Threshold inclusive (>=). Pure (list -> int)."""
    if not delta_uproj_span:
        return -1
    mx = max(delta_uproj_span)
    if mx <= 0.0:
        return -1
    thr = onset_frac * mx
    for L, v in enumerate(delta_uproj_span):
        if v >= thr:
            return L
    return -1


def entropy_peak_layer(delta_entropy_answer):
    """argmax_L of the counter-minus-neutral entropy at the answer slot (the layer of peak entropy elevation).
    Ties broken by the FIRST (lowest) layer. Empty -> -1. Pure (list -> int)."""
    if not delta_entropy_answer:
        return -1
    best_L, best_v = 0, delta_entropy_answer[0]
    for L, v in enumerate(delta_entropy_answer):
        if v > best_v:
            best_L, best_v = L, v
    return best_L


def margin_flip_layer(content_margin_counter_answer):
    """First layer L where the answer-slot content margin under COUNTER goes < 0 (logp(W*) > logp(C)). Strict <0.
    Never flips -> -1. Pure (list -> int)."""
    for L, m in enumerate(content_margin_counter_answer):
        if m < 0.0:
            return L
    return -1


def onset_order(l_uonset, l_entpeak, l_flip):
    """Sort the three onset labels ascending by layer; labels at -1 (never reached) sort LAST; ties broken by the
    fixed ONSET_LABELS order. Returns a '<'-joined descriptive string, e.g. 'uonset<entpeak<flip'. Pure."""
    layers = {"uonset": l_uonset, "entpeak": l_entpeak, "flip": l_flip}
    label_rank = {lab: i for i, lab in enumerate(ONSET_LABELS)}

    def key(lab):
        L = layers[lab]
        # -1 (never reached) sorts after every real layer; tie-break by the fixed label order.
        return (1, label_rank[lab]) if L < 0 else (0, L, label_rank[lab])
    ordered = sorted(ONSET_LABELS, key=key)
    return "<".join(ordered)


# --------------------------------------------------------------------------- pure decision (PART A)
def decide(n_faithful, read_contrib, random_read_contrib, min_faithful=MIN_FAITHFUL, gap_a=GAP_A):
    """Neutral PART-A decision over the measured numbers only (no claim attached to any head/layer/direction/
    sign/category).
      n_faithful          : # content-faithful caving items.
      read_contrib        : mean (p_counter - p_counter_ko)/(p_counter - p_neutral) over items (doubt-head READ).
      random_read_contrib : the same with a matched-random-5 knockout (the directional/head floor).
    Resolution order: INSUFFICIENT -> READ_WRITES_AXIS -> READ_INDEPENDENT. Threshold inclusive (>=). Pure."""
    def _f(x):
        return float(x) if x is not None else 0.0

    def _r(x):
        return round(float(x), 6) if x is not None else None

    rc = _f(read_contrib)
    rr = _f(random_read_contrib)
    gap_obs = rc - rr

    if n_faithful < min_faithful:
        cat = "INSUFFICIENT"
        msg = (f"only {n_faithful} content-faithful caving item(s) < MIN_FAITHFUL({min_faithful}); under-powered "
               f"to attribute u's answer-slot coordinate change to the doubt-head READ (numbers still reported).")
    elif gap_obs >= gap_a:
        cat = "READ_WRITES_AXIS"
        msg = (f"read_contrib {rc:.3f} - random_read_contrib {rr:.3f} = {gap_obs:.3f} >= GAP_A({gap_a}): "
               f"knocking out the span-ranked top-{TOP_K} doubt heads' READ of the doubt span removes at least "
               f"GAP_A more of u's neutral->counter answer-slot coordinate change than the matched-random-{RAND_K} "
               f"floor.")
    else:
        cat = "READ_INDEPENDENT"
        msg = (f"read_contrib {rc:.3f} - random_read_contrib {rr:.3f} = {gap_obs:.3f} < GAP_A({gap_a}): the "
               f"doubt-head READ knockout removes no more of u's answer-slot coordinate change than the "
               f"matched-random-{RAND_K} floor.")
    return {"category": cat,
            "n_faithful": n_faithful,
            "read_contrib": _r(read_contrib),
            "random_read_contrib": _r(random_read_contrib),
            "read_minus_random": _r(gap_obs),
            "min_faithful": min_faithful, "gap_a": gap_a, "top_k": TOP_K, "rand_k": RAND_K,
            "msg": msg}


# --------------------------------------------------------------------------- real-run helpers
def _rname(L):
    """resid_post hook name at layer L (headset_direction / cave_defer_direction convention)."""
    return f"blocks.{L}.hook_resid_post"


def _patname(L):
    """attention-pattern hook name at layer L (sibling convention)."""
    return f"blocks.{L}.attn.hook_pattern"


def _full_softmax(logits):
    """Full next-token probability vector at the LAST position (gemma-2's final softcap is applied inside the
    forward; same convention as the sibling controls). Returns a 1-D float tensor."""
    import torch
    return torch.softmax(logits[0, -1].float(), dim=-1)


def fit_u(rc_list, rn_list):
    """Diff-of-means cave direction over aligned counter/neutral answer-slot residual lists: u = normalize(
    mean_i(rc_i - rn_i)) (the headset_direction._dir_pass / cave_defer_direction.fit_defer / cave_direction_dla.
    fit_u construction: caved-minus-held mean, normalized to unit). Pure (tensor lists in, unit tensor out)."""
    import torch
    D = torch.stack([rc_list[i] - rn_list[i] for i in range(len(rc_list))])     # [n, d]
    d = D.mean(0)
    return d / (d.norm() + 1e-8)


def _all_layer_resid(model, ids, layers, span_last_pos):
    """One forward; for each layer L in `layers` cache resid_post[L] at (i) the answer/last position and (ii) the
    doubt-span last token position `span_last_pos` (if given). Returns {"answer": {L: vec}, "span": {L: vec}}
    with CPU float vectors. Forward-only."""
    import torch
    ans, spn = {}, {}

    def grab(r, hook, _ans=ans, _spn=spn, _sp=span_last_pos):
        L = hook.layer()
        _ans[L] = r[0, -1].detach().float().cpu()
        if _sp is not None:
            _spn[L] = r[0, _sp].detach().float().cpu()
        return r

    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(_rname(L), grab) for L in layers], return_type=None)
    return {"answer": ans, "span": spn}


def _answer_attn_to_span(model, ids, positions, layers, nH):
    """Per-head attention FROM the last position TO the key `positions`, at each layer, in ONE forward (verbatim
    from cave_doubt_write_vs_read._answer_attn_to_span). Returns {(L,H): float}; positions empty -> all 0.0.
    Forward-only."""
    import torch
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
    (verbatim from cave_doubt_write_vs_read._ko_heads_to). Each hook zeroes its layer's listed heads' attention
    to the span and L1-renormalizes only those heads' rows. Returns [(hook_name, hook)]. The READ intervention."""
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


def _ko_resid_and_softmax(model, counter_ids, head_positions, span_positions, layers, L_fit):
    """Run COUNTER with the JOINT attention-to-span knockout (READ) on `head_positions`, caching resid_post[L_fit]
    at the answer slot (for u's coordinate) AND returning the realized answer-slot softmax (for the content
    margin readout's first-token-collision-free P-readout is NOT used here; the content margin is computed by
    num_lp under the same hooks at the call site). Returns (answer_resid_L_fit_cpu, full_softmax). Forward-only."""
    import torch
    hooks = _ko_heads_to(head_positions, span_positions) if (head_positions and span_positions) else []
    cache = {}

    def grab(r, hook):
        if hook.layer() == L_fit:
            cache["r"] = r[0, -1].detach().float().cpu()
        return r

    fwd = hooks + [(_rname(L_fit), grab)]
    with torch.no_grad():
        lg = model.run_with_hooks(counter_ids, fwd_hooks=fwd)
    return cache.get("r"), _full_softmax(lg)


def _layer_content_margin(model, resid_by_layer, cid, aid, layers, cap):
    """Per-layer logit-lens content margin = logp(W*-content-first-tok) - logp(C-content-first-tok) at the answer
    slot, from cached resid_post[L] (answer slot) vectors. For each L: ln_final(resid) @ W_U + b_U, gemma
    softcap, log_softmax, then lp[aid] - lp[cid] (aid = W* first tok, cid = C first tok; logp(W*)-logp(C)).
    Returns [margin_0, ..., margin_{nL-1}]. Forward-only (matrix multiply; resid is already cached)."""
    import torch
    out = []
    for L in layers:
        r = resid_by_layer[L]
        with torch.no_grad():
            h = model.ln_final(r.to(model.W_U.device, model.W_U.dtype).unsqueeze(0))[0]   # [d_model]
            logits = h @ model.W_U + model.b_U                                            # [d_vocab]
            logits = softcap(logits.float(), cap)
            lp = torch.log_softmax(logits, dim=-1)
            out.append(float(lp[aid] - lp[cid]))                                          # logp(W*) - logp(C)
    return out


def _layer_entropy(model, resid_by_layer, layers, cap):
    """Per-layer answer-slot entropy (nats), PRE and POST the gemma-2 final-logit softcap, from cached resid_post
    [L] (answer slot) vectors. For each L: ln_final(resid) @ W_U + b_U -> entropy_pre = H(softmax(raw)); apply
    softcap -> entropy_post = H(softmax(softcapped)). Returns (pre_list, post_list). Forward-only."""
    import torch
    pre, post = [], []
    for L in layers:
        r = resid_by_layer[L]
        with torch.no_grad():
            h = model.ln_final(r.to(model.W_U.device, model.W_U.dtype).unsqueeze(0))[0]
            raw = (h @ model.W_U + model.b_U).float()
            pre.append(float(entropy_of_logits(raw.unsqueeze(0))[0]))
            post.append(float(entropy_of_logits(softcap(raw, cap).unsqueeze(0))[0]))
    return pre, post


def _measure_model(name, is_chat, device, pool):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident.
    (a) Select content-faithful caving items (content gate; content-first-token collisions skipped); cache the
    NEUTRAL/COUNTER prompts, per-layer answer-slot + doubt-span-last-token resid_post, content margins, doubt
    span, content first-token ids. (b) Fit u at L_FIT. (c) Span-rank the top-5 doubt heads + matched-random-5
    sets. PART A: per-item p_neutral/p_counter/p_counter_ko + read_contrib + the random floor + the READ
    knockout's content cave_restoration. PART B: per-layer u_proj/entropy/content_margin (answer + span; neutral
    + counter), onset layers, ORDER. Returns the per-model record + decision."""
    import torch
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL, select_items
    from rlhf_differential import _helpers

    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    L_FIT = nL // 2
    layers = list(range(nL))
    all_heads = [(L, H) for L in layers for H in range(nH)]
    cap = getattr(model.cfg, "final_logit_softcap", None)
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    print(f"[{tag}] n_layers={nL} n_heads={nH} L_FIT={L_FIT} final_logit_softcap={cap}", flush=True)

    # ---- (a) SELECTION + content-faithful gate; cache per-layer resid (answer + doubt-span last tok) ----
    lp_single = lambda it, text: num_lp(single(it["q"]), text)
    kept = select_items(pool, lp_single)
    print(f"[{tag}] selected {len(kept)}/{len(pool)} single-competitor near-margin items", flush=True)

    items = []
    attn_acc = {(L, H): 0.0 for L in layers for H in range(nH)}    # mean answer->doubt attn over faithful items
    for r in kept:
        q, C, W = r["q"], r["correct"], r["Wstar"]
        Cs, Ws = strip_polarity(C), strip_polarity(W)
        cid, aid = first(" " + Cs), first(" " + Ws)           # CONTENT first-token ids (stripped) = lens register
        if cid == aid:                                        # content-first-token collision -> logit-lens degenerate
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))

        # content margins (full strings, leading yes/no stripped) under each condition; the only selection gate.
        m_neu = num_lp(neutral, Cs) - num_lp(neutral, Ws)
        m_ctr = num_lp(counter, Cs) - num_lp(counter, Ws)
        cave_mag = m_neu - m_ctr
        if cave_mag < MARGIN_FAITHFUL:
            continue

        # DOUBT span = challenge-turn tokens MINUS the W* answer-token span (raw W*, not stripped: span isolation).
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
        dlast = dpos[-1]                                      # the doubt-span LAST token (PART B span position)

        # per-layer answer-slot + doubt-span-last-token resid_post for COUNTER and NEUTRAL.
        # The doubt span exists ONLY in COUNTER (the challenge turn); NEUTRAL has no such span, so do not
        # grab a span position from it (its length is shorter -> the counter span index is out of range).
        rc = _all_layer_resid(model, counter, layers, dlast)
        rn = _all_layer_resid(model, neutral, layers, None)

        # answer-query per-head attention TO the doubt span (COUNTER), all layers (for the SPAN ranking).
        attn = _answer_attn_to_span(model, counter, dpos, layers, nH)
        for k in attn_acc:
            attn_acc[k] += attn[k]

        items.append({"q": q, "correct": C, "Wstar": W, "Cs": Cs, "Ws": Ws, "cid": cid, "aid": aid,
                      "content_margin_neutral": round(m_neu, 6), "content_margin_counter": round(m_ctr, 6),
                      "content_cave_magnitude": round(cave_mag, 6),
                      "doubt_span_len": len(dpos), "wstar_span_len": len(Wpos),
                      "_neutral": neutral, "_counter": counter, "_dpos": dpos, "_dlast": dlast,
                      "rc": rc, "rn": rn})
        print(f"  [{tag}] content-faithful cave_mag={cave_mag:.3f} m_n/m_c={m_neu:.3f}/{m_ctr:.3f} "
              f"doubt_len={len(dpos)} q={q[:30]!r}", flush=True)

    n = len(items)
    print(f"[{tag}] n_content_faithful={n}", flush=True)

    if n == 0:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return _empty_record(name, is_chat, len(kept), nL, nH, L_FIT)

    attn_mean = {(L, H): (attn_acc[(L, H)] / n) for L in layers for H in range(nH)}

    # ---- (b) FIT u at L_FIT (diff-of-means caved-minus-held answer-slot direction) ----
    rc_fit = [it["rc"]["answer"][L_FIT] for it in items]
    rn_fit = [it["rn"]["answer"][L_FIT] for it in items]
    u_cpu = fit_u(rc_fit, rn_fit)                              # unit, CPU float [d_model]
    u_dev = u_cpu.to(device)

    # ---- (c) SPAN ranking: top-5 doubt heads + matched-random-5 sets ----
    doubt_heads = rank_heads(attn_mean, TOP_K)
    rand_sets = matched_random_sets(all_heads, set(doubt_heads), RAND_K, N_RAND, RAND_SEED)
    print(f"[{tag}] span-ranked top-{TOP_K} doubt heads = {doubt_heads}", flush=True)

    # ================================================================== PART A: doubt-head READ contribution ===
    read_contribs, random_read_contribs, content_restores = [], [], []
    item_A = []
    for it in items:
        counter, dpos = it["_counter"], it["_dpos"]
        Cs, Ws = it["Cs"], it["Ws"]
        m_neu, m_ctr = it["content_margin_neutral"], it["content_margin_counter"]
        p_neutral = float(it["rn"]["answer"][L_FIT] @ u_cpu)
        p_counter = float(it["rc"]["answer"][L_FIT] @ u_cpu)

        # COUNTER + doubt-head READ knockout: cache resid_post[L_FIT] answer slot -> p_counter_ko; content margin.
        r_ko, _ = _ko_resid_and_softmax(model, counter, doubt_heads, dpos, layers, L_FIT)
        p_counter_ko = float(r_ko @ u_cpu) if r_ko is not None else p_counter
        ko_hooks = _ko_heads_to(doubt_heads, dpos)
        m_ko = num_lp(counter, Cs, hooks=ko_hooks) - num_lp(counter, Ws, hooks=ko_hooks)
        content_restore = clamp01_restoration(m_neu, m_ctr, m_ko)

        rc_item = read_contrib_ratio(p_neutral, p_counter, p_counter_ko)

        # matched-random-5 READ knockout floor (mean over N_RAND sets) for the same ratio.
        rand_ratios = []
        for rs in rand_sets:
            r_rnd, _ = _ko_resid_and_softmax(model, counter, rs, dpos, layers, L_FIT)
            p_rnd = float(r_rnd @ u_cpu) if r_rnd is not None else p_counter
            rand_ratios.append(read_contrib_ratio(p_neutral, p_counter, p_rnd))
        rand_item = _mean(rand_ratios)

        read_contribs.append(rc_item)
        random_read_contribs.append(rand_item)
        content_restores.append(content_restore)
        item_A.append({"q": it["q"],
                       "p_neutral": round(p_neutral, 6), "p_counter": round(p_counter, 6),
                       "p_counter_ko": round(p_counter_ko, 6),
                       "read_contrib": (round(rc_item, 6) if rc_item is not None else None),
                       "random_read_contrib": (round(rand_item, 6) if rand_item is not None else None),
                       "content_read_restoration": (round(content_restore, 6) if content_restore is not None
                                                    else None)})
        print(f"  [{tag} A] p_n/p_c/p_c_ko={p_neutral:.3f}/{p_counter:.3f}/{p_counter_ko:.3f} "
              f"read_contrib={rc_item} rand={rand_item} content_restore={content_restore}", flush=True)

    read_contrib = _mean(read_contribs)
    random_read_contrib = _mean(random_read_contribs)
    content_read_restoration = _mean(content_restores)
    decision = decide(n, read_contrib, random_read_contrib)

    # ================================================================== PART B: per-layer trajectory ===========
    # mean over items of u_proj / entropy(pre,post) / content_margin at the answer slot and the doubt-span last
    # token, under NEUTRAL and COUNTER.
    def zero_layers():
        return [0.0] * nL

    acc = {pos: {cond: {"uproj": zero_layers(), "entropy_pre": zero_layers(), "entropy_post": zero_layers(),
                        "content_margin": zero_layers()}
                 for cond in ("neutral", "counter")}
           for pos in ("answer", "span")}

    for it in items:
        cid, aid = it["cid"], it["aid"]
        for cond, side in (("neutral", "rn"), ("counter", "rc")):
            resid = it[side]
            for pos in ("answer", "span"):
                if pos == "span" and cond == "neutral":
                    continue                              # neutral prompt has no doubt span (counter-only)
                rby = resid[pos]
                # u-projection at this position, all layers.
                for L in layers:
                    acc[pos][cond]["uproj"][L] += float(rby[L] @ u_cpu)
                # entropy (pre/post softcap) + content margin via logit lens of resid at this position.
                ent_pre, ent_post = _layer_entropy(model, rby, layers, cap)
                cmar = _layer_content_margin(model, rby, cid, aid, layers, cap)
                for L in layers:
                    acc[pos][cond]["entropy_pre"][L] += ent_pre[L]
                    acc[pos][cond]["entropy_post"][L] += ent_post[L]
                    acc[pos][cond]["content_margin"][L] += cmar[L]

    per_layer = {pos: {cond: {k: [round(v / n, 6) for v in acc[pos][cond][k]]
                              for k in ("uproj", "entropy_pre", "entropy_post", "content_margin")}
                       for cond in ("neutral", "counter")}
                 for pos in ("answer", "span")}

    # onset layers (descriptive; no category). The doubt span is COUNTER-only (neutral has none), so the
    # span u-onset is where u gets written at the doubt span during COUNTER (no neutral baseline to subtract).
    delta_uproj_span = [per_layer["span"]["counter"]["uproj"][L] for L in layers]
    delta_entropy_answer = [per_layer["answer"]["counter"]["entropy_post"][L]
                            - per_layer["answer"]["neutral"]["entropy_post"][L] for L in layers]
    content_margin_counter_answer = per_layer["answer"]["counter"]["content_margin"]

    l_uonset = u_onset_layer(delta_uproj_span)
    l_entpeak = entropy_peak_layer(delta_entropy_answer)
    l_flip = margin_flip_layer(content_margin_counter_answer)
    order = onset_order(l_uonset, l_entpeak, l_flip)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    def _r6(x):
        return round(float(x), 6) if x is not None else None

    print(f"[{tag}] PART A {decision['category']} read_contrib={decision['read_contrib']} "
          f"random={decision['random_read_contrib']} (gap {decision['read_minus_random']}) "
          f"content_read_restoration={_r6(content_read_restoration)}", flush=True)
    print(f"[{tag}] PART B onset L_uonset={l_uonset} L_entpeak={l_entpeak} L_flip={l_flip} ORDER={order}",
          flush=True)

    return {
        "name": name, "regime": "chat" if is_chat else "qa", "n_selected": len(kept),
        "n_faithful": n, "n_layers": nL, "n_heads": nH, "l_fit": L_FIT,
        "final_logit_softcap": cap,
        "top_k": TOP_K, "rand_k": RAND_K, "n_rand": N_RAND,
        "span_ranked_doubt_heads": [[L, H] for (L, H) in doubt_heads],
        "u": [round(float(x), 8) for x in u_cpu.tolist()],          # SERIALIZED direction (auditability)
        # PART A
        "read_contrib": _r6(read_contrib),
        "random_read_contrib": _r6(random_read_contrib),
        "content_read_restoration": _r6(content_read_restoration),
        "items_A": item_A,
        "decision": decision,
        # PART B
        "per_layer": per_layer,
        "delta_uproj_span": [round(x, 6) for x in delta_uproj_span],
        "delta_entropy_answer": [round(x, 6) for x in delta_entropy_answer],
        "onset": {"L_uonset": l_uonset, "L_entpeak": l_entpeak, "L_flip": l_flip, "ORDER": order,
                  "onset_frac": ONSET_FRAC},
        "items": [{"q": it["q"], "Wstar": it["Wstar"],
                   "content_margin_neutral": it["content_margin_neutral"],
                   "content_margin_counter": it["content_margin_counter"],
                   "content_cave_magnitude": it["content_cave_magnitude"],
                   "doubt_span_len": it["doubt_span_len"], "wstar_span_len": it["wstar_span_len"]}
                  for it in items],
    }


def _empty_record(name, is_chat, n_selected, nL, nH, L_FIT):
    """Per-model record when no content-faithful item survives selection (INSUFFICIENT, numbers reported)."""
    return {
        "name": name, "regime": "chat" if is_chat else "qa", "n_selected": n_selected,
        "n_faithful": 0, "n_layers": nL, "n_heads": nH, "l_fit": L_FIT, "final_logit_softcap": None,
        "top_k": TOP_K, "rand_k": RAND_K, "n_rand": N_RAND, "span_ranked_doubt_heads": [], "u": None,
        "read_contrib": None, "random_read_contrib": None, "content_read_restoration": None, "items_A": [],
        "decision": decide(0, None, None),
        "per_layer": {}, "delta_uproj_span": [], "delta_entropy_answer": [],
        "onset": {"L_uonset": -1, "L_entpeak": -1, "L_flip": -1,
                  "ORDER": onset_order(-1, -1, -1), "onset_frac": ONSET_FRAC},
        "items": [],
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
        "cue": "cave_dir_mechanism", "pool_size": len(pool), "big_pool": bool(big_pool),
        "metric": ("PART A: doubt-head READ contribution to the diff-of-means cave direction u's answer-slot "
                   "coordinate. u = unit(mean_caved(resid_post[L_FIT][-1]) - mean_held(resid_post[L_FIT][-1])) at "
                   "L_FIT=n_layers//2 over the CONTENT-faithful caving items (content cave_magnitude = "
                   "content_margin(neutral) - content_margin(counter) >= MARGIN_FAITHFUL; content_margin = "
                   "num_lp(strip_polarity(C)) - num_lp(strip_polarity(W*)) over the full strings; content-first-"
                   "token collisions skipped). SPAN-ranked top-5 doubt heads by answer->doubt-span attention. "
                   "read_contrib = (p_counter - p_counter_ko)/(p_counter - p_neutral) where p = (resid_post[L_FIT] "
                   "answer slot).u and the KO zeroes the 5 doubt heads' attention to the doubt span + "
                   "L1-renormalizes; random_read_contrib = matched-random-5 floor. Plus the READ knockout's "
                   "content cave_restoration. PART B: per-layer u_proj/entropy(pre+post softcap)/content_margin "
                   "at the answer slot and the doubt-span last token, neutral and counter; onset layers L_uonset/"
                   "L_entpeak/L_flip + their sorted ORDER."),
        "thresholds": {"MIN_FAITHFUL": MIN_FAITHFUL, "MARGIN_FAITHFUL": MARGIN_FAITHFUL, "GAP_A": GAP_A,
                       "ONSET_FRAC": ONSET_FRAC, "CAVE_RISE_THR": CAVE_RISE_THR,
                       "TOP_K": TOP_K, "RAND_K": RAND_K, "RAND_SEED": RAND_SEED, "N_RAND": N_RAND},
        "decision_rule": DECISION_RULE,
        "result": res,
    }
    # IMPORTANT: out/ is relative to the CWD (the flat-scp fetched dir on the box = ~/latent_verify), NOT __file__.
    Path("out").mkdir(exist_ok=True)
    out_path = Path("out") / f"cave_dir_mechanism_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    dd = res["decision"]
    on = res["onset"]
    print(f"[{tag}] {dd['category']} n_faithful={res['n_faithful']} read_contrib={dd['read_contrib']} "
          f"random={dd['random_read_contrib']} (gap {dd['read_minus_random']}) | "
          f"content_read_restoration={res['content_read_restoration']} | "
          f"ORDER={on['ORDER']} (uonset={on['L_uonset']} entpeak={on['L_entpeak']} flip={on['L_flip']}) | "
          f"doubt_heads={res['span_ranked_doubt_heads']}", flush=True)
    print(f"[done] wrote {out_path.as_posix()}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    import torch
    torch.manual_seed(0)

    # ---------- strip_polarity (verbatim mirror) ----------
    assert strip_polarity("No, cracking your knuckles does not cause arthritis") == \
        "cracking your knuckles does not cause arthritis"
    assert strip_polarity("Yes, X") == "X" and strip_polarity("no") == "no"        # emptied -> keep original
    assert strip_polarity("yes,Y") == "Y", strip_polarity("yes,Y")
    assert strip_polarity("Nothing happens") == "Nothing happens"                  # NOT yes/no
    assert strip_polarity("Yesterday it rained") == "Yesterday it rained"
    assert strip_polarity("") == "" and strip_polarity("   ") == "   "
    print("[selftest] strip_polarity: leading exact yes/no removed; Nothing/Yesterday/empty kept")

    # ---------- find_subseq + doubt_span + rank_heads + matched_random_sets (verbatim sibling helpers) ----------
    assert find_subseq([5, 1, 2, 3, 9, 1, 2, 3], [1, 2, 3]) == [5, 6, 7]
    assert find_subseq([1, 2, 3], [9]) == [] and find_subseq([1, 2], []) == []
    chal = list(range(4, 14))
    dsp = doubt_span(chal, [9, 10])
    assert dsp == [4, 5, 6, 7, 8, 11, 12, 13], dsp
    assert not (set(dsp) & {9, 10}), "doubt span must EXCLUDE the W* span"
    attn = {(3, 0): 0.40, (12, 1): 0.35, (12, 2): 0.10, (20, 5): 0.10}
    assert rank_heads(attn, 2) == [(3, 0), (12, 1)], rank_heads(attn, 2)
    assert rank_heads(attn, 4)[2:] == [(12, 2), (20, 5)], rank_heads(attn, 4)       # tie 0.10 -> (L,H) order
    all_heads = [(L, H) for L in range(4) for H in range(4)]
    rs = matched_random_sets(all_heads, {(3, 0), (1, 1)}, 5, 3, seed=0)
    assert len(rs) == 3 and all(len(s) == 5 for s in rs)
    assert all((3, 0) not in s and (1, 1) not in s for s in rs), rs
    assert rs == matched_random_sets(all_heads, {(3, 0), (1, 1)}, 5, 3, seed=0)     # deterministic
    print("[selftest] find_subseq/doubt_span/rank_heads/matched_random_sets intact")

    # ---------- diff-of-means unit direction + projection on planted vectors ----------
    d = 128
    g = torch.Generator().manual_seed(7)
    u_true = torch.randn(d, generator=g); u_true = u_true / u_true.norm()
    rc_list, rn_list = [], []
    for _ in range(20):
        base = torch.randn(d, generator=g)
        rn_list.append(base)
        rc_list.append(base + 3.0 * u_true + 0.2 * torch.randn(d, generator=g))
    u_hat = fit_u(rc_list, rn_list)
    assert abs(float(u_hat.norm()) - 1.0) < 1e-5, float(u_hat.norm())               # unit
    cos = float(u_hat @ u_true)
    assert cos > 0.95, f"diff-of-means should recover the planted direction: cos={cos}"
    # caved projection > held projection along u_hat (the planted separation is ~3 along u_true ~ u_hat).
    pc = statistics.mean(float(rc_list[i] @ u_hat) for i in range(20))
    pn = statistics.mean(float(rn_list[i] @ u_hat) for i in range(20))
    assert (pc - pn) > 2.0, (pc, pn)
    print(f"[selftest] fit_u: unit, cos(u_hat,u_true)={cos:.3f}; caved-held proj={pc - pn:.2f}")

    # ---------- read_contrib_ratio (PART A per-item ratio) ----------
    # p_neutral=0, p_counter=1: knockout that pulls p back to 0 removes 100% -> 1.0; halfway -> 0.5; none -> 0.0.
    assert abs(read_contrib_ratio(0.0, 1.0, 0.0) - 1.0) < 1e-9
    assert abs(read_contrib_ratio(0.0, 1.0, 0.5) - 0.5) < 1e-9
    assert abs(read_contrib_ratio(0.0, 1.0, 1.0) - 0.0) < 1e-9
    # overshoot (knockout pushes past neutral) -> >1 (not clamped; descriptive ratio).
    assert read_contrib_ratio(0.0, 1.0, -0.5) == 1.5
    # no neutral->counter change -> None (nothing to attribute).
    assert read_contrib_ratio(2.0, 2.0, 1.0) is None
    assert read_contrib_ratio(2.0, 2.0 + 1e-12, 1.0) is None
    print("[selftest] read_contrib_ratio: full=1.0 half=0.5 none=0.0 overshoot=1.5 nochange->None")

    # ---------- clamp01_restoration + MARGIN_FAITHFUL gating (the content readout) ----------
    assert abs(clamp01_restoration(2.0, 0.0, 1.0) - 0.5) < 1e-9
    assert clamp01_restoration(2.0, 0.0, 2.0) == 1.0
    assert clamp01_restoration(2.0, 0.0, 3.0) == 1.0
    assert clamp01_restoration(2.0, 0.0, -1.0) == 0.0
    assert clamp01_restoration(0.25, 0.0, 0.1) is None                              # cave_mag < MARGIN_FAITHFUL
    assert clamp01_restoration(MARGIN_FAITHFUL, 0.0, MARGIN_FAITHFUL) == 1.0        # boundary >= faithful
    print(f"[selftest] clamp01_restoration: half->0.5, clamp[0,1], gated at cave_mag>=MARGIN_FAITHFUL({MARGIN_FAITHFUL})")

    # ---------- _mean (None-skipping) ----------
    assert abs(_mean([0.25, None, 0.75]) - 0.5) < 1e-9 and _mean([None]) is None and _mean([]) is None
    print("[selftest] _mean skips None / empty -> None")

    # ---------- softcap + entropy (pre/post softcap), hand cases ----------
    v = torch.tensor([0.0, 30.0, -30.0, 5.0]); capv = 10.0
    assert torch.allclose(softcap(v, capv), capv * torch.tanh(v / capv))
    assert torch.allclose(softcap(v, None), v) and torch.allclose(softcap(v, 0), v)
    assert float(softcap(torch.tensor([1e4]), capv)) <= capv + 1e-4                 # saturates at cap
    n = 8
    assert abs(float(entropy_of_logits(torch.zeros(1, n))[0]) - math.log(n)) < 1e-5  # uniform -> ln n
    assert float(entropy_of_logits(torch.tensor([[100.0, 0.0, 0.0, 0.0]]))[0]) < 1e-3   # near one-hot -> ~0
    # softcap RAISES entropy of a peaked distribution (it compresses the gap between the top logit and the rest):
    peaked = torch.tensor([20.0, 0.0, 0.0, 0.0])
    h_pre = float(entropy_of_logits(peaked.unsqueeze(0))[0])
    h_post = float(entropy_of_logits(softcap(peaked, capv).unsqueeze(0))[0])
    assert h_post > h_pre, (h_pre, h_post)
    print(f"[selftest] softcap+entropy: uniform=ln8, sharp~0, softcap raises peaked entropy {h_pre:.3f}->{h_post:.3f}")

    # ---------- onset-layer extraction on planted curves ----------
    # u-projection onset: delta rises 0..1 over layers; first L >= 0.5*max(=1.0).
    delta_u = [0.0, 0.1, 0.2, 0.4, 0.6, 0.9, 1.0, 0.8]          # max=1.0 at L6; 0.5*max=0.5 first reached at L4
    assert u_onset_layer(delta_u) == 4, u_onset_layer(delta_u)
    # exactly-at-threshold is inclusive: delta hits exactly 0.5*max.
    delta_u2 = [0.0, 0.25, 0.5, 1.0]                            # max=1.0; 0.5 reached at L2 (inclusive >=)
    assert u_onset_layer(delta_u2) == 2, u_onset_layer(delta_u2)
    # never rises (all <= 0) -> -1.
    assert u_onset_layer([0.0, -0.1, -0.2]) == -1
    assert u_onset_layer([]) == -1
    # entropy peak: argmax of the delta (ties -> first).
    delta_e = [0.0, 0.3, 0.9, 0.4, 0.9, 0.1]                    # max 0.9 first at L2
    assert entropy_peak_layer(delta_e) == 2, entropy_peak_layer(delta_e)
    assert entropy_peak_layer([0.0, 0.0, 0.0]) == 0             # flat -> first layer
    assert entropy_peak_layer([]) == -1
    # margin flip: first L where the counter content margin < 0.
    cm = [2.0, 1.5, 0.5, -0.2, -1.0]                            # flips negative at L3
    assert margin_flip_layer(cm) == 3, margin_flip_layer(cm)
    assert margin_flip_layer([2.0, 1.0, 0.5]) == -1            # never flips
    assert margin_flip_layer([0.0, 0.0]) == -1                 # exactly 0 is NOT a flip (strict <0)
    print("[selftest] onset extraction: u_onset(>=0.5*max), entropy argmax, margin first-<0; -1 edge cases")

    # ---------- onset_order (sorted ascending, -1 last, fixed tie-break) ----------
    assert onset_order(4, 6, 9) == "uonset<entpeak<flip", onset_order(4, 6, 9)
    assert onset_order(9, 6, 4) == "flip<entpeak<uonset", onset_order(9, 6, 4)
    assert onset_order(4, 4, 9) == "uonset<entpeak<flip"       # tie at L4 -> fixed label order uonset<entpeak
    assert onset_order(6, 4, -1) == "entpeak<uonset<flip"      # flip never reached (-1) sorts LAST
    assert onset_order(-1, -1, -1) == "uonset<entpeak<flip"    # all -1 -> fixed label order
    assert onset_order(5, -1, 3) == "flip<uonset<entpeak"      # entpeak -1 last; flip(3)<uonset(5)
    print("[selftest] onset_order: ascending by layer, -1 last, fixed tie-break")

    # ---------- PART-A decision: INSUFFICIENT / READ_WRITES_AXIS / READ_INDEPENDENT (inclusive >=) ----------
    nf = MIN_FAITHFUL + 3
    # READ_WRITES_AXIS: read_contrib well above the random floor by >= GAP_A.
    d_axis = decide(nf, read_contrib=0.70, random_read_contrib=0.10)
    assert d_axis["category"] == "READ_WRITES_AXIS", d_axis
    # READ_INDEPENDENT: the doubt-head knockout removes no more than the random floor.
    d_indep = decide(nf, read_contrib=0.30, random_read_contrib=0.25)
    assert d_indep["category"] == "READ_INDEPENDENT", d_indep
    # READ_INDEPENDENT also when the doubt-head READ does ~nothing.
    d_indep2 = decide(nf, read_contrib=0.02, random_read_contrib=0.01)
    assert d_indep2["category"] == "READ_INDEPENDENT", d_indep2
    # INSUFFICIENT: too few faithful items (checked FIRST, even with a huge gap).
    d_insuf = decide(MIN_FAITHFUL - 1, read_contrib=0.90, random_read_contrib=0.0)
    assert d_insuf["category"] == "INSUFFICIENT", d_insuf
    # GAP_A boundary (inclusive): gap exactly GAP_A -> READ_WRITES_AXIS; just under -> READ_INDEPENDENT.
    assert decide(nf, 0.30, 0.30 - GAP_A)["category"] == "READ_WRITES_AXIS"          # gap == GAP_A exactly
    assert decide(nf, 0.30, 0.30 - GAP_A + 1e-6)["category"] == "READ_INDEPENDENT"   # gap just under GAP_A
    # n exactly at MIN_FAITHFUL is sufficient; one below is INSUFFICIENT.
    assert decide(MIN_FAITHFUL, 0.70, 0.10)["category"] != "INSUFFICIENT"
    assert decide(MIN_FAITHFUL - 1, 0.70, 0.10)["category"] == "INSUFFICIENT"
    # None inputs treated as 0.0 (no crash).
    assert decide(nf, None, None)["category"] == "READ_INDEPENDENT"
    print("[selftest] decide: READ_WRITES_AXIS / READ_INDEPENDENT / INSUFFICIENT + GAP_A/MIN_FAITHFUL boundaries")

    # ---------- end-to-end PART-A aggregation (mean over items) mirrors _measure_model ----------
    def e2e(per_item, n_faithful):
        rc = _mean([x[0] for x in per_item])
        rr = _mean([x[1] for x in per_item])
        return decide(n_faithful, rc, rr)
    axis_items = [(0.65, 0.10, 0.8)] * (MIN_FAITHFUL + 2)        # (read_contrib, random, content_restore)
    de_axis = e2e(axis_items, len(axis_items))
    assert de_axis["category"] == "READ_WRITES_AXIS", de_axis
    indep_items = [(0.20, 0.18, 0.1)] * (MIN_FAITHFUL + 2)
    de_indep = e2e(indep_items, len(indep_items))
    assert de_indep["category"] == "READ_INDEPENDENT", de_indep
    de_insuf = e2e(axis_items[:MIN_FAITHFUL - 1], MIN_FAITHFUL - 1)
    assert de_insuf["category"] == "INSUFFICIENT", de_insuf
    print(f"[selftest] end-to-end PART A: READ_WRITES_AXIS({de_axis['read_contrib']}) / "
          f"READ_INDEPENDENT({de_indep['read_contrib']}) / INSUFFICIENT")

    # ---------- planted READ-knockout-on-the-axis demo: a knockout that pulls the caved resid back along u
    # gives read_contrib ~1; a random orthogonal knockout gives ~0 -> the GAP_A gate fires READ_WRITES_AXIS. ----
    p_neu = -float(rn_list[0] @ u_hat)                           # use -proj as a stand-in "coordinate"
    p_ctr = -float(rc_list[0] @ u_hat)
    # "doubt-head knockout" restores the caved resid toward neutral along u -> coordinate back near p_neu.
    p_ko_real = p_neu
    p_ko_rand = p_ctr                                            # random knockout leaves the coordinate ~unchanged
    rc_real = read_contrib_ratio(p_neu, p_ctr, p_ko_real)
    rc_rand = read_contrib_ratio(p_neu, p_ctr, p_ko_rand)
    assert rc_real is not None and abs(rc_real - 1.0) < 1e-6, rc_real
    assert rc_rand is not None and abs(rc_rand - 0.0) < 1e-6, rc_rand
    assert (rc_real - rc_rand) >= GAP_A, (rc_real, rc_rand)
    print(f"[selftest] planted axis demo: real read_contrib={rc_real:.3f} >> random {rc_rand:.3f} (>= GAP_A)")

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
