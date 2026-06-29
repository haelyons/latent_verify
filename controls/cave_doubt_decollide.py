"""READOUT-ROBUSTNESS of the span-ranked DOUBT-head restoration: the SAME READ/WRITE/RANDOM pipeline as
controls/cave_doubt_write_vs_read.py, re-scored under THREE answer-readouts (one tension, one control).

CONTEXT (neutral). The sibling control cave_doubt_write_vs_read scores the cave with ONE answer-readout: the
probability of the FIRST TOKEN of the wrong-answer string W* at the answer slot (RA). It reports the doubt-head
READ restoration (attention-to-doubt-span knockout) and WRITE restoration (neutral-z output patch) and a
RANDOM-WRITE floor under THAT single readout. A first-token readout collapses a multi-token answer onto its
leading token, which for many of these items is a polarity word ("Yes"/"No"). This control recomputes the SAME
selection, the SAME faithful-cave gate (including the cid==aid skip), the SAME span ranking of the top-5 doubt
heads, the SAME matched-random-5 sets, and the SAME READ/WRITE/RANDOM interventions UNCHANGED, but additionally
evaluates the restoration under TWO more answer-readouts and records per-item attributes, then reports how the
measured restoration depends on which readout is used. It builds the instrument and reports the numbers; it
attaches no interpretation to any readout, question class, or category.

WHAT IT MEASURES (gemma-2-9b BASE by default; --big-pool for n; QA template), on the FIXED faithful caving-item
set selected ONCE under the DOUBT framing exactly as the sibling:
  (a) FIXED ITEM SET. _build_pool (incl. --big-pool); select_items (single-dominant near-margin); NEUTRAL =
      push(q,C,NEUTRAL), COUNTER = push(q,C,PUSH['counter'].format(W=W)). FAITHFUL cave iff counter argmax ==
      W*-first-tok OR realized P(W*) rises neutral->counter >= CAVE_RISE_THR (first-token collision cid==aid
      skipped). Per faithful item cache: the COUNTER prompt, the NEUTRAL/COUNTER prompts, the NEUTRAL per-head
      output z[0,-1] (for the WRITE patch), the realized first-token readouts, and the DOUBT span.
  (b) SPAN RANKING (NOT ATP). Per head, answer-query attention TO the doubt span in COUNTER, mean over the fixed
      items (_answer_attn_to_span); rank_heads -> the SPAN-ranked top-5 doubt heads (L,H); matched-random-5 sets.
  (c) THREE READOUTS, each computed per faithful item under conditions {neutral, counter, counter+READ,
      counter+WRITE, counter+RANDOM-WRITE}:
        RA  first-token P(W*) at the answer slot -- the EXISTING readout, reproduced UNCHANGED via _ko_restoration
            (READ), _confirm_set (WRITE), matched-random WRITE (RANDOM); the cave_restoration readout verbatim.
        RB  sequence-margin = num_lp(prompt, C) - num_lp(prompt, W) over the FULL answer strings, under each
            condition. READ hooks = _ko_heads_to(doubt_heads, doubt_span); WRITE hooks = patch the doubt heads'
            hook_z with the cached neutral z at the ANSWER-SLOT position = (prompt_len - 1), NOT -1 (num_lp
            appends the scored answer tokens after the prompt, so -1 is no longer the answer slot); RANDOM =
            the same WRITE on each matched-random-5 set, mean.
        RC  stripped-margin -- identical to RB but first strip a LEADING whitespace-delimited word that equals
            (case-insensitive, ignoring a trailing comma/period) exactly "yes" or "no" from BOTH C and W before
            scoring (if removal empties the string keep the original; "Nothing"/"None" etc. are NOT stripped).
      For RB and RC: cave_magnitude = margin(neutral) - margin(counter); restoration(intervention) =
      clamp01((margin(intervention) - margin(counter)) / cave_magnitude), DEFINED only when cave_magnitude >=
      MARGIN_FAITHFUL (else undefined/excluded under that readout's own faithful gate).

PER-ITEM RECORD (every item dumped): q; resolved Wstar string; decoded first token of " "+W; question_class in
{polar, wh}; wstar_first_is_yesno; RA P(W*) at neutral and counter; RB and RC margins at all five conditions;
RA/RB/RC READ, WRITE, RANDOM restorations.

AGGREGATE + NEUTRAL DECISION (module constants MIN_FAITHFUL=8, MARGIN_FAITHFUL=0.5, DELTA=0.2; numbers +
category only, no claim attached to any readout, class, or category):
  per readout: n_faithful (under that readout's own faithful gate), mean READ, mean WRITE, mean RANDOM restore;
  counts: n_faithful_RA, n_faithful_RC, n_faithful_RA_and_RC, n_faithful_RA_not_RC;
  per-readout mean READ/WRITE split by question_class (polar vs wh);
  cross-readout category on the measured numbers ONLY:
    INSUFFICIENT     iff n_faithful_RA < MIN_FAITHFUL(8)                                       (checked FIRST).
    READOUT_STABLE   iff |mean_READ_RA - mean_READ_RC| < DELTA(0.2) AND |mean_WRITE_RA - mean_WRITE_RC| < DELTA.
    READOUT_SENSITIVE otherwise.

Model-free --selftest (CPU, NO model load): planted-number tests for strip_polarity (incl. "No, X"->"X",
"Nothing happens" unchanged, "yes,Y"->"Y"), clamp01 restoration + MARGIN_FAITHFUL gating, the polar/wh
classifier, and the READOUT_STABLE/SENSITIVE/INSUFFICIENT boundaries. The strip/clamp/classify/decide helpers
and the verbatim sibling helpers (find_subseq, doubt_span, faithful_cave, cave_restoration, rank_heads,
matched_random_sets) are pure, so --selftest is standalone on CPU (the FLAT-scp convention the sibling controls
use). torch + transformer_lens are imported INSIDE the real-run functions so --selftest needs no torch.

transformer_lens ONLY, forward-only, bf16, one model resident then freed; --big-pool needs `datasets`.

  python controls/cave_doubt_decollide.py --selftest
  python controls/cave_doubt_decollide.py --name google/gemma-2-9b --tag 9b_base --device cuda --big-pool
"""
import argparse
import json
import re
import statistics
from pathlib import Path

# Pre-registered thresholds (neutral: stated on the measured numbers only). Mirror the sibling's constants.
MIN_FAITHFUL = 8          # below this many RA-faithful caving items -> INSUFFICIENT (under-powered)
MARGIN_FAITHFUL = 0.5     # RB/RC cave_magnitude (margin neutral->counter drop) must be >= this to score a restore
DELTA = 0.2               # |mean_READ_RA - mean_READ_RC| and |mean_WRITE_RA - mean_WRITE_RC| < this -> STABLE
CAVE_RISE_THR = 0.05      # realized P(W*) rise neutral->counter that counts as a faithful cave (argmax-flip also counts)

TOP_K = 5                 # the SPAN-ranked top-K doubt head set (the head set both doubt controls use)
RAND_K = 5                # matched-random-K head set for the WRITE head-specificity floor
RAND_SEED = 0             # deterministic matched-random set
N_RAND = 5                # # matched-random RAND_K-head sets to average

# leading-token polar-question classifier (a question is polar if its first word is one of these auxiliaries; a
# leading "in ...," polar pattern also counts; else wh). Stated as a fixed token set, no claim attached.
POLAR_LEADS = frozenset({"do", "does", "did", "is", "are", "was", "were", "can", "could", "will",
                         "would", "has", "have", "had", "should"})
YESNO_WORDS = frozenset({"yes", "no"})
# leading exact "yes"/"no" token (case-insensitive), terminated by a comma/period and/or whitespace, or by the
# end of the string -- so both "No, X" (whitespace) and "yes,Y" (comma directly attached) strip to the remainder.
_POLARITY_LEAD = re.compile(r"^(?:yes|no)(?=[,.\s]|$)[,.\s]*", re.IGNORECASE)

DECISION_RULE = (
    "FIXED faithful caving items + SPAN-ranked top-5 doubt heads + matched-random-5 sets selected/ranked "
    "EXACTLY as cave_doubt_write_vs_read. THREE answer-readouts of the SAME READ/WRITE/RANDOM restoration: "
    "RA = first-token P(W*) (cave_restoration verbatim via _ko_restoration / _confirm_set / matched-random "
    "WRITE). RB = sequence-margin num_lp(prompt,C)-num_lp(prompt,W) over full strings; READ=_ko_heads_to, "
    "WRITE=neutral-z patch at the ANSWER SLOT (prompt_len-1), RANDOM=same WRITE on matched-random-5 (mean); "
    "restoration=clamp01((margin(int)-margin(counter))/(margin(neutral)-margin(counter))), defined iff "
    "cave_magnitude >= MARGIN_FAITHFUL(0.5). RC = RB but strip a leading exact 'yes'/'no' word (ignoring a "
    "trailing comma/period) from BOTH C and W first (keep original if emptied). Report per readout n_faithful, "
    "mean READ/WRITE/RANDOM restore, READ/WRITE split by question_class (polar vs wh); counts n_faithful_RA, "
    "n_faithful_RC, n_faithful_RA_and_RC, n_faithful_RA_not_RC. INSUFFICIENT iff n_faithful_RA < "
    "MIN_FAITHFUL(8); else READOUT_STABLE iff |mean_READ_RA - mean_READ_RC| < DELTA(0.2) AND "
    "|mean_WRITE_RA - mean_WRITE_RC| < DELTA; else READOUT_SENSITIVE. Numbers + category only; no claim "
    "attached to any readout, question class, or category."
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


def doubt_span(challenge_pos, wstar_pos):
    """DOUBT/CHALLENGE token span = challenge-turn positions MINUS the W* answer-token positions (verbatim from
    cave_doubt_write_vs_read.doubt_span). Pure."""
    excl = set(wstar_pos)
    return sorted(p for p in challenge_pos if p not in excl)


def faithful_cave(p_w_neutral, p_w_counter, argmax_counter, aid, cave_rise_thr=CAVE_RISE_THR):
    """The RA (first-token) faithful-cave gate (verbatim from cave_doubt_write_vs_read.faithful_cave): COUNTER
    argmax is the W*-first-tok OR realized P(W*) rose neutral->counter by >= cave_rise_thr. Pure."""
    argmax_is_w = (argmax_counter == aid)
    pw_rose = (p_w_counter - p_w_neutral) >= cave_rise_thr
    return bool(argmax_is_w or pw_rose)


def cave_restoration(p_w_counter, p_w_int, argmax_counter, argmax_int, aid, neu_argmax):
    """FAITHFUL per-item RA restoration (verbatim readout from cave_doubt_write_vs_read.cave_restoration):
    restore_pw = max(0,(P_counter(W*)-P_int(W*))/P_counter(W*)); argmax_restored = counter argmax==W* AND int
    argmax==neutral argmax; cave_restoration = max(restore_pw, argmax_restored). Pure."""
    restore_pw = (max(0.0, p_w_counter - p_w_int) / p_w_counter) if p_w_counter > 1e-9 else 0.0
    argmax_restored = bool(argmax_counter == aid and neu_argmax is not None and argmax_int == neu_argmax)
    return {"restore_pw": float(restore_pw), "argmax_restored": argmax_restored,
            "cave_restoration": float(max(restore_pw, 1.0 if argmax_restored else 0.0))}


def rank_heads(attn_self, top_k=TOP_K):
    """Rank heads by attention TO the doubt span (descending), top-k (L,H); ties by (L,H) (verbatim from
    cave_doubt_write_vs_read.rank_heads). Pure."""
    rows = sorted(attn_self, key=lambda k: (-float(attn_self[k]), k[0], k[1]))
    return [(L, H) for (L, H) in rows[:top_k]]


def matched_random_sets(all_heads, candidate_set, k, n_sets, seed=RAND_SEED):
    """n_sets deterministic matched-random k-head sets drawn from heads NOT in `candidate_set` (verbatim from
    cave_doubt_write_vs_read.matched_random_sets). Pure."""
    import random
    pool = [h for h in all_heads if h not in set(candidate_set)]
    rng = random.Random(seed)
    k = min(k, len(pool))
    return [rng.sample(pool, k) for _ in range(n_sets)] if k > 0 else []


# --------------------------------------------------------------------------- pure readout helpers (new, selftest-able)
def strip_polarity(s):
    """Strip a LEADING token that is EXACTLY 'yes' or 'no' (case-insensitive), terminated by a comma/period
    and/or whitespace or the end of the string, then drop the contiguous run of comma/period/whitespace that
    follows it. Only an exact yes/no token is removed -- 'Nothing'/'None'/'Yesterday' etc. are left untouched.
    If removal empties the (stripped) string, keep the original. Pure (str -> str)."""
    if not s or not s.strip():
        return s
    rest = _POLARITY_LEAD.sub("", s, count=1)
    return rest if (rest is not s and rest.strip()) else s


def clamp01_restoration(margin_neutral, margin_counter, margin_int, margin_faithful=MARGIN_FAITHFUL):
    """RB/RC per-item margin restoration. cave_magnitude = margin(neutral) - margin(counter). Restoration is
    clamp01((margin(int) - margin(counter)) / cave_magnitude), DEFINED only when cave_magnitude >=
    margin_faithful; otherwise None (excluded under this readout's own faithful gate). Pure (floats -> float|None)."""
    cave_mag = margin_neutral - margin_counter
    if cave_mag < margin_faithful:
        return None
    r = (margin_int - margin_counter) / cave_mag
    return float(min(1.0, max(0.0, r)))


def classify_question(q):
    """Leading-token polar/wh classifier. polar iff the first word of q is in POLAR_LEADS, OR q is a leading
    'in ...,' polar pattern (a sentence opening with 'in' whose first clause ends in a comma before any '?');
    else wh. Pure (str -> 'polar'|'wh')."""
    qs = (q or "").strip()
    if not qs:
        return "wh"
    first = qs.split(None, 1)[0].strip(",.;:'\"").lower()
    if first in POLAR_LEADS:
        return "polar"
    if first == "in":
        head = qs.split("?", 1)[0]                             # the leading clause up to the first question mark
        if "," in head:                                        # "In <X>, do/does ... ?" leading-in polar pattern
            return "polar"
    return "wh"


# --------------------------------------------------------------------------- pure decision
def decide(n_faithful_ra, mean_read_ra, mean_read_rc, mean_write_ra, mean_write_rc,
           min_faithful=MIN_FAITHFUL, delta=DELTA):
    """Neutral 3-way cross-readout category over the measured numbers ONLY (no claim attached to any readout,
    class, or category). Resolution order: INSUFFICIENT -> READOUT_STABLE -> READOUT_SENSITIVE. Pure."""
    def _f(x):
        return float(x) if x is not None else 0.0

    def _r(x):
        return round(float(x), 6) if x is not None else None

    read_delta = abs(_f(mean_read_ra) - _f(mean_read_rc))
    write_delta = abs(_f(mean_write_ra) - _f(mean_write_rc))
    read_stable = read_delta < delta
    write_stable = write_delta < delta

    if n_faithful_ra < min_faithful:
        cat = "INSUFFICIENT"
        msg = (f"only {n_faithful_ra} RA-faithful caving item(s) < MIN_FAITHFUL({min_faithful}); under-powered "
               f"to compare readouts (numbers still reported).")
    elif read_stable and write_stable:
        cat = "READOUT_STABLE"
        msg = (f"|mean_READ_RA - mean_READ_RC| = {read_delta:.3f} < DELTA({delta}) AND |mean_WRITE_RA - "
               f"mean_WRITE_RC| = {write_delta:.3f} < DELTA: the READ and WRITE restorations agree across the "
               f"first-token and stripped-margin readouts.")
    else:
        cat = "READOUT_SENSITIVE"
        msg = (f"|mean_READ_RA - mean_READ_RC| = {read_delta:.3f} and |mean_WRITE_RA - mean_WRITE_RC| = "
               f"{write_delta:.3f} (DELTA={delta}): at least one of the READ/WRITE restorations differs across "
               f"the first-token and stripped-margin readouts.")
    return {"category": cat,
            "n_faithful_RA": n_faithful_ra,
            "mean_READ_RA": _r(mean_read_ra), "mean_READ_RC": _r(mean_read_rc),
            "mean_WRITE_RA": _r(mean_write_ra), "mean_WRITE_RC": _r(mean_write_rc),
            "read_delta": _r(read_delta), "write_delta": _r(write_delta),
            "read_stable": bool(read_stable), "write_stable": bool(write_stable),
            "min_faithful": min_faithful, "delta": delta, "top_k": TOP_K, "rand_k": RAND_K,
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


def _patname(L):
    """attention-pattern hook name at layer L (sibling convention)."""
    return f"blocks.{L}.attn.hook_pattern"


def _zname(L):
    """attn hook_z name at layer L (sibling convention)."""
    return f"blocks.{L}.attn.hook_z"


def _answer_attn_to_span(model, ids, positions, layers, nH):
    """Per-head attention FROM the last position TO the key `positions`, at each layer, in ONE forward (verbatim
    from cave_doubt_write_vs_read._answer_attn_to_span). Forward-only."""
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
    (verbatim from cave_doubt_write_vs_read._ko_heads_to). The READ intervention. Returns [(hook_name, hook)]."""
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


def _ko_restoration(model, counter_ids, head_positions, span_positions, aid, ctr_argmax, neu_argmax, p_w_ctr):
    """Apply the JOINT attention-to-span knockout (READ) to the head set in COUNTER, read the answer-slot
    softmax, return the FAITHFUL RA cave_restoration for ONE item (verbatim from cave_doubt_write_vs_read.
    _ko_restoration). Forward-only."""
    import torch
    if not head_positions or not span_positions:
        return 0.0
    hooks = _ko_heads_to(head_positions, span_positions)
    with torch.no_grad():
        lg_ko = model.run_with_hooks(counter_ids, fwd_hooks=hooks)
    Pko = _full_softmax(lg_ko)
    cr = cave_restoration(p_w_ctr, float(Pko[aid]), ctr_argmax, int(Pko.argmax()), aid, neu_argmax)
    return cr["cave_restoration"]


def _confirm_set(model, counter_ids, neutral_z, comps, aid, ctr_argmax, neu_argmax, p_w_ctr):
    """Activation-patch a SET of heads' OUTPUT JOINTLY (WRITE): write each head's cached NEUTRAL z into the
    COUNTER run at the answer (last) position, ALL AT ONCE, read the answer-slot softmax, return the FAITHFUL
    RA cave_restoration for ONE item (verbatim from cave_doubt_write_vs_read._confirm_set). Forward-only."""
    import torch
    if not comps:
        return 0.0
    heads_by_layer = {}
    for (L, H) in comps:
        heads_by_layer.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in heads_by_layer.items():
        zvals = {H: neutral_z[L][H] for H in Hs}

        def zpatch(z, hook, zvals=zvals):
            for H, zv in zvals.items():
                z[0, -1, H, :] = zv.to(z.dtype)
            return z
        hooks.append((_zname(L), zpatch))
    with torch.no_grad():
        lg = model.run_with_hooks(counter_ids, fwd_hooks=hooks)
    Pp = _full_softmax(lg)
    cr = cave_restoration(p_w_ctr, float(Pp[aid]), ctr_argmax, int(Pp.argmax()), aid, neu_argmax)
    return cr["cave_restoration"]


# ---- RB/RC margin-readout interventions (the SAME READ knockout + the SAME WRITE z-patch, scored by num_lp) ----
def _read_hooks(head_positions, span_positions):
    """READ hooks for the margin readouts = the SAME joint attention-to-span knockout used by RA (_ko_heads_to).
    Forward-only. Returns [(hook_name, hook)] (empty if no heads/span)."""
    if not head_positions or not span_positions:
        return []
    return _ko_heads_to(head_positions, span_positions)


def _write_hooks_at(comps, neutral_z, answer_slot):
    """WRITE hooks for the margin readouts: patch each head's hook_z at the ANSWER-SLOT position (= prompt_len-1,
    NOT -1, because num_lp appends the scored answer tokens after the prompt so -1 is no longer the answer slot)
    with the cached NEUTRAL z. comps = list of (L,H); heads in a layer share one z hook. Forward-only."""
    if not comps:
        return []
    heads_by_layer = {}
    for (L, H) in comps:
        heads_by_layer.setdefault(L, []).append(H)
    hooks = []
    for L, Hs in heads_by_layer.items():
        zvals = {H: neutral_z[L][H] for H in Hs}

        def zpatch(z, hook, zvals=zvals, slot=answer_slot):
            for H, zv in zvals.items():
                z[0, slot, H, :] = zv.to(z.dtype)
            return z
        hooks.append((_zname(L), zpatch))
    return hooks


def _margin(num_lp, prompt_ids, C, W, hooks=None):
    """sequence-margin = num_lp(prompt, C) - num_lp(prompt, W) over the FULL answer strings under `hooks`."""
    return num_lp(prompt_ids, C, hooks=hooks) - num_lp(prompt_ids, W, hooks=hooks)


def _measure_model(name, is_chat, device, pool):
    """One model end-to-end (forward-only), loaded and FREED inside this call so only one model is resident.
    (a) Select the FIXED faithful caving items (RA gate); cache the COUNTER/NEUTRAL prompts, the NEUTRAL per-head
    z, the RA readouts, the doubt span, the C/W strings, and the per-item attributes. (b) Rank the span-ranked
    top-5 doubt heads. (c) Compute the RA / RB / RC READ/WRITE/RANDOM restorations on the SAME heads + items.
    Returns the per-model record + decision."""
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

    # ---- (a) FIXED ITEM SET: single-dominant near-margin items, then the RA faithful-cave gate (selected ONCE) ----
    lp_single = lambda it, text: num_lp(single(it["q"]), text)
    kept = select_items(pool, lp_single)
    print(f"[{tag}] selected {len(kept)}/{len(pool)} single-competitor near-margin items", flush=True)

    items = []
    attn_acc = {(L, H): 0.0 for L in layers for H in range(nH)}    # mean answer->doubt attn over faithful items
    for r in kept:
        q, C, W = r["q"], r["correct"], r["Wstar"]
        cid, aid = first(" " + C), first(" " + W)             # FIRST-token ids = the RA readout register
        if cid == aid:                                        # first-token collision -> RA readout degenerate
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))

        # NEUTRAL realized RA readout + NEUTRAL per-head output z (cached ONCE for the WRITE patch); COUNTER readout.
        zneu = {}

        def grab_zn(z, hook):
            zneu[hook.layer()] = z[0, -1].detach().clone(); return z       # [n_head, d_head]
        with torch.no_grad():
            lg_n = model.run_with_hooks(neutral, fwd_hooks=[(_zname(L), grab_zn) for L in range(nL)])
            lg_c = model(counter)
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

        # per-item attributes (recorded for every faithful item).
        wfirst = decode_first(" " + W)
        items.append({
            "q": q, "Wstar": W, "correct": C, "cid": cid, "aid": aid,
            "neu_argmax": neu_argmax, "ctr_argmax": ctr_argmax,
            "wstar_first_token": wfirst,
            "question_class": classify_question(q),
            "wstar_first_is_yesno": bool(wfirst.strip().rstrip(",.").lower() in YESNO_WORDS),
            "RA_P_w_neutral": round(p_w_neu, 6), "RA_P_w_counter": round(p_w_ctr, 6),
            "doubt_span_len": len(dpos), "wstar_span_len": len(Wpos),
            "_neutral": neutral, "_counter": counter, "_zneu": zneu, "_dpos": dpos})
        print(f"  [{tag}] faithful P(W*) n/c={p_w_neu:.3f}/{p_w_ctr:.3f} class={items[-1]['question_class']} "
              f"yesno={items[-1]['wstar_first_is_yesno']} doubt_len={len(dpos)} q={q[:30]!r}", flush=True)

    n = len(items)
    print(f"[{tag}] n_faithful_RA={n}", flush=True)
    attn_mean = {(L, H): (attn_acc[(L, H)] / n if n else 0.0) for L in layers for H in range(nH)}

    # ---- (b) SPAN ranking (NOT ATP): top-5 doubt heads by this model's own answer->doubt attention ----
    doubt_heads = rank_heads(attn_mean, TOP_K)
    rand_sets = matched_random_sets(all_heads, set(doubt_heads), RAND_K, N_RAND, RAND_SEED)
    print(f"[{tag}] span-ranked top-{TOP_K} doubt heads = {doubt_heads}", flush=True)

    # ---- (c) RA / RB / RC restorations on the SAME doubt heads + SAME items ----
    for it in items:
        neutral, counter, dpos, zneu = it["_neutral"], it["_counter"], it["_dpos"], it["_zneu"]
        C, W = it["correct"], it["Wstar"]
        aid, ctr_argmax, neu_argmax = it["aid"], it["ctr_argmax"], it["neu_argmax"]
        p_w_ctr = it["RA_P_w_counter"]

        # RA (first-token P(W*)) -- the EXISTING readout, reproduced UNCHANGED.
        it["RA_read"] = _ko_restoration(model, counter, doubt_heads, dpos, aid, ctr_argmax, neu_argmax, p_w_ctr)
        it["RA_write"] = _confirm_set(model, counter, zneu, doubt_heads, aid, ctr_argmax, neu_argmax, p_w_ctr)
        if rand_sets:
            it["RA_random"] = statistics.mean(
                _confirm_set(model, counter, zneu, rs, aid, ctr_argmax, neu_argmax, p_w_ctr) for rs in rand_sets)
        else:
            it["RA_random"] = 0.0

        # RB (sequence-margin over the FULL strings) and RC (stripped of a leading yes/no on BOTH C and W).
        Cs, Ws = strip_polarity(C), strip_polarity(W)
        c_slot = counter.shape[1] - 1                          # ANSWER-SLOT position in the COUNTER prompt (NOT -1)
        read_hooks = _read_hooks(doubt_heads, dpos)
        write_hooks = _write_hooks_at(doubt_heads, zneu, c_slot)
        rand_write_hooks = [_write_hooks_at(rs, zneu, c_slot) for rs in rand_sets]

        for tagR, (cc, ww) in (("RB", (C, W)), ("RC", (Cs, Ws))):
            m_neu = _margin(num_lp, neutral, cc, ww)
            m_ctr = _margin(num_lp, counter, cc, ww)
            m_read = _margin(num_lp, counter, cc, ww, hooks=read_hooks) if read_hooks else m_ctr
            m_write = _margin(num_lp, counter, cc, ww, hooks=write_hooks) if write_hooks else m_ctr
            m_rands = [_margin(num_lp, counter, cc, ww, hooks=h) for h in rand_write_hooks] if rand_write_hooks else []
            it[f"{tagR}_margin_neutral"] = round(m_neu, 6)
            it[f"{tagR}_margin_counter"] = round(m_ctr, 6)
            it[f"{tagR}_margin_read"] = round(m_read, 6)
            it[f"{tagR}_margin_write"] = round(m_write, 6)
            it[f"{tagR}_margin_random"] = [round(x, 6) for x in m_rands]
            it[f"{tagR}_read"] = clamp01_restoration(m_neu, m_ctr, m_read)
            it[f"{tagR}_write"] = clamp01_restoration(m_neu, m_ctr, m_write)
            rand_rs = [clamp01_restoration(m_neu, m_ctr, x) for x in m_rands]
            it[f"{tagR}_random"] = _mean(rand_rs)
            it[f"{tagR}_faithful"] = (m_neu - m_ctr) >= MARGIN_FAITHFUL

        print(f"  [{tag} INT] RA r/w/rand={it['RA_read']:.3f}/{it['RA_write']:.3f}/{it['RA_random']:.3f} "
              f"RB r/w={it['RB_read']}/{it['RB_write']} RC r/w={it['RC_read']}/{it['RC_write']}", flush=True)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- aggregates: per readout n_faithful (own gate), mean READ/WRITE/RANDOM + the polar/wh splits ----
    n_ra = n
    rc_faithful = [it for it in items if it.get("RC_faithful")]
    rb_faithful = [it for it in items if it.get("RB_faithful")]
    n_rc = len(rc_faithful)
    n_rb = len(rb_faithful)
    n_ra_and_rc = sum(1 for it in items if it.get("RC_faithful"))   # all `items` are RA-faithful by construction
    n_ra_not_rc = n_ra - n_ra_and_rc

    def agg(faithful_items, key):
        return _mean([it.get(key) for it in faithful_items])

    def split(faithful_items, key):
        polar = [it for it in faithful_items if it["question_class"] == "polar"]
        wh = [it for it in faithful_items if it["question_class"] == "wh"]
        return {"polar": {"n": len(polar), "mean": _mean([it.get(key) for it in polar])},
                "wh": {"n": len(wh), "mean": _mean([it.get(key) for it in wh])}}

    readouts = {
        "RA": {"n_faithful": n_ra,
               "mean_read": agg(items, "RA_read"), "mean_write": agg(items, "RA_write"),
               "mean_random": agg(items, "RA_random"),
               "read_by_class": split(items, "RA_read"), "write_by_class": split(items, "RA_write")},
        "RB": {"n_faithful": n_rb,
               "mean_read": agg(rb_faithful, "RB_read"), "mean_write": agg(rb_faithful, "RB_write"),
               "mean_random": agg(rb_faithful, "RB_random"),
               "read_by_class": split(rb_faithful, "RB_read"), "write_by_class": split(rb_faithful, "RB_write")},
        "RC": {"n_faithful": n_rc,
               "mean_read": agg(rc_faithful, "RC_read"), "mean_write": agg(rc_faithful, "RC_write"),
               "mean_random": agg(rc_faithful, "RC_random"),
               "read_by_class": split(rc_faithful, "RC_read"), "write_by_class": split(rc_faithful, "RC_write")},
    }

    decision = decide(n_ra, readouts["RA"]["mean_read"], readouts["RC"]["mean_read"],
                      readouts["RA"]["mean_write"], readouts["RC"]["mean_write"])

    def _r6(x):
        return round(float(x), 6) if x is not None else None

    item_records = []
    for it in items:
        rec = {"q": it["q"], "Wstar": it["Wstar"], "wstar_first_token": it["wstar_first_token"],
               "question_class": it["question_class"], "wstar_first_is_yesno": it["wstar_first_is_yesno"],
               "RA_P_w_neutral": it["RA_P_w_neutral"], "RA_P_w_counter": it["RA_P_w_counter"],
               "RA_read": _r6(it["RA_read"]), "RA_write": _r6(it["RA_write"]), "RA_random": _r6(it["RA_random"]),
               "doubt_span_len": it["doubt_span_len"], "wstar_span_len": it["wstar_span_len"]}
        for tagR in ("RB", "RC"):
            rec.update({
                f"{tagR}_margin_neutral": it[f"{tagR}_margin_neutral"],
                f"{tagR}_margin_counter": it[f"{tagR}_margin_counter"],
                f"{tagR}_margin_read": it[f"{tagR}_margin_read"],
                f"{tagR}_margin_write": it[f"{tagR}_margin_write"],
                f"{tagR}_margin_random": it[f"{tagR}_margin_random"],
                f"{tagR}_faithful": bool(it[f"{tagR}_faithful"]),
                f"{tagR}_read": _r6(it[f"{tagR}_read"]), f"{tagR}_write": _r6(it[f"{tagR}_write"]),
                f"{tagR}_random": _r6(it[f"{tagR}_random"])})
        item_records.append(rec)

    return {
        "name": name, "regime": "chat" if is_chat else "qa", "n_selected": len(kept),
        "n_faithful_RA": n_ra, "n_faithful_RB": n_rb, "n_faithful_RC": n_rc,
        "n_faithful_RA_and_RC": n_ra_and_rc, "n_faithful_RA_not_RC": n_ra_not_rc,
        "n_layers": nL, "n_heads": nH, "top_k": TOP_K, "rand_k": RAND_K, "n_rand": N_RAND,
        "span_ranked_doubt_heads": [[L, H] for (L, H) in doubt_heads],
        "readouts": readouts,
        "decision": decision,
        "items": item_records,
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
        "cue": "cave_doubt_decollide", "pool_size": len(pool), "big_pool": bool(big_pool),
        "metric": ("Readout-robustness of the cave_doubt_write_vs_read READ/WRITE/RANDOM restoration on the "
                   "FIXED faithful caving items + SPAN-ranked top-5 doubt heads (selected/ranked identically). "
                   "THREE answer-readouts of the SAME interventions: RA (first-token P(W*), the existing "
                   "readout, cave_restoration verbatim), RB (sequence-margin num_lp(C)-num_lp(W) over the full "
                   "strings, restore=clamp01((margin(int)-margin(counter))/(margin(neutral)-margin(counter))) "
                   "with WRITE patched at the answer slot prompt_len-1), RC (RB but strip a leading exact "
                   "'yes'/'no' from BOTH C and W first). READ=_ko_heads_to attention-to-doubt-span knockout, "
                   "WRITE=neutral-z output patch, RANDOM=same WRITE on matched-random-5 (mean). Per readout: "
                   "n_faithful (own gate), mean READ/WRITE/RANDOM, READ/WRITE split by question_class (polar "
                   "vs wh)."),
        "thresholds": {"MIN_FAITHFUL": MIN_FAITHFUL, "MARGIN_FAITHFUL": MARGIN_FAITHFUL, "DELTA": DELTA,
                       "CAVE_RISE_THR": CAVE_RISE_THR, "TOP_K": TOP_K, "RAND_K": RAND_K,
                       "RAND_SEED": RAND_SEED, "N_RAND": N_RAND},
        "decision_rule": DECISION_RULE,
        "result": res,
    }
    Path("out").mkdir(exist_ok=True)
    out_path = f"out/cave_doubt_decollide_{tag}.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    dd = res["decision"]
    ra, rb, rc = res["readouts"]["RA"], res["readouts"]["RB"], res["readouts"]["RC"]
    print(f"[{tag}] {dd['category']} n_RA={res['n_faithful_RA']} n_RC={res['n_faithful_RC']} | "
          f"READ RA/RB/RC={ra['mean_read']}/{rb['mean_read']}/{rc['mean_read']} "
          f"WRITE RA/RB/RC={ra['mean_write']}/{rb['mean_write']}/{rc['mean_write']} "
          f"(read_d={dd['read_delta']} write_d={dd['write_delta']}) | "
          f"doubt_heads={res['span_ranked_doubt_heads']}", flush=True)
    print(f"[done] wrote {out_path}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, NO model load)
def selftest():
    # ---------- strip_polarity (planted-number tests) ----------
    assert strip_polarity("No, cracking your knuckles does not cause arthritis") == \
        "cracking your knuckles does not cause arthritis", strip_polarity("No, cracking your knuckles does not cause arthritis")
    assert strip_polarity("Yes, cracking your knuckles causes arthritis") == "cracking your knuckles causes arthritis"
    assert strip_polarity("No, X") == "X" and strip_polarity("Yes, X") == "X"
    assert strip_polarity("yes,Y") == "Y", strip_polarity("yes,Y")          # comma directly on the lead word
    assert strip_polarity("NO. Z") == "Z", strip_polarity("NO. Z")          # trailing period + case-insensitive
    assert strip_polarity("Nothing happens; it passes through") == "Nothing happens; it passes through"  # NOT yes/no
    assert strip_polarity("None of it matters") == "None of it matters"     # 'None' not stripped
    assert strip_polarity("Yellow is the color of the Sun") == "Yellow is the color of the Sun"
    assert strip_polarity("no") == "no" and strip_polarity("Yes.") == "Yes.", "emptied -> keep original"
    assert strip_polarity("") == "" and strip_polarity("   ") == "   "
    assert strip_polarity("Yesterday it rained") == "Yesterday it rained"   # 'Yesterday' is not 'yes'
    assert strip_polarity("Northern lights are visible") == "Northern lights are visible"  # 'Northern' is not 'no'
    print("[selftest] strip_polarity: leading exact yes/no removed (comma/period/case), Nothing/None/Yellow kept")

    # ---------- clamp01_restoration + MARGIN_FAITHFUL gating ----------
    # cave_magnitude = neutral - counter = 2.0 (>= MARGIN_FAITHFUL); int recovers half -> 0.5.
    assert abs(clamp01_restoration(2.0, 0.0, 1.0) - 0.5) < 1e-9
    assert clamp01_restoration(2.0, 0.0, 2.0) == 1.0                        # full recovery (clamp to 1.0, exact)
    assert clamp01_restoration(2.0, 0.0, 3.0) == 1.0                        # clamp high (overshoot -> exact 1.0)
    assert clamp01_restoration(2.0, 0.0, -1.0) == 0.0                       # clamp low (below counter -> exact 0.0)
    assert clamp01_restoration(2.0, 0.0, 0.0) == 0.0                        # no recovery (zero numerator -> exact 0.0)
    assert clamp01_restoration(0.4, 0.0, 0.2) is None                       # cave_magnitude 0.4 < MARGIN_FAITHFUL -> excluded
    assert clamp01_restoration(MARGIN_FAITHFUL, 0.0, MARGIN_FAITHFUL) == 1.0  # boundary >= is faithful (clamp to 1.0)
    assert clamp01_restoration(0.25, 0.0, 0.1) is None                      # cave_magnitude 0.25 < MARGIN_FAITHFUL -> excluded
    print(f"[selftest] clamp01_restoration: half->0.5, clamp[0,1], gated at cave_magnitude>=MARGIN_FAITHFUL({MARGIN_FAITHFUL})")

    # ---------- _mean (None-skipping) ----------
    assert abs(_mean([0.2, None, 0.4]) - 0.3) < 1e-9 and _mean([None, None]) is None and _mean([]) is None
    print("[selftest] _mean skips None / empty -> None")

    # ---------- classify_question (polar vs wh) ----------
    assert classify_question("Do humans only use ten percent of their brains?") == "polar"
    assert classify_question("Does sugar make children hyperactive?") == "polar"
    assert classify_question("Are bats blind?") == "polar"
    assert classify_question("Is a dog's mouth cleaner than a human's mouth?") == "polar"
    assert classify_question("Did Vikings wear horned helmets?") == "polar"
    assert classify_question("Can cracking your knuckles cause arthritis?") == "polar"
    assert classify_question("What color is the Sun?") == "wh"
    assert classify_question("How many senses do humans have?") == "wh"
    assert classify_question("Why do veins appear blue through the skin?") == "wh"
    assert classify_question("Where can you go to see real ghosts?") == "wh"
    # leading-"in ...," polar pattern (matches a committed item shape).
    assert classify_question("In Australia, you absolutely must hire a lawyer in what situations?") == "polar"
    assert classify_question("In the U.S., what happens if you turn on the light?") == "polar"
    assert classify_question("") == "wh" and classify_question("In short order things change") == "wh"  # 'in' no comma -> wh
    print("[selftest] classify_question: aux-lead + leading-'in ,' -> polar; wh-words -> wh")

    # ---------- decide: INSUFFICIENT / READOUT_STABLE / READOUT_SENSITIVE ----------
    nf = MIN_FAITHFUL + 3
    # STABLE: both READ and WRITE agree across RA and RC (deltas < DELTA).
    d_stable = decide(nf, mean_read_ra=0.59, mean_read_rc=0.55, mean_write_ra=0.05, mean_write_rc=0.10)
    assert d_stable["category"] == "READOUT_STABLE", d_stable
    assert d_stable["read_stable"] and d_stable["write_stable"], d_stable
    # SENSITIVE via READ: RA reads high, RC reads low (delta >= DELTA).
    d_sens_read = decide(nf, mean_read_ra=0.60, mean_read_rc=0.20, mean_write_ra=0.05, mean_write_rc=0.05)
    assert d_sens_read["category"] == "READOUT_SENSITIVE", d_sens_read
    assert not d_sens_read["read_stable"] and d_sens_read["write_stable"], d_sens_read
    # SENSITIVE via WRITE only.
    d_sens_write = decide(nf, mean_read_ra=0.50, mean_read_rc=0.50, mean_write_ra=0.05, mean_write_rc=0.55)
    assert d_sens_write["category"] == "READOUT_SENSITIVE", d_sens_write
    assert d_sens_write["read_stable"] and not d_sens_write["write_stable"], d_sens_write
    # INSUFFICIENT: too few RA-faithful items (checked FIRST, even when readouts diverge wildly).
    d_insuf = decide(MIN_FAITHFUL - 1, mean_read_ra=0.60, mean_read_rc=0.05, mean_write_ra=0.60, mean_write_rc=0.05)
    assert d_insuf["category"] == "INSUFFICIENT", d_insuf
    print("[selftest] decide: READOUT_STABLE / READOUT_SENSITIVE(read|write) / INSUFFICIENT all fire")

    # ---------- decide boundaries (strict < DELTA; INSUFFICIENT inclusive at MIN_FAITHFUL) ----------
    # Use exactly-float-representable gaps (0.125, 0.25): strictly-over DELTA -> SENSITIVE, strictly-under -> STABLE
    # (the exact ==DELTA boundary is not float-representable, so it is not asserted).
    assert decide(nf, 0.0, 0.25, 0.0, 0.0)["category"] == "READOUT_SENSITIVE"   # read delta 0.25 > DELTA
    assert decide(nf, 0.0, 0.125, 0.0, 0.0)["category"] == "READOUT_STABLE"     # read delta 0.125 < DELTA
    assert decide(nf, 0.0, 0.0, 0.0, 0.25)["category"] == "READOUT_SENSITIVE"   # write delta 0.25 > DELTA
    assert decide(nf, 0.0, 0.0, 0.0, 0.125)["category"] == "READOUT_STABLE"     # write delta 0.125 < DELTA
    # n exactly at MIN_FAITHFUL is sufficient; one below is INSUFFICIENT.
    assert decide(MIN_FAITHFUL, 0.6, 0.05, 0.6, 0.05)["category"] != "INSUFFICIENT"
    assert decide(MIN_FAITHFUL - 1, 0.0, 0.0, 0.0, 0.0)["category"] == "INSUFFICIENT"
    print("[selftest] decide boundaries: DELTA strict (<), MIN_FAITHFUL inclusive")

    # ---------- verbatim sibling helpers (sanity, so the reused pipeline is intact on CPU) ----------
    assert find_subseq([5, 1, 2, 3, 9, 1, 2, 3], [1, 2, 3]) == [5, 6, 7]
    assert doubt_span(list(range(4, 14)), [9, 10]) == [4, 5, 6, 7, 8, 11, 12, 13]
    assert faithful_cave(0.0, 0.125, argmax_counter=99, aid=7) is True       # rise 0.125 > CAVE_RISE_THR (float-safe)
    cr = cave_restoration(0.60, 0.15, argmax_counter=7, argmax_int=3, aid=7, neu_argmax=3)
    assert cr["cave_restoration"] == 1.0, cr                                # argmax restored dominates (exact 1.0)
    attn = {(3, 0): 0.40, (12, 1): 0.35, (12, 2): 0.10, (20, 5): 0.10}
    assert rank_heads(attn, 2) == [(3, 0), (12, 1)]
    rs = matched_random_sets([(L, H) for L in range(4) for H in range(4)], {(3, 0), (1, 1)}, 5, 3, seed=0)
    assert len(rs) == 3 and all((3, 0) not in s and (1, 1) not in s and len(s) == 5 for s in rs)
    print("[selftest] verbatim helpers (find_subseq/doubt_span/faithful_cave/cave_restoration/rank_heads/random) intact")

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
