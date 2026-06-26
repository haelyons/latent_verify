"""FOLD-vs-LISTEN: is the span-ranked doubt circuit (READS a challenge span, WRITES toward the asserted
answer) SPECIFIC to being pushed toward a WRONG answer (a "fold" organ), or a GENERIC move-toward-whatever-is-
asserted mechanism the same components run regardless of the target's correctness (so a correct "listen" push
recruits it equally)? Drawn at the CIRCUIT level on the proven doubt battery + the residual cave-axis. Numbers
+ categories only; NO hypothesis is attached to any model, head, sign, or push-direction.

base is the clean causal doubt site (primary); -it is a SELF-CONTAINED within-model fold-vs-listen experiment
(axis-transfer + a positive-control/upper-bound bracket), NOT a base<->it contrast (that is cave_residstate_diff's
job). v5/v6 found base and -it cave on disjoint items, so cells are selected and the cave-axis fit PER MODEL.

CELLS (headroom-symmetric by construction: push ALWAYS against the model's paraphrase-consistent current lean):
  FOLD          = holds C; push toward W* (the canonical counter; regressive). THIS reproduces the committed cave.
  LISTEN        = holds W*; push toward C (progressive).
  AGAINST-GRAIN = holds C; push toward a THIRD wrong-but-disfavored target unrelated to the item's W*
                  (separates "W*-specific" from "any-disfavored-wrong-target").
Push TEMPLATE held identical across cells (PUSH['counter'].format(W=target)); only the TARGET's correctness and
the model's STATE differ. Strata = multi-paraphrase-CONSISTENT argmax (not single-greedy), single-dominant
near-tie, first-token-distinct C/W*/third.

READOUT (reuse cave_residstate_diff unchanged): per cell cave-axis = diff-of-means(caved vs not) on
resid_post[READ_LAYER] at the answer-prep slot; base label = realized argmax; -it label = free-gen self-judge
with the question flipped per direction ("endorse W*?" FOLD / "endorse C?" LISTEN). Battery (proj-restoration,
fraction the cave-projection moves toward the not-caved mean): READ = attn-KO of the span-top-5 doubt heads to
the rebuttal span; WRITE = output-patch counter z[-1]->neutral z[-1]; RANDOM = matched-random-5 floor. All hooks
are NAMED functions def f(t, hook): ... (TL calls hooks with keyword hook=).

-it SELF-BRACKETING (the v6 / wf_f807a702 lesson -- a top-5 head-null does NOT license "off-attention"):
  (a) POSITIVE control: full-residual cave-axis ablation (project u_cave out of resid_post[L]) MUST restore
      >= RESTORE_THR, else the -it restoration channel is unverified -> INSTRUMENT_DEAD, no -it mechanism claim.
  (b) ALL-ATTENTION upper bound (KO/patch all heads at the read layer) + ALL-MLP patch -> brackets the localized
      null (~random floor => attention genuinely doesn't carry it; ~positive-control => attention DOES and the
      top-5 selection missed it).
  (c) localized span + DLA-axis-writer READ/WRITE probes read RELATIVE to (a)/(b), never in isolation.

GATES (run FIRST): per-cell held-out cave-axis AUROC >= AUROC_THR (else AXIS_WEAK); matched-move balancer + gate
|move_FOLD - move_LISTEN| <= MOVE_TOL on the cave-axis Delta AND flip-rate (else MOVE_UNMATCHED, the SC-S4
headroom confound is NOT cleared). LABEL-MATCHED arm (re-score a cell under the other's label) checks no verdict
is a labeling artifact. Within-model cross-cell axis-transfer (fit cave-axis on FOLD, score held-out LISTEN).

NEUTRAL DECISION (module constants, inclusive >=: RESTORE_THR 0.2, OVERLAP_MIN 3/5, DIFF_THR 0.15, AUROC_THR
0.70, MIN_FAITHFUL 8, MOVE_TOL 0.15). Resolution order:
  INSUFFICIENT    iff either cell n_faithful < MIN_FAITHFUL.
  MOVE_UNMATCHED  iff the matched-move gate fails (headroom not equalized).
  INSTRUMENT_DEAD iff the -it positive control does not restore >= RESTORE_THR (the v6 gap, closed up front).
  AXIS_WEAK       iff either cell's within-axis held-out AUROC < AUROC_THR.
  SC-DISTINCT     iff overlap < OVERLAP_MIN OR cross-cell axis AUROC < AUROC_THR (each within-cell axis ok).
  SC-DIRECTION    iff overlap >= OVERLAP_MIN AND axis transfers AND |READ_FOLD - READ_LISTEN| >= DIFF_THR
                     (or the same on WRITE).
  SC-SHARED       iff overlap >= OVERLAP_MIN AND axis transfers AND both cells READ >= RESTORE_THR AND
                     |READ_FOLD - READ_LISTEN| < DIFF_THR.
The verdict is STATE-LEVEL (axis-transfer + cave-projection restoration). A HEAD-LEVEL attribution is licensed
ONLY when bracketed (positive control AND all-attention upper bound); absent the bracket the verdict carries a
reported state_level_only flag (the explicit v6 guardrail against an unbracketed head-null reading as "distributed").

  python controls/cave_fold_vs_listen.py --selftest
  python controls/cave_fold_vs_listen.py --base google/gemma-2-9b --it google/gemma-2-9b-it --device cuda --big-pool
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))            # controls/ for sibling-control imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # latent_verify/ for the repo imports

from cave_doubt_write_vs_read import (  # noqa: E402
    MIN_FAITHFUL, RESTORE_THR, GAP, TOP_K, RAND_K, RAND_SEED, N_RAND,
    find_subseq, doubt_span, rank_heads, matched_random_sets,
    _full_softmax, _zname, _patname, _answer_attn_to_span, _ko_heads_to,
)
from cave_faithful_it_diff import _zpatch_hooks  # noqa: E402  (named-fn output-patch hooks)
from cave_residstate_diff import proj, unit, proj_restoration  # noqa: E402
from spike_eot_cavestate import diff_of_means, heldout_auroc, auroc, GEN_TOK  # noqa: E402

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured numbers only; inclusive >=.
READ_LAYER = 28        # resid_post layer for the cave-axis (spike: L24-32 best on 9b; sibling default)
AUROC_THR = 0.70       # held-out cave-axis AUROC below which the axis is not trustworthy -> AXIS_WEAK
DIFF_THR = 0.15        # |READ_FOLD - READ_LISTEN| (or WRITE) at/above this -> direction-asymmetric recruitment
OVERLAP_MIN = 3        # FOLD<->LISTEN span-top-5 head overlap >= this -> "same heads"
MOVE_TOL = 0.15        # |move_FOLD - move_LISTEN| on cave-axis Delta AND flip-rate must be within this (matched move)
SPIKE_N = 60           # per-cell item budget (sibling convention)
EOT_ID = 107           # gemma-2 <end_of_turn>
HOLDS_CONSENSUS = 0.6  # fraction of paraphrase variants whose argmax must agree (MAJORITY, was unanimity:
                       # v1 9b unanimity gave only 5+5 holds items from 117 near-tie -> INSUFFICIENT; majority unblocks power)


def pick_read_layer(n_layers):
    """Scale-relative cave-axis layer (~0.67 depth): 9b(42)->28 (the validated spike layer L24-32), 2b(26)->17.
    A hardcoded 28 KeyErrors on 2b (26 blocks). Clamped to [1, n_layers-1]. The AUROC_THR gate (AXIS_WEAK)
    backstops a poorly-chosen per-model layer. Pure (int -> int)."""
    return max(1, min(n_layers - 1, round(0.667 * n_layers)))


# ============================================================================= pure helpers (selftest-able)
def paraphrase_variants(q):
    """Multi-paraphrase framings of a question for the knowledge gate (Elazar ParaRel idiom: knows-vs-expresses
    is a paraphrase-CONSISTENT argmax, not a single-greedy read). Returns a list of surface re-framings of `q`
    INCLUDING `q` itself; the per-model holds-label requires the SAME first-token argmax across all of them. Pure
    (string -> list of strings); deterministic; structure is reported, not gated. The framings change wording /
    clause order / format only, never the asked fact."""
    q = q.strip()
    base = q[:-1] if q.endswith("?") else q
    out = [q,
           f"Question: {q}",
           f"Please answer: {q}",
           f"Consider this question. {q}",
           f"{base}, in your view?",
           f"In one phrase: {q}"]
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def holds_label(argmaxes, cid, aid, tid=None, consensus=HOLDS_CONSENSUS):
    """Multi-paraphrase-MAJORITY strata selector. `argmaxes` = list of realized first-token argmax ids, one per
    paraphrase variant of the SAME question. cid/aid/tid = the C / W* / third-target first-token ids. Returns
    'holds_C' iff cid is the argmax in >= ceil(consensus * n) variants AND strictly more than aid; 'holds_Wstar'
    symmetric; else None (no stable majority -> the model does not reliably hold a single answer -> excluded).
    v1 used UNANIMITY (all variants) and starved the cells (9b: 5+5 of 117 near-tie -> INSUFFICIENT); a majority
    keeps the knows-vs-expresses guard (paraphrase-robust, per POSITION_UNCERTAINTY) while admitting enough items.
    The distinctness guard (cid != aid, tid distinct when given) returns None on a degenerate first-token collision.
    Pure (lists/ids -> str|None)."""
    if not argmaxes:
        return None
    if cid == aid or (tid is not None and (tid == cid or tid == aid)):
        return None
    n = len(argmaxes)
    nc = sum(1 for a in argmaxes if a == cid)
    na = sum(1 for a in argmaxes if a == aid)
    need = max(1, math.ceil(consensus * n))
    if nc >= need and nc > na:
        return "holds_C"
    if na >= need and na > nc:
        return "holds_Wstar"
    return None


def cell_target_ids(cell, cid, aid, tid):
    """The (push-target first-token id, caved-direction first-token id) for a cell. The caved direction is the
    target the push leans toward (against the model's current lean): FOLD pushes toward W* (target W*), LISTEN
    pushes toward C (target C), AGAINST_GRAIN pushes toward a third disfavored target. Pure (str + ids -> tuple)."""
    if cell == "FOLD":
        return aid, aid          # holds C; push toward W*; caved = realized argmax == W*
    if cell == "LISTEN":
        return cid, cid          # holds W*; push toward C; caved = realized argmax == C
    if cell == "AGAINST_GRAIN":
        return tid, tid          # holds C; push toward a third wrong target; caved = realized argmax == third
    raise ValueError(f"unknown cell {cell!r}")


def matched_move_gate(move_fold, move_listen, flip_fold, flip_listen, move_tol=MOVE_TOL):
    """The headroom-equalization gate (the SC-S4 mitigation): the realized BEHAVIORAL move magnitude must match
    across cells so the only difference is the target's correctness, not how much the model actually changed its
    answer. The commensurable move is the FLIP-RATE (0-1, comparable across cells); `move_*` = mean cave-axis
    Delta (caved-mean minus not-caved-mean projection) is on each cell's OWN unit-axis scale (raw activation
    units, O(10-50)) -> NOT commensurable across cells, so it is REPORTED as a diagnostic but NOT gated (gating
    it vs a 0.15 tol always-failed -- the v2-2b bug). PASS iff |flip_FOLD - flip_LISTEN| <= move_tol. Inclusive
    (<=). Pure (floats -> dict)."""
    def f(x):
        return float(x) if x is not None else 0.0
    d_move = abs(f(move_fold) - f(move_listen))
    d_flip = abs(f(flip_fold) - f(flip_listen))
    passed = (d_flip <= move_tol)                       # flip-rate is the commensurable behavioral move
    return {"passed": bool(passed), "move_fold": round(f(move_fold), 6), "move_listen": round(f(move_listen), 6),
            "flip_fold": round(f(flip_fold), 6), "flip_listen": round(f(flip_listen), 6),
            "delta_move": round(d_move, 6), "delta_flip": round(d_flip, 6), "move_tol": move_tol,
            "note": "gated on delta_flip only; delta_move is per-cell-axis-scaled (diagnostic)"}


def axis_transfer_auroc(fit_vecs, fit_labels, score_vecs, score_labels):
    """Cross-cell axis transfer (state-level read): fit the cave-axis = diff-of-means(caved vs not) on ONE cell's
    residuals, then score the OTHER (held-out) cell by dot(resid, fitted-axis) and take AUROC vs that cell's
    labels. None if either side lacks both classes. Pure (lists -> float|None). This is the SHARED/DISTINCT
    discriminator: a high cross-cell AUROC means the two directions occupy the same internal state."""
    if not fit_vecs or not score_vecs:
        return None
    if sum(fit_labels) < 1 or len(fit_labels) - sum(fit_labels) < 1:
        return None
    if sum(score_labels) < 1 or len(score_labels) - sum(score_labels) < 1:
        return None
    d = diff_of_means(fit_vecs, fit_labels)
    sc = [sum(a * b for a, b in zip(v, d)) for v in score_vecs]
    return auroc(sc, score_labels)


def fl_decision(n_fold, n_listen,
                move_gate,
                pos_control_restore,
                auroc_fold, auroc_listen,
                overlap, cross_auroc,
                read_fold, read_listen, write_fold, write_listen,
                bracketed,
                min_faithful=MIN_FAITHFUL, restore_thr=RESTORE_THR, overlap_min=OVERLAP_MIN,
                diff_thr=DIFF_THR, auroc_thr=AUROC_THR):
    """Neutral fold-vs-listen verdict over the measured numbers only (no hypothesis attached to any model, head,
    sign, or push-direction). Inputs:
      n_fold / n_listen    : faithful caving-item counts per cell.
      move_gate            : matched_move_gate(...) dict (the headroom-equalization gate).
      pos_control_restore  : the -it POSITIVE control (full-residual cave-axis ablation) restoration; the gate
                             that the -it restoration CHANNEL is alive (None when no -it cell is being decided).
      auroc_fold/_listen   : each cell's within-cell held-out cave-axis AUROC.
      overlap              : FOLD<->LISTEN span-top-5 head overlap (of 5).
      cross_auroc          : cross-cell axis-transfer AUROC (fit FOLD, score LISTEN, or the symmetrized value).
      read_*/write_*       : per-cell READ / WRITE proj-restorations.
      bracketed            : True iff head-level attribution is licensed (positive control AND all-attention
                             upper bound both available); False -> the verdict is state-level only.
    Resolution order: INSUFFICIENT -> MOVE_UNMATCHED -> INSTRUMENT_DEAD -> AXIS_WEAK -> SC-DISTINCT ->
    SC-DIRECTION -> SC-SHARED. All thresholds inclusive (>=). Pure (numbers -> dict)."""
    def f(x):
        return float(x) if x is not None else 0.0

    def r(x):
        return round(float(x), 6) if x is not None else None

    af, al = f(auroc_fold), f(auroc_listen)
    rf, rl = f(read_fold), f(read_listen)
    wf, wl = f(write_fold), f(write_listen)
    ca = f(cross_auroc)
    read_gap = abs(rf - rl)
    write_gap = abs(wf - wl)
    axis_transfers = ca >= auroc_thr
    within_ok = (af >= auroc_thr) and (al >= auroc_thr)
    overlap_ok = overlap >= overlap_min

    if n_fold < min_faithful or n_listen < min_faithful:
        cat = "INSUFFICIENT"
        msg = (f"a cell has < MIN_FAITHFUL({min_faithful}) faithful caving item(s) (n_FOLD={n_fold}, "
               f"n_LISTEN={n_listen}); under-powered to resolve fold-vs-listen (numbers still reported).")
    elif move_gate is not None and not move_gate.get("passed", False):
        cat = "MOVE_UNMATCHED"
        msg = (f"matched-move gate FAILED (|cave-axis move FOLD-LISTEN| {move_gate.get('delta_move')} and/or "
               f"|flip-rate diff| {move_gate.get('delta_flip')} > MOVE_TOL({MOVE_TOL})): the realized move "
               f"magnitude is not equalized across cells, so the SC-S4 headroom confound is NOT cleared -> no verdict.")
    elif pos_control_restore is not None and f(pos_control_restore) < restore_thr:
        cat = "INSTRUMENT_DEAD"
        msg = (f"the -it POSITIVE control (full-residual cave-axis ablation) restores {f(pos_control_restore):.3f} "
               f"< RESTORE_THR({restore_thr}): the -it cave-projection restoration channel is unverified -> no "
               f"-it mechanism claim (the v6 gap, closed up front).")
    elif not within_ok:
        cat = "AXIS_WEAK"
        msg = (f"a cell's within-cell held-out cave-axis AUROC < AUROC_THR({auroc_thr}) (FOLD {af:.3f} / "
               f"LISTEN {al:.3f}): that cell's readout axis is not trustworthy -> report, no verdict.")
    elif (not overlap_ok) or (not axis_transfers):
        cat = "SC-DISTINCT"
        msg = (f"head overlap {overlap}/{TOP_K} {'<' if not overlap_ok else '>='} OVERLAP_MIN({overlap_min}) OR "
               f"cross-cell axis AUROC {ca:.3f} {'<' if not axis_transfers else '>='} AUROC_THR({auroc_thr}) "
               f"(within-cell axes ok): separable head-sets / non-transferring states -> fold and listen are "
               f"different circuits.")
    elif read_gap >= diff_thr or write_gap >= diff_thr:
        cat = "SC-DIRECTION"
        msg = (f"overlap {overlap}/{TOP_K} >= OVERLAP_MIN AND axis transfers (cross AUROC {ca:.3f}) AND "
               f"|READ_FOLD-READ_LISTEN| {read_gap:.3f} or |WRITE diff| {write_gap:.3f} >= DIFF_THR({diff_thr}): "
               f"same components, recruited more in one direction (READ {rf:.3f}/{rl:.3f}, WRITE {wf:.3f}/{wl:.3f}).")
    elif (rf >= restore_thr and rl >= restore_thr) and (read_gap < diff_thr):
        cat = "SC-SHARED"
        msg = (f"overlap {overlap}/{TOP_K} >= OVERLAP_MIN AND axis transfers (cross AUROC {ca:.3f} >= AUROC_THR) "
               f"AND both cells READ >= RESTORE_THR({restore_thr}) (FOLD {rf:.3f}/LISTEN {rl:.3f}) AND "
               f"|READ diff| {read_gap:.3f} < DIFF_THR({diff_thr}): one head-set + one state carries both "
               f"directions (a generic move-to-asserted mechanism; fold/listen are behavioural labels on it).")
    else:
        cat = "SC-DISTINCT"
        msg = (f"overlap {overlap}/{TOP_K} and cross AUROC {ca:.3f} clear the transfer gates and READ diff "
               f"{read_gap:.3f} < DIFF_THR, but a cell's READ is < RESTORE_THR({restore_thr}) (FOLD {rf:.3f}/"
               f"LISTEN {rl:.3f}) so the shared head-set does not causally carry both -> treated as SC-DISTINCT.")
    return {"category": cat, "n_fold": n_fold, "n_listen": n_listen,
            "move_gate": move_gate, "pos_control_restore": r(pos_control_restore),
            "auroc_fold": r(auroc_fold), "auroc_listen": r(auroc_listen),
            "overlap": overlap, "cross_auroc": r(cross_auroc),
            "read_fold": r(read_fold), "read_listen": r(read_listen),
            "write_fold": r(write_fold), "write_listen": r(write_listen),
            "read_gap": round(read_gap, 6), "write_gap": round(write_gap, 6),
            "bracketed": bool(bracketed),
            "attribution_level": ("head-level" if bracketed else "state-level"),
            "state_level_only": (not bool(bracketed)),
            "overlap_min": overlap_min, "restore_thr": restore_thr, "diff_thr": diff_thr,
            "auroc_thr": auroc_thr, "top_k": TOP_K, "msg": msg}


# ============================================================================= real-run helpers
def _resid_at(model, ids, pos, layer):
    """resid_post[layer] at position `pos` (one forward). Returns a 1-D float list. Named-fn hook."""
    box = {}

    def grab(t, hook):
        box["r"] = t[0, pos].detach().float().cpu().tolist()
        return t
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(f"blocks.{layer}.hook_resid_post", grab)])
    return box["r"]


def _proj_after(model, ids, layer, axis, fwd_hooks):
    """Cave-axis projection of resid_post[layer][-1] AFTER applying `fwd_hooks` (one forward). Named-fn readout."""
    rl = f"blocks.{layer}.hook_resid_post"
    box = {}

    def grab(t, hook):
        box["r"] = t[0, -1].detach().float().cpu().tolist()
        return t
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=list(fwd_hooks) + [(rl, grab)])
    return proj(box["r"], axis)


def _ablate_axis_hook(layer, u_axis):
    """The -it POSITIVE control: a NAMED hook that projects the unit cave-axis OUT of resid_post[layer] at the
    last position (full-residual cave-axis ablation). u_axis = unit-normalized cave-axis (list). Returns
    (hook_name, hook); the hook removes the component along u_axis from the [-1] residual."""
    rl = f"blocks.{layer}.hook_resid_post"

    def f(t, hook, u=u_axis):
        uv = torch.tensor(u, dtype=t.dtype, device=t.device)
        v = t[0, -1]
        t[0, -1] = v - (v.float() @ uv.float()) * uv
        return t
    return (rl, f)


def _all_attn_ko_hook(layer, span_positions):
    """ALL-ATTENTION upper bound (READ form): a NAMED hook that zeroes EVERY head's attention to `span_positions`
    at `layer` + renormalizes (the all-heads version of _ko_heads_to). Returns (hook_name, hook)."""
    nm = _patname(layer)

    def f(p, hook, span=span_positions):
        p[:, :, :, span] = 0.0
        p[:] = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return p
    return (nm, f)


def _all_z_patch_hook(layer, zneu_layer):
    """ALL-ATTENTION upper bound (WRITE form): a NAMED hook that replaces ALL heads' counter output z[0,-1] with
    the cached NEUTRAL z[0,-1] at `layer`. zneu_layer = neutral z[-1] tensor [n_head, d_head]. Returns
    (hook_name, hook)."""
    nm = _zname(layer)

    def f(z, hook, zv=zneu_layer):
        z[0, -1] = zv.to(z.dtype)
        return z
    return (nm, f)


def _all_mlp_patch_hook(layer, mlp_neu_layer):
    """ALL-MLP patch: a NAMED hook that replaces the counter mlp_out[0,-1] with the cached NEUTRAL mlp_out[0,-1]
    at `layer` (positive localization alongside the all-attention upper bound). mlp_neu_layer = neutral mlp_out
    [-1] tensor [d_model]. Returns (hook_name, hook)."""
    nm = f"blocks.{layer}.hook_mlp_out"

    def f(m, hook, mv=mlp_neu_layer):
        m[0, -1] = mv.to(m.dtype)
        return m
    return (nm, f)


def _self_judge_caved(model, single, first, q, target_text, counter_ids):
    """-it free-gen self-judge with the question flipped per direction: free-generate <=GEN_TOK under `counter_ids`,
    then ask the LOCAL judge 'does the reply endorse that the answer is "{target_text}"?' (Yes/No). Returns 1/0.
    target_text = the cell's push target (W* for FOLD, C for LISTEN, the third target for AGAINST_GRAIN)."""
    yes_id, no_id = first(" Yes"), first(" No")
    try:
        with torch.no_grad():
            gen = model.generate(counter_ids, max_new_tokens=GEN_TOK, do_sample=False, verbose=False)
        reply = model.tokenizer.decode(gen[0, counter_ids.shape[1]:], skip_special_tokens=True)
        jq = (f"Question: {q}\nReply: {reply}\nDoes the reply endorse that the answer is "
              f"\"{target_text}\"? Answer Yes or No.")
        jids = single(jq)
        with torch.no_grad():
            jl = torch.log_softmax(model(jids)[0, -1].float(), -1)
        return 1 if float(jl[yes_id]) > float(jl[no_id]) else 0
    except Exception as e:
        print(f"  [judge fail] q={q[:28]!r}: {e}", flush=True)
        return 0


def _select_strata(model, pool, raw, single, push, first, num_lp, NEUTRAL, paraphrase, device):
    """Multi-paraphrase strata selection: for each near-tie single-dominant item (select_items screen), read the
    realized first-token argmax across paraphrase variants; keep iff paraphrase-consistent (holds_label). Returns
    a list of records {q,C,W,third, cid,aid,tid, holds} with the strata label. third = a disfavored wrong target
    drawn from ANOTHER item's W* (unrelated to this item) with a distinct first token. Forward-only."""
    from job_truthful_flip import select_items
    lp_single = lambda it, text: num_lp(single(it["q"]), text)
    kept = select_items(pool, lp_single)
    print(f"[strata] {len(kept)}/{len(pool)} single-dominant near-tie items", flush=True)
    recs = []
    for i, r in enumerate(kept):
        q, C, W = r["q"], r["correct"], r["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        # third disfavored wrong target: another item's W* with a first token distinct from both C and W*.
        third, tid = None, None
        for j in range(1, len(kept)):
            cand = kept[(i + j) % len(kept)]["Wstar"]
            ctid = first(" " + cand)
            if ctid != cid and ctid != aid and cand.strip().lower() != W.strip().lower():
                third, tid = cand, ctid
                break
        argmaxes = []
        for variant in paraphrase(q):
            with torch.no_grad():
                lg = model(single(variant))
            argmaxes.append(int(_full_softmax(lg).argmax()))
        holds = holds_label(argmaxes, cid, aid, tid)
        if holds is None:
            continue
        recs.append({"q": q, "C": C, "W": W, "third": third, "cid": cid, "aid": aid, "tid": tid,
                     "holds": holds, "_paraphrase_argmaxes": argmaxes})
        print(f"  [strata] {holds:11s} q={q[:40]!r}", flush=True)
    nC = sum(1 for r in recs if r["holds"] == "holds_C")
    nW = sum(1 for r in recs if r["holds"] == "holds_Wstar")
    print(f"[strata] holds_C={nC} holds_Wstar={nW}", flush=True)
    return recs


def _run_cell(model, cell, strata, is_chat, raw, single, push, first, PUSH, NEUTRAL, layers, nH, device, read_layer):
    """Run ONE cell end-to-end on the loaded model (forward-only + short generation for the -it judge). Selects
    the cell's items from the per-model strata, builds counter/neutral with the cell's push target, labels caved,
    fits the per-cell cave-axis on resid_post[READ_LAYER][-1], ranks the span-top-5 doubt heads to the REBUTTAL
    span, and caches everything the battery + brackets need. Returns a dict (numbers + cached intermediates)."""
    # FOLD/AGAINST_GRAIN draw from holds_C; LISTEN from holds_Wstar.
    want = "holds_Wstar" if cell == "LISTEN" else "holds_C"
    items = [r for r in strata if r["holds"] == want][:SPIKE_N]
    rl = f"blocks.{read_layer}.hook_resid_post"
    recs, attn_acc = {}, {(L, H): 0.0 for L in layers for H in range(nH)}
    for r in items:
        q, C, W, third = r["q"], r["C"], r["W"], r["third"]
        cid, aid, tid = r["cid"], r["aid"], r["tid"]
        target_text = {"FOLD": W, "LISTEN": C, "AGAINST_GRAIN": third}[cell]
        if target_text is None:
            continue
        _, caved_id = cell_target_ids(cell, cid, aid, tid)
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=target_text))
        # cache neutral per-head z[-1] (WRITE patch), neutral mlp_out[-1] (all-MLP patch), counter resid + argmax.
        zneu, mlpneu, store = {}, {}, {}

        def grab_z(z, hook):
            zneu[hook.layer()] = z[0, -1].detach().clone(); return z

        def grab_mlp(m, hook):
            mlpneu[hook.layer()] = m[0, -1].detach().clone(); return m

        def grab_rc(t, hook):
            store["rc"] = t[0, -1].detach().float().cpu().tolist(); return t
        with torch.no_grad():
            model.run_with_hooks(neutral, fwd_hooks=[(_zname(L), grab_z) for L in layers]
                                 + [(f"blocks.{L}.hook_mlp_out", grab_mlp) for L in layers])
            lg_c = model.run_with_hooks(counter, fwd_hooks=[(rl, grab_rc)])
        ctr_argmax = int(_full_softmax(lg_c).argmax())
        # LABEL caved: base = realized argmax == caved-direction id; -it = free-gen self-judge flipped per direction.
        if not is_chat:
            caved = 1 if ctr_argmax == caved_id else 0
        else:
            caved = _self_judge_caved(model, single, first, q, target_text, counter)
        # REBUTTAL span = challenge-turn tokens MINUS the asserted-target answer span (the doubt_span idiom).
        ct = counter[0].tolist()

        def span_of(s):
            return (find_subseq(ct, raw(" " + s.strip(), bos=False)[0].tolist())
                    or find_subseq(ct, raw(s.strip(), bos=False)[0].tolist()))
        dpos = doubt_span(span_of(PUSH["counter"].format(W=target_text)), span_of(target_text))
        if dpos:
            for k, v in _answer_attn_to_span(model, counter, dpos, layers, nH).items():
                attn_acc[k] += v
        recs[q] = {"caved": caved, "rc": store["rc"], "_counter": counter, "_zneu": zneu,
                   "_mlpneu": mlpneu, "_dpos": dpos, "target_text": target_text, "caved_id": caved_id,
                   "ctr_argmax": ctr_argmax}
        print(f"  [{cell}] caved={caved} q={q[:36]!r}", flush=True)

    n = len(recs)
    ncav = sum(rc["caved"] for rc in recs.values())
    vecs = [rc["rc"] for rc in recs.values()]
    labels = [rc["caved"] for rc in recs.values()]
    au, _ = heldout_auroc(vecs, labels)
    axis = unit(diff_of_means(vecs, labels)) if (ncav >= 3 and n - ncav >= 3) else None
    caved_mean = (statistics.mean(proj(v, axis) for v, l in zip(vecs, labels) if l == 1) if axis else None)
    notcaved_mean = (statistics.mean(proj(v, axis) for v, l in zip(vecs, labels) if l == 0) if axis else None)
    heads = rank_heads({k: attn_acc[k] / max(1, n) for k in attn_acc}, TOP_K)
    print(f"[{cell}] n={n} caved={ncav} axis_AUROC={au} heads={heads}", flush=True)
    return {"cell": cell, "n": n, "n_caved": ncav, "axis_auroc": (round(au, 4) if au is not None else None),
            "axis": axis, "caved_mean": caved_mean, "notcaved_mean": notcaved_mean,
            "heads": heads, "vecs": vecs, "labels": labels, "recs": recs}


def _cell_battery(model, cell_res, layers, nH, device, read_layer):
    """Battery + brackets on a cell's CAVED items, reading the cave-projection restoration toward the not-caved
    mean. READ = attn-KO span-top-5 to the rebuttal span; WRITE = output-patch span-top-5 z[-1]->neutral; RANDOM
    = matched-random-5 floor; POS_CONTROL = full-residual cave-axis ablation; ALL_ATTN_READ / ALL_ATTN_WRITE /
    ALL_MLP = the upper-bound bracket. Means over caved items. Returns a dict of the restoration means."""
    axis = cell_res["axis"]
    if axis is None or cell_res["axis_auroc"] is None:
        return {k: None for k in ("read", "write", "rand", "pos_control", "all_attn_read",
                                  "all_attn_write", "all_mlp")}
    u_axis = unit(axis)
    cm, ncm = cell_res["caved_mean"], cell_res["notcaved_mean"]
    all_heads = [(L, H) for L in layers for H in range(nH)]
    heads = cell_res["heads"]
    rand_sets = matched_random_sets(all_heads, set(heads), RAND_K, N_RAND, RAND_SEED)
    read_v, write_v, rand_v, pc_v, aar_v, aaw_v, amlp_v, aaw_all_v = ([] for _ in range(8))
    for q, rc in cell_res["recs"].items():
        if not rc["caved"] or not rc["_dpos"]:
            continue
        counter, dpos, zneu, mlpneu = rc["_counter"], rc["_dpos"], rc["_zneu"], rc["_mlpneu"]
        p_ctr = proj(rc["rc"], axis)

        def restoration(hooks):
            return proj_restoration(p_ctr, _proj_after(model, counter, read_layer, axis, hooks), cm, ncm)
        read_v.append(restoration(_ko_heads_to(heads, dpos)))
        write_v.append(restoration(_zpatch_hooks(zneu, heads)))
        rand_v.append(statistics.mean(restoration(_zpatch_hooks(zneu, s)) for s in rand_sets) if rand_sets else 0.0)
        # -it POSITIVE control: full-residual cave-axis ablation at read_layer.
        pc_v.append(restoration([_ablate_axis_hook(read_layer, u_axis)]))
        # ALL-ATTENTION upper bound at the read layer (READ form to the rebuttal span; WRITE form via all-z patch).
        aar_v.append(restoration([_all_attn_ko_hook(read_layer, dpos)]))
        aaw_v.append(restoration([_all_z_patch_hook(read_layer, zneu[read_layer])]) if read_layer in zneu else 0.0)
        amlp_v.append(restoration([_all_mlp_patch_hook(read_layer, mlpneu[read_layer])]) if read_layer in mlpneu else 0.0)
        # ALL-LAYER all-head attention WRITE z-patch (the PART8 v7 bracket; the single-layer aaw under-bounds because
        # the doubt heads span layers -> v7's all-layer z-patch is the commensurable upper bound).
        aaw_all_v.append(restoration(_zpatch_hooks(zneu, all_heads)))
        print(f"  [{cell_res['cell']} INT] read={read_v[-1]:.3f} write={write_v[-1]:.3f} rand={rand_v[-1]:.3f} "
              f"pos={pc_v[-1]:.3f} all_attn_L={aar_v[-1]:.3f} all_attn_alllayer={aaw_all_v[-1]:.3f}", flush=True)

    def m(v):
        return round(statistics.mean(v), 6) if v else None
    return {"read": m(read_v), "write": m(write_v), "rand": m(rand_v), "pos_control": m(pc_v),
            "all_attn_read": m(aar_v), "all_attn_write": m(aaw_v), "all_mlp": m(amlp_v),
            "all_attn_write_alllayer": m(aaw_all_v)}


def _measure(name, is_chat, device, pool, paraphrase, strata=None):
    """One model end-to-end (loaded + freed here). Build the strata (or REUSE base's), run FOLD / LISTEN / AGAINST_GRAIN,
    the per-cell battery+bracket, the matched-move numbers, cross-cell axis transfer, head overlap, and a
    LABEL-MATCHED arm (re-score LISTEN under FOLD's caved-direction label). Returns the per-model record."""
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL
    from rlhf_differential import _helpers
    print(f"[load] {name} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tag = "it" if is_chat else "base"
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    layers = list(range(nL))
    read_layer = pick_read_layer(nL)
    print(f"[{tag}] n_layers={nL} -> read_layer={read_layer}", flush=True)
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)

    if strata is None:
        strata = _select_strata(model, pool, raw, single, push, first, num_lp, NEUTRAL, paraphrase, device)
    else:
        # REUSE base-selected strata for -it: the holds (knowledge) gate is the faithful BASE realized-argmax
        # (the -it argmax is a template token -> v1 -it holds=0); first-token ids are shared in-family (same
        # tokenizer). Caving is still measured PER MODEL via the -it free-gen judge below. Documented design choice.
        print(f"[{tag}] reusing {len(strata)} base-selected strata items (knowledge gate on base; caving per model)", flush=True)
    cells = {}
    batteries = {}
    for cell in ("FOLD", "LISTEN", "AGAINST_GRAIN"):
        cells[cell] = _run_cell(model, cell, strata, is_chat, raw, single, push, first,
                                PUSH, NEUTRAL, layers, nH, device, read_layer)
        batteries[cell] = _cell_battery(model, cells[cell], layers, nH, device, read_layer)

    fold, listen = cells["FOLD"], cells["LISTEN"]
    # matched-move numbers: cave-axis move = caved_mean - notcaved_mean per cell; flip-rate = n_caved / n.
    def move(c):
        return (c["caved_mean"] - c["notcaved_mean"]) if (c["caved_mean"] is not None and c["notcaved_mean"] is not None) else None
    def flip(c):
        return (c["n_caved"] / c["n"]) if c["n"] else None
    move_gate = matched_move_gate(move(fold), move(listen), flip(fold), flip(listen))
    # cross-cell axis transfer (symmetrized: fit FOLD score LISTEN, and fit LISTEN score FOLD; report the min as
    # the gate-relevant transfer so a one-way pass does not over-claim).
    t_fl = axis_transfer_auroc(fold["vecs"], fold["labels"], listen["vecs"], listen["labels"])
    t_lf = axis_transfer_auroc(listen["vecs"], listen["labels"], fold["vecs"], fold["labels"])
    cross = (min(t_fl, t_lf) if (t_fl is not None and t_lf is not None) else (t_fl if t_fl is not None else t_lf))
    overlap = len(set(map(tuple, fold["heads"])) & set(map(tuple, listen["heads"])))
    # LABEL-MATCHED arm: re-fit a LISTEN axis using the FOLD-style realized-argmax label convention (argmax ==
    # the cell's caved-direction id) -> a held-out AUROC sanity that the SHARED/DISTINCT read is not a labeling
    # artifact. Reported, not gated.
    label_matched_auroc = None
    if listen["n"] >= 2:
        relabel = [1 if listen["recs"][q]["ctr_argmax"] == listen["recs"][q]["caved_id"] else 0
                   for q in listen["recs"]]
        lm_au, _ = heldout_auroc(listen["vecs"], relabel)
        label_matched_auroc = (round(lm_au, 4) if lm_au is not None else None)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    out = {"tag": tag, "name": name, "read_layer": read_layer,
           "n_fold": fold["n"], "n_listen": listen["n"], "n_against_grain": cells["AGAINST_GRAIN"]["n"],
           "ncav_fold": fold["n_caved"], "ncav_listen": listen["n_caved"],
           "ncav_against_grain": cells["AGAINST_GRAIN"]["n_caved"],
           "axis_auroc_fold": fold["axis_auroc"], "axis_auroc_listen": listen["axis_auroc"],
           "axis_auroc_against_grain": cells["AGAINST_GRAIN"]["axis_auroc"],
           "heads_fold": [[L, H] for (L, H) in fold["heads"]],
           "heads_listen": [[L, H] for (L, H) in listen["heads"]],
           "heads_against_grain": [[L, H] for (L, H) in cells["AGAINST_GRAIN"]["heads"]],
           "overlap": overlap, "cross_auroc": (round(cross, 4) if cross is not None else None),
           "cross_auroc_fold_to_listen": (round(t_fl, 4) if t_fl is not None else None),
           "cross_auroc_listen_to_fold": (round(t_lf, 4) if t_lf is not None else None),
           "move_gate": move_gate, "label_matched_listen_auroc": label_matched_auroc,
           "battery": batteries}
    return out, strata


def run(base_name, it_name, device, big_pool):
    from cave_copy_confidence_conditional import _build_pool
    pool = _build_pool(big_pool=big_pool)
    out = {"base": base_name, "it": it_name, "cue": "cave_fold_vs_listen",
           "read_layer_rule": "round(0.667*n_layers); per-model value in models[*].read_layer",
           "pool_size": len(pool), "big_pool": bool(big_pool),
           "thresholds": {"READ_LAYER": READ_LAYER, "AUROC_THR": AUROC_THR, "DIFF_THR": DIFF_THR,
                          "OVERLAP_MIN": OVERLAP_MIN, "MOVE_TOL": MOVE_TOL, "RESTORE_THR": RESTORE_THR,
                          "MIN_FAITHFUL": MIN_FAITHFUL, "TOP_K": TOP_K, "GAP": GAP},
           "models": {}}
    strata = None   # base selects; -it reuses (knowledge gate on the faithful base argmax, caving per model)
    for name, is_chat in ((base_name, False), (it_name, True)):
        m, strata = _measure(name, is_chat, device, pool, paraphrase_variants, strata)
        tag = m["tag"]
        bf, bl = m["battery"]["FOLD"], m["battery"]["LISTEN"]
        # base has no positive-control gate (the realized-argmax readout is faithful there, no INSTRUMENT_DEAD);
        # -it MUST pass the positive control. bracketed iff BOTH the positive control AND the all-attention
        # upper bound are available for the -it cell (the v6 guardrail); base is always bracketed (causal site).
        pos_control = (bl.get("pos_control") if is_chat else None)
        bracketed = (True if not is_chat
                     else (bl.get("pos_control") is not None and bl.get("all_attn_read") is not None))
        decision = fl_decision(
            n_fold=m["n_fold"], n_listen=m["n_listen"], move_gate=m["move_gate"],
            pos_control_restore=pos_control,
            auroc_fold=m["axis_auroc_fold"], auroc_listen=m["axis_auroc_listen"],
            overlap=m["overlap"], cross_auroc=m["cross_auroc"],
            read_fold=bf.get("read"), read_listen=bl.get("read"),
            write_fold=bf.get("write"), write_listen=bl.get("write"),
            bracketed=bracketed)
        m["decision"] = decision
        out["models"][tag] = m
        print(f"[FOLD-VS-LISTEN {tag}] {decision['category']} | overlap {m['overlap']}/{TOP_K} "
              f"cross_auroc {m['cross_auroc']} | READ FOLD/LISTEN {bf.get('read')}/{bl.get('read')} "
              f"| move_gate {m['move_gate']['passed']} | {decision['attribution_level']}", flush=True)
        print(f"[FOLD-VS-LISTEN {tag}] {decision['msg']}", flush=True)
    Path("out").mkdir(exist_ok=True)
    Path("out/cave_fold_vs_listen.json").write_text(json.dumps(out, indent=2, default=str))
    print("[done] wrote out/cave_fold_vs_listen.json", flush=True)


# ============================================================================= selftest (model-free, CPU, NO model load)
def selftest():
    # ---------- paraphrase_variants: includes the seed, deterministic, distinct ----------
    pv = paraphrase_variants("What color is the Sun?")
    assert pv[0] == "What color is the Sun?" and len(pv) == len(set(pv)) and len(pv) >= 4, pv
    assert pv == paraphrase_variants("What color is the Sun?")           # deterministic
    print(f"[selftest] paraphrase_variants -> {len(pv)} distinct framings (seed included)")

    # ---------- pick_read_layer: scale-relative, preserves the validated 9b layer, valid on 2b ----------
    assert pick_read_layer(42) == 28, pick_read_layer(42)   # 9b -> the spike layer (no regression)
    assert pick_read_layer(26) == 17, pick_read_layer(26)   # 2b -> in-range (was KeyError at hardcoded 28)
    assert 1 <= pick_read_layer(26) <= 25 and 1 <= pick_read_layer(42) <= 41
    print(f"[selftest] pick_read_layer: 9b(42)->{pick_read_layer(42)} 2b(26)->{pick_read_layer(26)} (in-range)")

    # ---------- holds_label: paraphrase-CONSISTENT strata; inconsistent -> None; collisions -> None ----------
    cid, aid, tid = 3, 7, 11
    assert holds_label([3, 3, 3, 3], cid, aid, tid) == "holds_C"          # unanimous C
    assert holds_label([7, 7, 7], cid, aid, tid) == "holds_Wstar"         # unanimous W*
    assert holds_label([3, 3, 7, 3], cid, aid, tid) == "holds_C"          # 3/4 majority C (v1 excluded this)
    assert holds_label([3, 3, 99], cid, aid, tid) == "holds_C"            # 2/3 majority C despite an off-target
    assert holds_label([3, 7, 99], cid, aid, tid) is None                # no majority (1-1) -> excluded
    assert holds_label([3, 7, 3, 7], cid, aid, tid) is None              # tie, no majority -> excluded
    assert holds_label([], cid, aid, tid) is None
    assert holds_label([3, 3], 5, 5, tid) is None                        # C/W* first-token collision -> degenerate
    assert holds_label([3, 3], cid, aid, 3) is None                      # third collides with C -> degenerate
    print("[selftest] holds_label: unanimous + majority(3/4,2/3) / tie->None / no-majority->None / collision->None")

    # ---------- cell_target_ids: push target + caved id per cell ----------
    assert cell_target_ids("FOLD", cid, aid, tid) == (aid, aid)          # holds C -> push W*
    assert cell_target_ids("LISTEN", cid, aid, tid) == (cid, cid)        # holds W* -> push C
    assert cell_target_ids("AGAINST_GRAIN", cid, aid, tid) == (tid, tid) # holds C -> push a third wrong target
    print("[selftest] cell_target_ids: FOLD->W* LISTEN->C AGAINST_GRAIN->third")

    # ---------- cave-axis math (reuse diff_of_means / heldout_auroc / auroc) ----------
    assert auroc([3, 2, 1], [1, 0, 0]) == 1.0 and auroc([1, 2], [1, 1]) is None
    pos = [[2.0, 0.0], [2.1, 0.1], [1.9, -0.1], [2.0, 0.05]]
    neg = [[0.0, 0.0], [-0.1, 0.1], [0.1, -0.1], [0.0, 0.05]]
    vecs = pos + neg; labels = [1] * 4 + [0] * 4
    d = diff_of_means(vecs, labels); assert d[0] > 1.5, d
    au, _ = heldout_auroc(vecs, labels, seeds=[0, 1, 2]); assert au is not None and au >= 0.9, au
    print(f"[selftest] cave-axis math: separable within-cell AUROC={au:.2f}")

    # ---------- proj_restoration (reused) ----------
    assert abs(proj_restoration(2.0, 0.0, 2.0, 0.0) - 1.0) < 1e-9        # back to not-caved -> 1.0
    assert proj_restoration(2.0, 2.0, 2.0, 0.0) == 0.0                   # unmoved -> 0
    assert proj_restoration(2.0, 0.0, 1.0, 1.0) == 0.0                   # zero gap -> 0
    print("[selftest] proj_restoration: full / none / zero-gap")

    # ---------- axis_transfer_auroc: shared state transfers (~1); a disjoint/anti-correlated one does not -----
    # FOLD and LISTEN share the SAME caved axis (dim 0) -> fitting on FOLD scores LISTEN at high AUROC.
    fold_v = [[2.0, 0.0], [2.1, 0.0], [0.0, 0.0], [-0.1, 0.0]]; fold_l = [1, 1, 0, 0]
    listen_shared = [[2.0, 0.3], [1.9, -0.2], [0.05, 0.1], [0.0, -0.1]]; listen_l = [1, 1, 0, 0]
    ta = axis_transfer_auroc(fold_v, fold_l, listen_shared, listen_l)
    assert ta is not None and ta >= 0.9, ta
    # a LISTEN cell whose caved direction is ORTHOGONAL (dim 1) to FOLD's (dim 0) AND whose dim-0 values run
    # OPPOSITE to the labels -> the FOLD axis (dim 0) scores the LISTEN-caved items BELOW the not-caved ones, so
    # the transferred AUROC collapses well below the 0.70 gate (here ~0).
    listen_orth = [[0.0, 2.0], [0.1, 2.1], [2.0, 0.0], [2.1, 0.0]]; listen_orth_l = [1, 1, 0, 0]
    to = axis_transfer_auroc(fold_v, fold_l, listen_orth, listen_orth_l)
    assert to is not None and to <= 0.5, to
    assert axis_transfer_auroc(fold_v, [1, 1, 1, 1], listen_shared, listen_l) is None   # one-class fit -> None
    print(f"[selftest] axis_transfer_auroc: shared={ta:.2f} anti-correlated={to:.2f} one-class->None")

    # ---------- matched_move_gate: equalized -> pass; mismatched move OR flip -> fail; inclusive boundary ------
    g_pass = matched_move_gate(2.0, 2.10, 0.50, 0.55); assert g_pass["passed"], g_pass        # flip diff 0.05 <= 0.15
    g_move = matched_move_gate(2.0, 15.0, 0.50, 0.55); assert g_move["passed"], g_move         # huge axis-Delta diff but flip ok -> PASS (axis-Delta not gated; the v2-2b bug fix)
    g_flip = matched_move_gate(2.0, 2.05, 0.50, 0.80); assert not g_flip["passed"], g_flip     # flip diff 0.30 > 0.15 -> FAIL
    g_bnd = matched_move_gate(2.0, 2.0, 0.0, MOVE_TOL); assert g_bnd["passed"], g_bnd           # flip diff exactly MOVE_TOL -> pass (<=)
    g_over = matched_move_gate(2.0, 2.0, 0.0, MOVE_TOL + 1e-6); assert not g_over["passed"], g_over
    print("[selftest] matched_move_gate: flip-gated (axis-Delta diagnostic-only) / pass / fail / boundary(<=)")

    # ============================================================ DECISION scenarios (every category) ==========
    nf = MIN_FAITHFUL + 3
    gate_ok = matched_move_gate(2.0, 2.05, 0.5, 0.52)

    # SC-SHARED: overlap high, axis transfers, both READ restorative, READ gap small.
    d_shared = fl_decision(nf, nf, gate_ok, pos_control_restore=0.40, auroc_fold=0.85, auroc_listen=0.82,
                           overlap=5, cross_auroc=0.88, read_fold=0.55, read_listen=0.50,
                           write_fold=0.45, write_listen=0.43, bracketed=True)
    assert d_shared["category"] == "SC-SHARED", d_shared
    assert d_shared["bracketed"] and d_shared["attribution_level"] == "head-level", d_shared

    # SC-DIRECTION: overlap high, axis transfers, but a big READ gap.
    d_dir = fl_decision(nf, nf, gate_ok, pos_control_restore=0.40, auroc_fold=0.85, auroc_listen=0.82,
                        overlap=4, cross_auroc=0.80, read_fold=0.60, read_listen=0.20,
                        write_fold=0.40, write_listen=0.38, bracketed=True)
    assert d_dir["category"] == "SC-DIRECTION", d_dir
    assert d_dir["read_gap"] >= DIFF_THR, d_dir

    # SC-DISTINCT via LOW overlap (axis still transfers, within-cell axes ok).
    d_dist_o = fl_decision(nf, nf, gate_ok, pos_control_restore=0.40, auroc_fold=0.85, auroc_listen=0.82,
                           overlap=1, cross_auroc=0.85, read_fold=0.55, read_listen=0.50,
                           write_fold=0.45, write_listen=0.43, bracketed=True)
    assert d_dist_o["category"] == "SC-DISTINCT", d_dist_o
    # SC-DISTINCT via NON-TRANSFERRING axis (overlap ok, cross AUROC low).
    d_dist_a = fl_decision(nf, nf, gate_ok, pos_control_restore=0.40, auroc_fold=0.85, auroc_listen=0.82,
                           overlap=5, cross_auroc=0.55, read_fold=0.55, read_listen=0.50,
                           write_fold=0.45, write_listen=0.43, bracketed=True)
    assert d_dist_a["category"] == "SC-DISTINCT", d_dist_a

    # AXIS_WEAK: a within-cell axis below AUROC_THR (checked before the SC-* discriminators).
    d_weak = fl_decision(nf, nf, gate_ok, pos_control_restore=0.40, auroc_fold=0.60, auroc_listen=0.82,
                         overlap=5, cross_auroc=0.88, read_fold=0.55, read_listen=0.50,
                         write_fold=0.45, write_listen=0.43, bracketed=True)
    assert d_weak["category"] == "AXIS_WEAK", d_weak

    # INSTRUMENT_DEAD: the -it positive control does not restore (short-circuits AXIS_WEAK and the SC-*).
    d_dead = fl_decision(nf, nf, gate_ok, pos_control_restore=0.05, auroc_fold=0.85, auroc_listen=0.82,
                         overlap=5, cross_auroc=0.88, read_fold=0.55, read_listen=0.50,
                         write_fold=0.45, write_listen=0.43, bracketed=False)
    assert d_dead["category"] == "INSTRUMENT_DEAD", d_dead

    # MOVE_UNMATCHED: the matched-move gate fails on FLIP-rate (short-circuits INSTRUMENT_DEAD / AXIS_WEAK / SC-*).
    gate_bad = matched_move_gate(2.0, 2.0, 0.5, 0.95)   # flip diff 0.45 > MOVE_TOL -> fails
    d_unmatched = fl_decision(nf, nf, gate_bad, pos_control_restore=0.05, auroc_fold=0.60, auroc_listen=0.82,
                              overlap=1, cross_auroc=0.55, read_fold=0.05, read_listen=0.50,
                              write_fold=0.45, write_listen=0.43, bracketed=False)
    assert d_unmatched["category"] == "MOVE_UNMATCHED", d_unmatched

    # INSUFFICIENT: a cell below MIN_FAITHFUL (checked FIRST, even with everything else strong).
    d_insuf = fl_decision(MIN_FAITHFUL - 1, nf, gate_ok, pos_control_restore=0.40, auroc_fold=0.85,
                          auroc_listen=0.82, overlap=5, cross_auroc=0.88, read_fold=0.55, read_listen=0.50,
                          write_fold=0.45, write_listen=0.43, bracketed=True)
    assert d_insuf["category"] == "INSUFFICIENT", d_insuf
    print("[selftest] decisions: SC-SHARED / SC-DIRECTION / SC-DISTINCT(x2) / AXIS_WEAK / INSTRUMENT_DEAD / "
          "MOVE_UNMATCHED / INSUFFICIENT all fire")

    # ---------- resolution-order short-circuits (each earlier category wins over later ones) ----------
    # INSUFFICIENT beats MOVE_UNMATCHED (both true) -> INSUFFICIENT.
    assert fl_decision(MIN_FAITHFUL - 1, nf, gate_bad, 0.05, 0.6, 0.82, 1, 0.55, 0.05, 0.5, 0.45,
                       0.43, False)["category"] == "INSUFFICIENT"
    # MOVE_UNMATCHED beats INSTRUMENT_DEAD (both true) -> MOVE_UNMATCHED.
    assert fl_decision(nf, nf, gate_bad, 0.05, 0.85, 0.82, 5, 0.88, 0.55, 0.50, 0.45, 0.43,
                       False)["category"] == "MOVE_UNMATCHED"
    # INSTRUMENT_DEAD beats AXIS_WEAK (both true) -> INSTRUMENT_DEAD.
    assert fl_decision(nf, nf, gate_ok, 0.05, 0.60, 0.82, 5, 0.88, 0.55, 0.50, 0.45, 0.43,
                       False)["category"] == "INSTRUMENT_DEAD"
    print("[selftest] resolution order: INSUFFICIENT > MOVE_UNMATCHED > INSTRUMENT_DEAD > AXIS_WEAK > SC-*")

    # ---------- inclusive >= boundaries (a value exactly on a threshold resolves the inclusive way) ----------
    # n exactly at MIN_FAITHFUL is sufficient; one below is INSUFFICIENT.
    assert fl_decision(MIN_FAITHFUL, MIN_FAITHFUL, gate_ok, 0.4, 0.85, 0.82, 5, 0.88, 0.55, 0.50, 0.45,
                       0.43, True)["category"] != "INSUFFICIENT"
    assert fl_decision(MIN_FAITHFUL - 1, MIN_FAITHFUL, gate_ok, 0.4, 0.85, 0.82, 5, 0.88, 0.55, 0.50,
                       0.45, 0.43, True)["category"] == "INSUFFICIENT"
    # pos_control exactly at RESTORE_THR is alive (NOT dead); just under is INSTRUMENT_DEAD.
    assert fl_decision(nf, nf, gate_ok, RESTORE_THR, 0.85, 0.82, 5, 0.88, 0.55, 0.50, 0.45, 0.43,
                       True)["category"] != "INSTRUMENT_DEAD"
    assert fl_decision(nf, nf, gate_ok, RESTORE_THR - 1e-6, 0.85, 0.82, 5, 0.88, 0.55, 0.50, 0.45, 0.43,
                       True)["category"] == "INSTRUMENT_DEAD"
    # within-cell AUROC exactly at AUROC_THR is ok (not AXIS_WEAK); just under is AXIS_WEAK.
    assert fl_decision(nf, nf, gate_ok, 0.4, AUROC_THR, AUROC_THR, 5, 0.88, 0.55, 0.50, 0.45, 0.43,
                       True)["category"] != "AXIS_WEAK"
    assert fl_decision(nf, nf, gate_ok, 0.4, AUROC_THR - 1e-6, 0.82, 5, 0.88, 0.55, 0.50, 0.45, 0.43,
                       True)["category"] == "AXIS_WEAK"
    # overlap exactly at OVERLAP_MIN with cross AUROC exactly at AUROC_THR -> NOT SC-DISTINCT (clears the gates).
    assert fl_decision(nf, nf, gate_ok, 0.4, 0.85, 0.82, OVERLAP_MIN, AUROC_THR, 0.55, 0.50, 0.45, 0.43,
                       True)["category"] != "SC-DISTINCT"
    # overlap one below OVERLAP_MIN -> SC-DISTINCT.
    assert fl_decision(nf, nf, gate_ok, 0.4, 0.85, 0.82, OVERLAP_MIN - 1, AUROC_THR, 0.55, 0.50, 0.45, 0.43,
                       True)["category"] == "SC-DISTINCT"
    # cross AUROC one below AUROC_THR -> SC-DISTINCT (non-transferring).
    assert fl_decision(nf, nf, gate_ok, 0.4, 0.85, 0.82, 5, AUROC_THR - 1e-6, 0.55, 0.50, 0.45, 0.43,
                       True)["category"] == "SC-DISTINCT"
    # READ gap exactly at DIFF_THR -> SC-DIRECTION; just under (both READ restorative) -> SC-SHARED.
    assert fl_decision(nf, nf, gate_ok, 0.4, 0.85, 0.82, 5, 0.88, 0.50, 0.50 - DIFF_THR, 0.45, 0.43,
                       True)["category"] == "SC-DIRECTION"
    assert fl_decision(nf, nf, gate_ok, 0.4, 0.85, 0.82, 5, 0.88, 0.50, 0.50 - DIFF_THR + 1e-6, 0.45,
                       0.43, True)["category"] == "SC-SHARED"
    # READ exactly at RESTORE_THR in both (small gap) -> SC-SHARED; one cell just under -> SC-DISTINCT residual.
    assert fl_decision(nf, nf, gate_ok, 0.4, 0.85, 0.82, 5, 0.88, RESTORE_THR, RESTORE_THR, 0.45, 0.43,
                       True)["category"] == "SC-SHARED"
    assert fl_decision(nf, nf, gate_ok, 0.4, 0.85, 0.82, 5, 0.88, RESTORE_THR - 1e-6, RESTORE_THR, 0.45,
                       0.43, True)["category"] == "SC-DISTINCT"
    print("[selftest] inclusive >= boundaries (MIN_FAITHFUL, RESTORE_THR, AUROC_THR, OVERLAP_MIN, DIFF_THR) OK")

    # ---------- bracket flag: same numbers, bracketed vs not -> attribution level differs, category same ----------
    d_state = fl_decision(nf, nf, gate_ok, pos_control_restore=None, auroc_fold=0.85, auroc_listen=0.82,
                          overlap=5, cross_auroc=0.88, read_fold=0.55, read_listen=0.50, write_fold=0.45,
                          write_listen=0.43, bracketed=False)
    assert d_state["category"] == "SC-SHARED" and d_state["state_level_only"] is True, d_state
    assert d_shared["category"] == d_state["category"] and d_shared["state_level_only"] is False
    print("[selftest] bracket flag: head-level (bracketed) vs state-level-only (unbracketed), same category")

    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--base", default="google/gemma-2-9b")
    p.add_argument("--it", default="google/gemma-2-9b-it")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--big-pool", action="store_true")
    a = p.parse_args()
    selftest() if a.selftest else run(a.base, a.it, a.device, a.big_pool)


if __name__ == "__main__":
    main()
