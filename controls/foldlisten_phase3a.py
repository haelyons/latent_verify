"""PHASE 3a (DESIGN_foldlisten_mechanism.md lines 198-225, first half) -- owed pre-Phase-3 instrument
patches (Part A) + read/write HANDLE DERIVATION (Part B). Content/realized readout throughout, 9b-it,
frozen 74-item mechanism family. This control FREEZES handles for Phase 3b; it must NOT evaluate any
cross-transport / one-lever / direct-total claim itself (those are 3b).

PART A -- owed instrument patches (handle-free; run on ALL 74 items):
  A1  Prefix-stability at EVERY generation stage (counter AND elicit): the challenge/neutral span
      [s0,s1) is recomputed by tokenizing the closed prefix with vs without the final user turn, and the
      shared prefix token ids + the span are asserted CONSTANT across stages (the elicit-stage context
      also contains the generated counter). On violation the (item,arm) is recorded SPAN_UNSTABLE and
      EXCLUDED from every accumulator (count printed). FULL prompt strings at every stage are stored.
  A2  Masked W*-stated NEUTRAL floor (the missing listen floor). neutral_wstar_mask: state W* as turn-2,
      neutral turn-3 ("Okay, thank you." -- copied from job_truthful_flip.NEUTRAL), mask the turn-3 span
      (all heads), generate counter+elicit; floor = rate of ending != W* (moved off the stated W*, i.e.
      interpret('listen',.)=='moved', abstentions excluded per the repo _rate convention). Also
      neutral_wstar_nomask for the unmasked drift reference. Decision listen_ko_reread compares the
      committed Phase-2 listen_mask rate (passed via --p2-summary) against this floor.
  A3  NEUTRAL-arm DLA baseline. Run the Phase-2 logit-lens attn/mlp-delta profiles (resid pre/mid/post,
      last prompt position) on neutral_mask and neutral_wstar_mask, using the SAME contrast pair as the
      corresponding pushed arm (neutral_mask -> fold's (W,C); neutral_wstar_mask -> listen's (C,W)) so
      profiles are comparable. Decision dla_baseline_verdict: top-5 |attn_delta| overlap of each neutral
      arm with the committed fold/listen top-5 (from --p2-summary).

PART B -- handle derivation (frozen BEFORE any cross-arm evaluation; no transport testing here):
  Item discipline: deterministic even/odd split of the 74 AFTER sorting by question string -> DERIVE
  (even, n=37) and EVAL (odd, n=37). 3a touches ONLY DERIVE for selection; EVAL items appear only via
  the handle-free Part A patches.
  B1  Read-side H_read_fold: on DERIVE fold_nomask, per-head attention MASS (hook_pattern) from the
      counter+elicit generated positions to the challenge-span keys; rank all (layer,head) by mean mass;
      top-10 candidates. Greedy forward selection (subset <= 6): at each step add the candidate whose
      head-subset mask (challenge-span key range, restricted to the subset's heads) most reduces the
      DERIVE fold commit-rate; stop when marginal drop < 0.03 or size 6. H_read_listen: same recipe on
      DERIVE listen_nomask (D-1 option i: listen ranked by its own attention mass, greedy on listen-rate).
  B2  Write-side H_write_fold: per-layer diff-of-means resid_post (fold_nomask minus neutral_nomask) at
      the elicit-slot last prompt position, over the Phase-2 attn band L28-37 on DERIVE items,
      unit-normalized per layer (10 per-layer directions). H_write_listen: listen_nomask minus
      neutral_wstar_nomask. Per-layer cosine(H_write_fold,H_write_listen) is REPORTED ONLY (the >=0.7
      decision belongs to 3b). Per-layer raw-diff and resid norm stats saved for 3b's norm-matched randoms.
  B3  Persist out/phase3_handles_<tag>.npz (per-layer directions + norms) and .json (subsets, rankings,
      greedy traces, derive/eval item lists, cosines, thresholds, decision_rule strings for 3b).

Reuse, never reimplement: commit_prog_v2 (word-boundary matcher, Addendum-4 fix-forward) from family_generate_judge; interpret/load_family from
foldlisten_judge; diff_of_means from think_probe_identity. The mask-hook / span / DLA machinery is copied
from controls/foldlisten_phase2.py (same hook names, same MASK_NEG), with A1 extending the prefix-stability
idiom to the elicit stage. Model-free --selftest (default) covers every pure-logic function. torch /
transformer_lens are imported ONLY in the --run path.

  python controls/foldlisten_phase3a.py --selftest
  python controls/foldlisten_phase3a.py --run --family mechanism_family_9bit.json --name google/gemma-2-9b-it \
      --p2-summary out/foldlisten_phase2_p2_9bit_summary.json --tag p3a_9bit --device cuda --chat
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# FLAT-scp: controls/ for sibling reuse, latent_verify/ for the repo imports (identical to the siblings).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from family_generate_judge import commit_prog_v2 as commit_prog  # word-boundary matcher (Addendum-4 fix-forward); commit-only readout
from foldlisten_judge import interpret, load_family      # pure cell-outcome mapping + family loader
from think_probe_identity import diff_of_means           # pure numpy diff-of-means (reused, not reimplemented)

# Pre-registered constants (neutral: stated on the measured numbers only). Generation caps copied from
# foldlisten_phase2.py so the readout slot is identical across phases.
MAX_NEW_TOKENS = 160
ELICIT_TOK = 24
ELICIT = "What is your final answer? Reply with only the answer."
MASK_NEG = -1e9                 # pre-softmax score for masked key positions (copied from phase2)

WRITE_BAND = tuple(range(28, 38))   # Phase-2 attn band L28-37 (10 layers); clipped to n_layers at run time
READ_TOPK = 10                  # top (layer,head) candidates ranked by attention mass
SUBSET_MAX = 6                  # greedy forward-selection subset cap
GREEDY_MIN_DROP = 0.03          # stop greedy when the best marginal commit-rate drop is below this
HANDLE_WEAK_DROP = 0.10         # read-side total drop below this -> WEAK_AT_DERIVE flag (still frozen)
MIN_BASE_RATE = 0.5             # DERIVE fold base rate below this -> handle_freeze UNDERPOWERED (phase2 idiom)
LISTEN_KO_MARGIN = 0.18         # listen_mask_rate - floor at/above this -> LISTEN_KO_CLEARS_FLOOR
DLA_OVERLAP_TOPK = 5            # top-k |attn_delta| layers compared neutral-arm vs committed pushed-arm
DLA_OVERLAP_MIN = 3             # overlap >= this -> GENERIC_ANSWER_FORMATION (phase2 OVERLAP_MIN idiom)
DLA_DISJOINT_MAX = 1            # overlap <= this -> DISCRIMINATES     (phase2 DISJOINT_MAX idiom)

NEUTRAL_ARMS = ("neutral_mask", "neutral_nomask", "neutral_wstar_mask", "neutral_wstar_nomask")
DERIVE_ARMS = ("fold_nomask", "listen_nomask")


# --------------------------------------------------------------------------- pure: span / stability (A1)
def challenge_span(len_without, len_with):
    """Key-position span [L0, L1) of the final user turn, from the closed-conversation token lengths
    without vs with that turn. Copied from foldlisten_phase2.py. Pure."""
    assert 0 < len_without < len_with, (len_without, len_with)
    return (int(len_without), int(len_with))


def assess_span_stability(stages):
    """A1 verdict over the per-stage prefix checks. `stages` = list of {"stage", "prefix_ok"(bool),
    "span"([s0,s1])}. STABLE iff every stage's shared prefix matched AND the span is constant across
    stages; else SPAN_UNSTABLE with a reason. Pure (list -> dict)."""
    spans = [tuple(s["span"]) for s in stages]
    prefix_ok = all(bool(s["prefix_ok"]) for s in stages)
    span_const = len(set(spans)) == 1
    stable = prefix_ok and span_const
    reason = None if stable else ("prefix_mismatch" if not prefix_ok else "span_changed")
    return {"stable": stable, "reason": reason, "spans": [list(sp) for sp in spans]}


# --------------------------------------------------------------------------- pure: DLA ranking (A3)
def topk_layers(profile, k=DLA_OVERLAP_TOPK):
    """Indices of the k largest |value| entries. Copied from foldlisten_phase2.py. Pure."""
    a = np.abs(np.asarray(profile, dtype=float))
    k = min(k, a.size)
    return sorted(int(i) for i in np.argsort(-a)[:k])


def dla_baseline_verdict(neutral_top, committed_top, ov_min=DLA_OVERLAP_MIN, dj_max=DLA_DISJOINT_MAX):
    """Neutral A3 category over the top-k |attn_delta| overlap of one NEUTRAL arm's profile with the
    committed pushed-arm top-k. INSUFFICIENT if either list is missing (no --p2-summary). Pure."""
    if neutral_top is None or committed_top is None:
        return {"category": "INSUFFICIENT", "n_overlap": None, "neutral_top": neutral_top,
                "committed_top": committed_top,
                "msg": "neutral or committed top-k unavailable (need --p2-summary); no comparison."}
    n_ov = len(set(int(x) for x in neutral_top) & set(int(x) for x in committed_top))
    if n_ov >= ov_min:
        cat = "GENERIC_ANSWER_FORMATION"
    elif n_ov <= dj_max:
        cat = "DISCRIMINATES"
    else:
        cat = "MIXED"
    return {"category": cat, "n_overlap": n_ov, "neutral_top": sorted(int(x) for x in neutral_top),
            "committed_top": sorted(int(x) for x in committed_top),
            "msg": (f"top-{DLA_OVERLAP_TOPK} |attn_delta| overlap neutral={sorted(neutral_top)} vs "
                    f"committed={sorted(committed_top)} = {n_ov} (GENERIC>= {ov_min}, DISCRIMINATES<= {dj_max}).")}


# --------------------------------------------------------------------------- pure: A2 listen-KO re-read
def listen_ko_reread(floor, listen_mask_rate, margin=LISTEN_KO_MARGIN):
    """Neutral A2 category. floor = neutral_wstar_mask rate of moving off W* (measured here);
    listen_mask_rate = committed Phase-2 listen KO mask rate (from --p2-summary). Pure.
      INSUFFICIENT              iff either input is None.
      LISTEN_KO_CLEARS_FLOOR    iff listen_mask_rate - floor >= margin (inclusive).
      LISTEN_KO_AT_FLOOR        otherwise."""
    if floor is None or listen_mask_rate is None:
        return {"category": "INSUFFICIENT", "floor": floor, "listen_mask_rate": listen_mask_rate,
                "delta": None, "msg": "floor or committed listen_mask_rate unavailable; no re-read."}
    delta = listen_mask_rate - floor
    cat = "LISTEN_KO_CLEARS_FLOOR" if delta >= margin else "LISTEN_KO_AT_FLOOR"
    return {"category": cat, "floor": floor, "listen_mask_rate": listen_mask_rate, "delta": delta,
            "msg": (f"listen_mask_rate {listen_mask_rate:.3f} - floor {floor:.3f} = {delta:.3f} "
                    f"{'>=' if delta >= margin else '<'} {margin}.")}


# --------------------------------------------------------------------------- pure: B1 head ranking + greedy
def rank_heads(mass):
    """(layer, head, mass) sorted by descending mass, ties broken by (layer, head). Pure. `mass` is a
    2D [n_layers, n_heads] array of mean attention mass to the challenge span."""
    mass = np.asarray(mass, dtype=float)
    nL, nH = mass.shape
    items = [(int(L), int(h), float(mass[L, h])) for L in range(nL) for h in range(nH)]
    items.sort(key=lambda t: (-t[2], t[0], t[1]))
    return items


def top_candidates(ranking, k=READ_TOPK):
    """First k (layer, head) tuples of a rank_heads ranking. Pure."""
    return [(L, h) for (L, h, _m) in ranking[:k]]


def greedy_select(base_rate, candidates, rate_of_subset, max_size=SUBSET_MAX, min_drop=GREEDY_MIN_DROP):
    """Greedy forward selection over head candidates. At each step evaluate current-subset+{cand} for
    every remaining candidate via `rate_of_subset(frozenset)`; add the candidate with the largest
    commit-rate DROP (smallest resulting rate; ties by candidate order). Stop when the best marginal drop
    < min_drop (that step recorded but NOT added) or the subset reaches max_size. Pure (rate_of_subset is
    an injected callable). Returns selected subset, per-step trace, final rate, total drop, stop reason."""
    order = {c: i for i, c in enumerate(candidates)}
    remaining = list(candidates)
    current, cur_rate, trace, reason = [], base_rate, [], "exhausted"
    while remaining and len(current) < max_size:
        scored = [(cand, rate_of_subset(frozenset(current + [cand]))) for cand in remaining]
        best_cand, best_r = min(scored, key=lambda t: (1.0 if t[1] is None else t[1], order[t[0]]))
        marg = None if (best_r is None or cur_rate is None) else (cur_rate - best_r)
        step = {"considered": len(scored), "best": list(best_cand), "subset_rate": best_r,
                "marginal_drop": marg, "cur_rate_before": cur_rate}
        if marg is not None and marg >= min_drop:
            current.append(best_cand)
            remaining.remove(best_cand)
            cur_rate = best_r
            step["added"] = True
            trace.append(step)
            if len(current) >= max_size:
                reason = "max_size"
                break
        else:
            step["added"] = False
            trace.append(step)
            reason = "min_drop"
            break
    total_drop = None if (base_rate is None or cur_rate is None) else (base_rate - cur_rate)
    return {"selected": [list(c) for c in current], "final_rate": cur_rate, "base_rate": base_rate,
            "total_drop": total_drop, "stopped_reason": reason, "trace": trace}


# --------------------------------------------------------------------------- pure: item split (Part B)
def derive_eval_split(items):
    """Deterministic even/odd split of the family AFTER sorting by question string. Even sorted-index ->
    DERIVE, odd -> EVAL. Pure (list[dict] -> (derive_items, eval_items)). Ties in `q` are broken by the
    tuple (q, correct, Wstar) so the order is total and reproducible."""
    ordered = sorted(items, key=lambda it: (it["q"], it.get("correct", ""), it.get("Wstar", "")))
    derive = [it for i, it in enumerate(ordered) if i % 2 == 0]
    ev = [it for i, it in enumerate(ordered) if i % 2 == 1]
    return derive, ev


# --------------------------------------------------------------------------- pure: B2 write directions
def unit_normalize(v):
    """v / ||v|| (safe). Pure numpy."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def cosine(a, b):
    """Cosine similarity of two vectors. Pure numpy."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na and nb else 0.0


def fit_write_directions(pos_by_idx, neg_by_idx):
    """Per-band-layer diff-of-means write direction (pos minus neg, unit-normalized). pos_by_idx /
    neg_by_idx map item index -> (n_band, d) resid arrays; the intersection of item indices is used so
    the two arms are compared over the same items. Reuses think_probe_identity.diff_of_means. Pure numpy.
    Returns dirs (n_band, d), raw diff norms, per-layer mean resid norms (pos & neg), and the item list."""
    idxs = sorted(set(pos_by_idx) & set(neg_by_idx))
    if not idxs:
        return None
    pos = np.stack([np.asarray(pos_by_idx[i], dtype=float) for i in idxs])   # (n, nb, d)
    neg = np.stack([np.asarray(neg_by_idx[i], dtype=float) for i in idxs])
    nb = pos.shape[1]
    dirs, raw_norm, pos_norm, neg_norm = [], [], [], []
    for b in range(nb):
        Xp, Xn = pos[:, b, :], neg[:, b, :]
        X = np.concatenate([Xp, Xn], axis=0)
        y = np.array([1] * len(Xp) + [0] * len(Xn))
        d = diff_of_means(X, y)                # mean(pos) - mean(neg)
        raw_norm.append(float(np.linalg.norm(d)))
        dirs.append(unit_normalize(d))
        pos_norm.append(float(np.linalg.norm(Xp, axis=1).mean()))
        neg_norm.append(float(np.linalg.norm(Xn, axis=1).mean()))
    return {"dirs": np.asarray(dirs), "raw_norm": np.asarray(raw_norm),
            "pos_resid_norm": np.asarray(pos_norm), "neg_resid_norm": np.asarray(neg_norm),
            "n_items": len(idxs), "item_idx": idxs}


# --------------------------------------------------------------------------- pure: handle-freeze decision
def handle_freeze_decision(base_fold_rate, fold_total_drop, listen_total_drop,
                           n_fold_cand, n_listen_cand,
                           min_base=MIN_BASE_RATE, weak_drop=HANDLE_WEAK_DROP):
    """Neutral handle-freeze category. UNDERPOWERED iff the DERIVE fold base rate is missing/below
    min_base OR there are no fold candidates (cannot derive). Else FROZEN. A read side whose greedy total
    drop is missing or < weak_drop is flagged WEAK_AT_DERIVE (still frozen). Pure."""
    underpowered = (base_fold_rate is None or base_fold_rate < min_base or n_fold_cand == 0)
    cat = "UNDERPOWERED" if underpowered else "FROZEN"

    def flag(td):
        return "WEAK_AT_DERIVE" if (td is None or td < weak_drop) else "OK"

    return {"category": cat, "read_side_fold": flag(fold_total_drop),
            "read_side_listen": flag(listen_total_drop), "base_fold_rate": base_fold_rate,
            "fold_total_drop": fold_total_drop, "listen_total_drop": listen_total_drop,
            "n_fold_candidates": n_fold_cand, "n_listen_candidates": n_listen_cand,
            "thresholds": {"MIN_BASE_RATE": min_base, "HANDLE_WEAK_DROP": weak_drop},
            "msg": (f"base_fold_rate={base_fold_rate} (min {min_base}); fold_drop={fold_total_drop}, "
                    f"listen_drop={listen_total_drop} (weak<{weak_drop}); "
                    f"n_cand fold/listen={n_fold_cand}/{n_listen_cand}.")}


# --------------------------------------------------------------------------- pure: counts / rate (phase2)
def arm_counts(records, arm):
    """moved/held/abstain of one arm over the elicited readout, via interpret(cell, commit). Records carry
    cell in {'fold','listen'} (the W*-stated neutral arms score the LISTEN cell). Pure."""
    c = {"moved": 0, "held": 0, "abstain": 0}
    for r in records:
        if r["arm"] != arm:
            continue
        c[interpret(r["cell"], r["commit_elicit"])] += 1
    return c


def _rate(c):
    d = c["moved"] + c["held"]
    return (c["moved"] / d) if d else None


# --------------------------------------------------------------------------- p2 committed baseline reader
def read_p2(path):
    """Extract the committed Phase-2 baselines this control cites (never recomputed). Returns None if no
    path is given; missing keys read as None so the downstream decisions fall to INSUFFICIENT."""
    if not path:
        return None
    d = json.loads(Path(path).read_text())
    rates = d.get("arm_rates", {})
    ov = d.get("overlap_precheck", {})
    return {"source": str(path), "listen_mask_rate": rates.get("listen_mask"),
            "neutral_mask_rate": rates.get("neutral_mask"),
            "top_fold": ov.get("top_fold"), "top_listen": ov.get("top_listen")}


# --------------------------------------------------------------------------- run (torch / TL ONLY here)
def run(family, name, tag, device, is_chat, n, p2_summary, wb_lo=None, wb_hi=None):
    import torch
    from transformer_lens import HookedTransformer
    from job_truthful_flip import PUSH, NEUTRAL
    from rlhf_differential import _helpers

    assert is_chat, "Phase 3a is registered on the -it substrate (C5); run with --chat"
    items = load_family(family)
    if n:
        items = items[:n]
    derive_items, eval_items = derive_eval_split(items)
    derive_qset = {it["q"] for it in derive_items}
    p2 = read_p2(p2_summary)
    print(f"[load] {name} on {device} (chat=True); family {family} -> {len(items)} items "
          f"(DERIVE {len(derive_items)} / EVAL {len(eval_items)}); p2={'set' if p2 else 'MISSING'}", flush=True)

    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    tok = model.tokenizer
    nL, nH, d_model = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model
    # Scale-transport: default = 9b's frozen WRITE_BAND L28-37 (UNCHANGED at 9b). For 2b/27b pass an
    # explicit band via --write-band-lo/--write-band-hi = the relative-depth analogue of L28-37 (fracs
    # ~[0.667, 0.905] of n_layers: 2b nL=26 -> L17-23, 27b nL=46 -> L31-41). The fixed L28-37 does NOT
    # transport: it is EMPTY at 2b (all >= nL=26) and mid-stack (not the late third) at 27b.
    write_band = tuple(range(wb_lo, wb_hi)) if (wb_lo is not None and wb_hi is not None) else WRITE_BAND
    band = [L for L in write_band if L < nL]
    print(f"[band] write_band={list(write_band)} -> clipped-to-nL={band} (nL={nL})", flush=True)
    assert band, f"write band empty at nL={nL}: pass --write-band-lo/--write-band-hi for this scale"
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    W_U = model.W_U

    def chat_ids(msgs, gen_prompt):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=gen_prompt, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        return ids.to(device)

    def ptext(ids):
        return tok.decode(ids[0], skip_special_tokens=False)

    def elicit_ids_of(q, stated, final_user, prior_gen):
        pg = prior_gen.strip() or "(no answer)"
        return chat_ids([{"role": "user", "content": q}, {"role": "assistant", "content": f"{stated}."},
                         {"role": "user", "content": final_user}, {"role": "assistant", "content": pg},
                         {"role": "user", "content": ELICIT}], gen_prompt=True)

    def span_and_stages(q, stated, final_user, eids):
        """Counter-stage span + A1 prefix checks at counter AND elicit stages (elicit context includes the
        generated counter). Returns (span, stages) with span None if the closed lengths are degenerate."""
        ids_qs = chat_ids([{"role": "user", "content": q}, {"role": "assistant", "content": f"{stated}."}],
                          gen_prompt=False)
        ids_qsc = chat_ids([{"role": "user", "content": q}, {"role": "assistant", "content": f"{stated}."},
                            {"role": "user", "content": final_user}], gen_prompt=False)
        la, lb = ids_qs.shape[1], ids_qsc.shape[1]
        if not (0 < la < lb):
            return None, [{"stage": "lengths", "prefix_ok": False, "span": [int(la), int(lb)]}]
        span = challenge_span(la, lb)
        counter_ok = bool(torch.equal(ids_qs[0], ids_qsc[0, :la]))
        elicit_ok = bool(eids.shape[1] >= lb and torch.equal(ids_qsc[0], eids[0, :lb]))
        return span, [{"stage": "counter", "prefix_ok": counter_ok, "span": list(span)},
                      {"stage": "elicit", "prefix_ok": elicit_ok, "span": list(span)}]

    def all_mask_hooks(span):
        """All-head mask of the challenge/neutral key span (copied from foldlisten_phase2.py)."""
        s0, s1 = span

        def f(scores, hook):
            if scores.shape[-1] > s0:
                scores[..., s0:min(s1, scores.shape[-1])] = MASK_NEG
            return scores
        return [(f"blocks.{L}.attn.hook_attn_scores", f) for L in range(nL)]

    def subset_mask_hooks(span, heads_by_layer):
        """Head-SUBSET mask of the challenge-span keys: only the listed heads at each listed layer (B1
        greedy). Same hook name / MASK_NEG / key range as the all-head mask, restricted on the head axis."""
        s0, s1 = span

        def make(hs):
            def f(scores, hook):
                k1 = min(s1, scores.shape[-1])
                if scores.shape[-1] > s0 and k1 > s0:
                    for h in hs:
                        if h < scores.shape[1]:
                            scores[:, h, :, s0:k1] = MASK_NEG
                return scores
            return f
        return [(f"blocks.{L}.attn.hook_attn_scores", make(list(hs))) for L, hs in heads_by_layer.items()]

    def generate(prompt_ids, n_new, hooks=None):
        with torch.no_grad():
            if hooks:
                with model.hooks(fwd_hooks=hooks):
                    g = model.generate(prompt_ids, max_new_tokens=n_new, do_sample=False,
                                       stop_at_eos=True, verbose=False)
            else:
                g = model.generate(prompt_ids, max_new_tokens=n_new, do_sample=False,
                                   stop_at_eos=True, verbose=False)
        return tok.decode(g[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip(), g

    def lens_margin(resid_1d, tok_pushed, tok_stated):
        h = model.ln_final(resid_1d.unsqueeze(0).unsqueeze(0))[0, 0]
        return float((h @ (W_U[:, tok_pushed] - W_U[:, tok_stated])).float())

    def dla_profiles(eids, pushed, stated):
        """Phase-2 logit-lens attn/mlp-delta profiles (resid pre/mid/post at last prompt position),
        contrast (pushed, stated). Copied from foldlisten_phase2.py."""
        t_p = first(" " + pushed.strip())
        t_s = first(" " + stated.strip())
        names = [f"blocks.{L}.hook_resid_{w}" for L in range(nL) for w in ("pre", "mid", "post")]
        with torch.no_grad():
            _, cache = model.run_with_cache(eids, names_filter=lambda x: x in set(names))
        attn_d, mlp_d = [], []
        for L in range(nL):
            m_pre = lens_margin(cache[f"blocks.{L}.hook_resid_pre"][0, -1], t_p, t_s)
            m_mid = lens_margin(cache[f"blocks.{L}.hook_resid_mid"][0, -1], t_p, t_s)
            m_post = lens_margin(cache[f"blocks.{L}.hook_resid_post"][0, -1], t_p, t_s)
            attn_d.append(m_mid - m_pre)
            mlp_d.append(m_post - m_mid)
        del cache
        return attn_d, mlp_d

    def capture_resid(eids):
        """resid_post at the elicit-slot LAST prompt position, band layers only. (n_band, d) numpy."""
        store = {}

        def make(L):
            def f(resid, hook):
                store[L] = resid[0, -1].detach().float().cpu().numpy()
                return resid
            return f
        with torch.no_grad():
            model.run_with_hooks(eids, fwd_hooks=[(f"blocks.{L}.hook_resid_post", make(L)) for L in band])
        return np.stack([store[L] for L in band], axis=0)

    def add_mass(full_ids, prompt_len, span, acc):
        """Accumulate per-head attention MASS (hook_pattern) summed over the challenge-span keys and over
        the GENERATED query positions [prompt_len:], into acc['sum'] (nL,nH) with acc['nq'] the query-pos
        count (pooled across the counter + elicit stages)."""
        s0, s1 = span
        tmp = np.zeros((nL, nH), dtype=float)
        holder = {"nq": 0}

        def make(L):
            def f(pattern, hook):
                k1 = min(s1, pattern.shape[-1])
                if pattern.shape[-2] > prompt_len and k1 > s0:
                    sub = pattern[0, :, prompt_len:, s0:k1]           # [nH, nq, span]
                    tmp[L] += sub.sum(-1).sum(-1).detach().float().cpu().numpy()
                    holder["nq"] = int(sub.shape[1])
                return pattern
            return f
        with torch.no_grad():
            model.run_with_hooks(full_ids, fwd_hooks=[(f"blocks.{L}.attn.hook_pattern", make(L))
                                                      for L in range(nL)])
        acc["sum"] += tmp
        acc["nq"] += holder["nq"]

    # ---------------------------------------------------------------- main measurement loop
    records = []
    n_span_unstable = 0
    span_unstable_log = []
    neutral_attn = {"neutral_mask": [], "neutral_wstar_mask": []}   # A3 profiles
    neutral_mlp = {"neutral_mask": [], "neutral_wstar_mask": []}
    mass_acc = {"fold": {"sum": np.zeros((nL, nH)), "nq": 0},
                "listen": {"sum": np.zeros((nL, nH)), "nq": 0}}      # B1
    write_resid = {"fold": {}, "listen": {}, "neutral": {}, "neutral_wstar": {}}  # B2 (item idx -> (nb,d))
    derive_specs = {"fold": [], "listen": []}                       # B1 greedy replay specs

    # Part A neutral arms: (arm, stated_key, contrast_pushed_key, masked?, dla_key, cell)
    neutral_plan = [
        ("neutral_mask",         "C", "W", True,  "neutral_mask",        "fold"),
        ("neutral_nomask",       "C", "W", False, None,                  "fold"),
        ("neutral_wstar_mask",   "W", "C", True,  "neutral_wstar_mask",  "listen"),
        ("neutral_wstar_nomask", "W", "C", False, None,                  "listen"),
    ]

    for idx, it in enumerate(items):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        in_derive = q in derive_qset

        # ---- Part A: neutral arms on ALL 74 ----
        for arm, s_key, p_key, masked, dla_key, cell in neutral_plan:
            stated = C if s_key == "C" else W
            pushed = W if p_key == "W" else C
            cids = push(q, stated, NEUTRAL)
            span, _stages_pre = span_and_stages(q, stated, NEUTRAL, cids)   # counter-stage prefix (eids later)
            if span is None:
                n_span_unstable += 1
                span_unstable_log.append({"item": idx, "arm": arm, "reason": "degenerate_lengths"})
                continue
            hooks = all_mask_hooks(span) if masked else None
            cg, _ = generate(cids, MAX_NEW_TOKENS, hooks)
            eids = elicit_ids_of(q, stated, NEUTRAL, cg)
            _, stages = span_and_stages(q, stated, NEUTRAL, eids)
            stab = assess_span_stability(stages)
            eg, _ = generate(eids, ELICIT_TOK, hooks)
            ce = commit_prog(eg, C, W)
            rec = {"item": idx, "q": q, "cell": cell, "arm": arm, "stated": stated, "pushed": pushed,
                   "in_derive": in_derive, "span": list(span), "span_stable": stab["stable"],
                   "span_stages": stages, "counter_prompt": ptext(cids), "counter_gen": cg,
                   "elicit_prompt": ptext(eids), "elicit_gen": eg, "commit_elicit": ce}
            if not stab["stable"]:
                n_span_unstable += 1
                span_unstable_log.append({"item": idx, "arm": arm, "reason": stab["reason"]})
                rec["excluded"] = True
                records.append(rec)
                continue
            records.append(rec)
            if masked:  # A3 DLA baseline, contrast pair matched to the corresponding pushed arm
                a, m = dla_profiles(eids, pushed, stated)
                neutral_attn[dla_key].append(a)
                neutral_mlp[dla_key].append(m)
            elif in_derive:  # B2 write baseline (unmasked neutral resid at the elicit slot)
                key = "neutral" if cell == "fold" else "neutral_wstar"
                write_resid[key][idx] = capture_resid(eids)

        # ---- Part B: fold/listen derivation arms on DERIVE only ----
        if in_derive:
            for cell, stated, pushed in (("fold", C, W), ("listen", W, C)):
                challenge = PUSH["counter"].format(W=pushed)
                cids = push(q, stated, challenge)
                span, _ = span_and_stages(q, stated, challenge, cids)
                if span is None:
                    n_span_unstable += 1
                    span_unstable_log.append({"item": idx, "arm": f"{cell}_nomask",
                                              "reason": "degenerate_lengths"})
                    continue
                cg, cg_ids = generate(cids, MAX_NEW_TOKENS, None)      # nomask
                eids = elicit_ids_of(q, stated, challenge, cg)
                _, stages = span_and_stages(q, stated, challenge, eids)
                stab = assess_span_stability(stages)
                eg, eg_ids = generate(eids, ELICIT_TOK, None)
                ce = commit_prog(eg, C, W)
                rec = {"item": idx, "q": q, "cell": cell, "arm": f"{cell}_nomask", "stated": stated,
                       "pushed": pushed, "in_derive": True, "span": list(span),
                       "span_stable": stab["stable"], "span_stages": stages,
                       "counter_prompt": ptext(cids), "counter_gen": cg, "elicit_prompt": ptext(eids),
                       "elicit_gen": eg, "commit_elicit": ce}
                if not stab["stable"]:
                    n_span_unstable += 1
                    span_unstable_log.append({"item": idx, "arm": f"{cell}_nomask",
                                              "reason": stab["reason"]})
                    rec["excluded"] = True
                    records.append(rec)
                    continue
                records.append(rec)
                # B1 read mass: pool counter-stage + elicit-stage generated query positions
                add_mass(cg_ids, cids.shape[1], span, mass_acc[cell])
                add_mass(eg_ids, eids.shape[1], span, mass_acc[cell])
                # B2 write capture (pos arm)
                write_resid[cell][idx] = capture_resid(eids)
                # B1 greedy replay spec
                derive_specs[cell].append({"item": idx, "q": q, "stated": stated, "pushed": pushed,
                                           "challenge": challenge, "span": span, "C": C, "W": W})
        print(f"  [{idx:03d}] {'DERIVE' if in_derive else 'EVAL '} done "
              f"(unstable so far={n_span_unstable}) q={q[:34]!r}", flush=True)

    # ---------------------------------------------------------------- B1 rankings + greedy
    mass_fold = mass_acc["fold"]["sum"] / max(mass_acc["fold"]["nq"], 1)
    mass_listen = mass_acc["listen"]["sum"] / max(mass_acc["listen"]["nq"], 1)
    rank_fold = rank_heads(mass_fold)
    rank_listen = rank_heads(mass_listen)
    cand_fold = top_candidates(rank_fold, READ_TOPK)
    cand_listen = top_candidates(rank_listen, READ_TOPK)

    base_fold = _rate(arm_counts([r for r in records if r["span_stable"]], "fold_nomask"))
    base_listen = _rate(arm_counts([r for r in records if r["span_stable"]], "listen_nomask"))

    def make_rate_fn(specs, cell):
        memo = {}

        def rate_of_subset(subset):
            if subset in memo:
                return memo[subset]
            heads_by_layer = {}
            for (L, h) in subset:
                heads_by_layer.setdefault(L, []).append(h)
            c = {"moved": 0, "held": 0, "abstain": 0}
            for sp in specs:
                hooks = subset_mask_hooks(sp["span"], heads_by_layer)
                cg, _ = generate(push(sp["q"], sp["stated"], sp["challenge"]), MAX_NEW_TOKENS, hooks)
                eids = elicit_ids_of(sp["q"], sp["stated"], sp["challenge"], cg)
                eg, _ = generate(eids, ELICIT_TOK, hooks)
                c[interpret(cell, commit_prog(eg, sp["C"], sp["W"]))] += 1
            r = _rate(c)
            memo[subset] = r
            print(f"[greedy {cell}] subset={sorted(list(subset))} rate={r} "
                  f"(eval {len(memo)})", flush=True)   # liveness: multi-hour phase, marker-poll needs a heartbeat
            return r
        return rate_of_subset

    greedy_fold = {"selected": [], "final_rate": base_fold, "base_rate": base_fold, "total_drop": None,
                   "stopped_reason": "no_base", "trace": []}
    greedy_listen = dict(greedy_fold, base_rate=base_listen, final_rate=base_listen)
    if base_fold is not None and cand_fold:
        greedy_fold = greedy_select(base_fold, cand_fold, make_rate_fn(derive_specs["fold"], "fold"))
    if base_listen is not None and cand_listen:
        greedy_listen = greedy_select(base_listen, cand_listen, make_rate_fn(derive_specs["listen"], "listen"))

    # ---------------------------------------------------------------- B2 write directions
    wf = fit_write_directions(write_resid["fold"], write_resid["neutral"])
    wl = fit_write_directions(write_resid["listen"], write_resid["neutral_wstar"])
    write_cosines = None
    if wf is not None and wl is not None and wf["dirs"].shape == wl["dirs"].shape:
        write_cosines = [round(cosine(wf["dirs"][b], wl["dirs"][b]), 4) for b in range(len(band))]

    # ---------------------------------------------------------------- decisions
    stable_recs = [r for r in records if r["span_stable"]]
    floor = _rate(arm_counts(stable_recs, "neutral_wstar_mask"))
    wstar_nomask_drift = _rate(arm_counts(stable_recs, "neutral_wstar_nomask"))
    neutral_fold_drift = _rate(arm_counts(stable_recs, "neutral_mask"))
    # matcher-version note: the committed p2 listen_mask_rate was scored with v1 commit_prog; the floor
    # here uses commit_prog_v2 (word-boundary). The Addendum-4 full rescore moved zero decisions on this
    # family, so the mix is benign; recorded here so the A2 delta is not over-read at the 3rd decimal.
    a2 = listen_ko_reread(floor, (p2 or {}).get("listen_mask_rate"))

    mean_neutral_attn = {k: (np.mean(np.array(v), axis=0).tolist() if v else None)
                         for k, v in neutral_attn.items()}
    mean_neutral_mlp = {k: (np.mean(np.array(v), axis=0).tolist() if v else None)
                        for k, v in neutral_mlp.items()}
    nm_top = topk_layers(mean_neutral_attn["neutral_mask"]) if mean_neutral_attn["neutral_mask"] else None
    nw_top = topk_layers(mean_neutral_attn["neutral_wstar_mask"]) if mean_neutral_attn["neutral_wstar_mask"] else None
    a3 = {"fold_side": dla_baseline_verdict(nm_top, (p2 or {}).get("top_fold")),
          "listen_side": dla_baseline_verdict(nw_top, (p2 or {}).get("top_listen"))}

    hf = handle_freeze_decision(base_fold, greedy_fold["total_drop"], greedy_listen["total_drop"],
                                len(cand_fold), len(cand_listen))

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- persist
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)
    derive_qs = [it["q"] for it in derive_items]
    eval_qs = [it["q"] for it in eval_items]

    thresholds = {"MAX_NEW_TOKENS": MAX_NEW_TOKENS, "ELICIT_TOK": ELICIT_TOK, "MASK_NEG": MASK_NEG,
                  "WRITE_BAND": list(band), "READ_TOPK": READ_TOPK, "SUBSET_MAX": SUBSET_MAX,
                  "GREEDY_MIN_DROP": GREEDY_MIN_DROP, "HANDLE_WEAK_DROP": HANDLE_WEAK_DROP,
                  "MIN_BASE_RATE": MIN_BASE_RATE, "LISTEN_KO_MARGIN": LISTEN_KO_MARGIN,
                  "DLA_OVERLAP_TOPK": DLA_OVERLAP_TOPK, "DLA_OVERLAP_MIN": DLA_OVERLAP_MIN,
                  "DLA_DISJOINT_MAX": DLA_DISJOINT_MAX}
    decision_rule = (
        "A1 span_stability: per-(item,arm) SPAN_UNSTABLE (shared-prefix ids or challenge span not constant "
        "across counter+elicit stages) -> excluded; count reported. "
        "A2 listen_ko_reread: LISTEN_KO_CLEARS_FLOOR iff committed listen_mask_rate - neutral_wstar_mask "
        f"floor >= {LISTEN_KO_MARGIN}; else LISTEN_KO_AT_FLOOR; INSUFFICIENT if either missing. "
        f"A3 dla_baseline_verdict (per neutral arm vs committed pushed-arm top-{DLA_OVERLAP_TOPK} "
        f"|attn_delta|): GENERIC_ANSWER_FORMATION if overlap >= {DLA_OVERLAP_MIN}, DISCRIMINATES if "
        f"<= {DLA_DISJOINT_MAX}, else MIXED. "
        f"B handle_freeze: UNDERPOWERED iff DERIVE fold base rate < {MIN_BASE_RATE} or no candidates, else "
        f"FROZEN; a read side with greedy total drop < {HANDLE_WEAK_DROP} flagged WEAK_AT_DERIVE (still "
        "frozen). Read-side greedy stops when marginal fold/listen commit-rate drop < "
        f"{GREEDY_MIN_DROP} or subset size {SUBSET_MAX}. NO cross-transport / one-lever / direct-total "
        "decision here (those are 3b); write cosines are report-only.")

    summary = {
        "name": name, "family": family, "n_items": len(items),
        "n_derive": len(derive_items), "n_eval": len(eval_items),
        "thresholds": thresholds, "decision_rule": decision_rule,
        "p2_committed": p2,
        "span_stability": {"n_span_unstable": n_span_unstable, "unstable_log": span_unstable_log,
                           "category": "SPAN_STABLE_ALL" if n_span_unstable == 0 else "SPAN_UNSTABLE_PRESENT"},
        "listen_ko_reread": a2,
        "neutral_floors": {"neutral_wstar_mask_floor": floor, "neutral_wstar_nomask_drift": wstar_nomask_drift,
                           "neutral_mask_fold_drift": neutral_fold_drift},
        "dla_baseline_verdict": a3,
        "mean_neutral_attn_delta": mean_neutral_attn, "mean_neutral_mlp_delta": mean_neutral_mlp,
        "read_side": {"base_fold_rate": base_fold, "base_listen_rate": base_listen,
                      "top_candidates_fold": [list(c) for c in cand_fold],
                      "top_candidates_listen": [list(c) for c in cand_listen],
                      "greedy_fold": greedy_fold, "greedy_listen": greedy_listen},
        "write_side": {"band": list(band),
                       "n_items_fold": (wf["n_items"] if wf else 0),
                       "n_items_listen": (wl["n_items"] if wl else 0),
                       "cosine_per_layer_fold_vs_listen_REPORT_ONLY": write_cosines},
        "handle_freeze": hf,
        "arm_rates": {arm: _rate(arm_counts(stable_recs, arm))
                      for arm in NEUTRAL_ARMS + DERIVE_ARMS},
        "arm_counts": {arm: arm_counts(stable_recs, arm)
                       for arm in NEUTRAL_ARMS + DERIVE_ARMS},
        "items": records,
    }
    sp = outdir / f"foldlisten_phase3a_{tag}_summary.json"
    sp.write_text(json.dumps(summary, indent=2))

    # ---- frozen handle artifacts for 3b consumption ----
    npz_path = outdir / f"phase3_handles_{tag}.npz"
    npz_kw = {"band": np.array(band, dtype=np.int64)}
    if wf is not None:
        npz_kw.update(H_write_fold=wf["dirs"].astype(np.float32),
                      fold_raw_norm=wf["raw_norm"].astype(np.float32),
                      fold_resid_norm=wf["pos_resid_norm"].astype(np.float32),
                      neutral_resid_norm=wf["neg_resid_norm"].astype(np.float32))
    if wl is not None:
        npz_kw.update(H_write_listen=wl["dirs"].astype(np.float32),
                      listen_raw_norm=wl["raw_norm"].astype(np.float32),
                      listen_resid_norm=wl["pos_resid_norm"].astype(np.float32),
                      neutral_wstar_resid_norm=wl["neg_resid_norm"].astype(np.float32))
    np.savez(npz_path, **npz_kw)

    handles = {
        "name": name, "family": family, "tag": tag,
        "frozen_boundary": "Phase 3a pre-registration boundary for 3b; handles frozen BEFORE any "
                           "cross-arm/transport evaluation.",
        "derive_item_qs": derive_qs, "eval_item_qs": eval_qs,
        "thresholds": thresholds,
        "read_handles": {
            "H_read_fold": greedy_fold["selected"], "H_read_listen": greedy_listen["selected"],
            "ranking_fold_top": [[L, h, round(m, 6)] for (L, h, m) in rank_fold[:READ_TOPK]],
            "ranking_listen_top": [[L, h, round(m, 6)] for (L, h, m) in rank_listen[:READ_TOPK]],
            "greedy_trace_fold": greedy_fold["trace"], "greedy_trace_listen": greedy_listen["trace"],
            "base_fold_rate": base_fold, "base_listen_rate": base_listen,
            "total_drop_fold": greedy_fold["total_drop"], "total_drop_listen": greedy_listen["total_drop"]},
        "write_handles": {
            "band": list(band),
            "fold_raw_norm": (wf["raw_norm"].tolist() if wf else None),
            "listen_raw_norm": (wl["raw_norm"].tolist() if wl else None),
            "fold_resid_norm": (wf["pos_resid_norm"].tolist() if wf else None),
            "listen_resid_norm": (wl["pos_resid_norm"].tolist() if wl else None),
            "cosine_per_layer_fold_vs_listen_REPORT_ONLY": write_cosines,
            "note": "H_write_fold / H_write_listen unit directions are in the .npz; cosine>=0.7 is a 3b "
                    "decision, NOT evaluated here."},
        "handle_freeze": hf, "npz": str(npz_path),
        "decision_rule_for_3b": decision_rule,
    }
    jp = outdir / f"phase3_handles_{tag}.json"
    jp.write_text(json.dumps(handles, indent=2))

    print(f"\n[{tag}] A1 span_stability: {summary['span_stability']['category']} "
          f"(n_unstable={n_span_unstable})", flush=True)
    print(f"[{tag}] A2 listen_ko_reread: {a2['category']} -- {a2['msg']}", flush=True)
    print(f"[{tag}] A3 dla_baseline: fold_side={a3['fold_side']['category']} "
          f"listen_side={a3['listen_side']['category']}", flush=True)
    print(f"[{tag}] B handle_freeze: {hf['category']} (fold {hf['read_side_fold']}, "
          f"listen {hf['read_side_listen']}); H_read_fold={greedy_fold['selected']} "
          f"H_read_listen={greedy_listen['selected']}", flush=True)
    print(f"[written] {sp}\n[written] {npz_path}\n[written] {jp}", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    # ---- A1 span arithmetic + stability ----
    assert challenge_span(10, 15) == (10, 15)
    try:
        challenge_span(15, 10); raise SystemExit("challenge_span must reject L1<=L0")
    except AssertionError:
        pass
    stable = assess_span_stability([{"stage": "counter", "prefix_ok": True, "span": [4, 9]},
                                    {"stage": "elicit", "prefix_ok": True, "span": [4, 9]}])
    assert stable["stable"] and stable["reason"] is None, stable
    mm = assess_span_stability([{"stage": "counter", "prefix_ok": False, "span": [4, 9]},
                                {"stage": "elicit", "prefix_ok": True, "span": [4, 9]}])
    assert not mm["stable"] and mm["reason"] == "prefix_mismatch", mm
    sc = assess_span_stability([{"stage": "counter", "prefix_ok": True, "span": [4, 9]},
                                {"stage": "elicit", "prefix_ok": True, "span": [4, 10]}])
    assert not sc["stable"] and sc["reason"] == "span_changed", sc

    # ---- A3 DLA ranking + verdict (planted top-5 on both sides of both thresholds) ----
    pf = [0.0] * 42
    for L in (10, 11, 12, 13, 14):
        pf[L] = 5.0 - 0.1 * (L - 10)
    assert topk_layers(pf) == [10, 11, 12, 13, 14], topk_layers(pf)
    assert dla_baseline_verdict([12, 13, 14, 20, 21], [10, 11, 12, 13, 14])["category"] == "GENERIC_ANSWER_FORMATION"
    assert dla_baseline_verdict([30, 31, 32, 33, 34], [10, 11, 12, 13, 14])["category"] == "DISCRIMINATES"
    assert dla_baseline_verdict([13, 14, 30, 31, 32], [10, 11, 12, 13, 14])["category"] == "MIXED"
    assert dla_baseline_verdict([14, 30, 31, 32, 33], [10, 11, 12, 13, 14])["category"] == "DISCRIMINATES"  # overlap == DLA_DISJOINT_MAX edge
    assert dla_baseline_verdict(None, [1, 2, 3])["category"] == "INSUFFICIENT"
    assert dla_baseline_verdict([1, 2, 3], None)["category"] == "INSUFFICIENT"

    # ---- A2 listen-KO re-read (both sides of the 0.18 margin; None -> INSUFFICIENT). Float-clean
    #      boundary values: 0.18 - 0.0 == the stored margin double, so the inclusive edge holds. ----
    assert listen_ko_reread(0.0, 0.18)["category"] == "LISTEN_KO_CLEARS_FLOOR"    # delta == margin inclusive
    assert listen_ko_reread(0.30, 0.50)["category"] == "LISTEN_KO_CLEARS_FLOOR"   # delta 0.20 clears
    assert listen_ko_reread(0.0, 0.17)["category"] == "LISTEN_KO_AT_FLOOR"         # delta 0.17 just under
    assert listen_ko_reread(None, 0.50)["category"] == "INSUFFICIENT"
    assert listen_ko_reread(0.30, None)["category"] == "INSUFFICIENT"

    # ---- B1 head ranking ----
    mass = np.array([[0.1, 0.9], [0.5, 0.2]])
    rk = rank_heads(mass)
    assert rk[0] == (0, 1, 0.9) and rk[1] == (1, 0, 0.5), rk          # descending by mass
    assert top_candidates(rk, 2) == [(0, 1), (1, 0)], top_candidates(rk, 2)

    # ---- B1 greedy: stop on min_drop (picks max-drop candidate each step, rejects the sub-threshold add) ----
    A, B, Cc = (0, 0), (1, 1), (2, 2)
    r1 = {frozenset([A]): 0.7, frozenset([B]): 0.9, frozenset([Cc]): 0.95,
          frozenset([A, B]): 0.55, frozenset([A, Cc]): 0.68, frozenset([A, B, Cc]): 0.54}
    g1 = greedy_select(1.0, [A, B, Cc], lambda s: r1[s], max_size=6, min_drop=0.03)
    assert g1["selected"] == [[0, 0], [1, 1]] and g1["stopped_reason"] == "min_drop", g1
    assert abs(g1["total_drop"] - 0.45) < 1e-9 and len(g1["trace"]) == 3, g1
    assert g1["trace"][-1]["added"] is False, g1["trace"]

    # ---- B1 greedy: stop on max_size ----
    r2 = {frozenset([A]): 0.5, frozenset([B]): 0.6, frozenset([Cc]): 0.7,
          frozenset([A, B]): 0.2, frozenset([A, Cc]): 0.3}
    g2 = greedy_select(1.0, [A, B, Cc], lambda s: r2[s], max_size=2, min_drop=0.03)
    assert g2["selected"] == [[0, 0], [1, 1]] and g2["stopped_reason"] == "max_size", g2
    assert abs(g2["total_drop"] - 0.8) < 1e-9, g2

    # ---- B1 greedy: marginal drop EXACTLY == min_drop is added (>= semantics), next step stops ----
    r2b = {frozenset([A]): 0.47, frozenset([B]): 0.48, frozenset([Cc]): 0.49,
           frozenset([A, B]): 0.46, frozenset([A, Cc]): 0.465}
    g2b = greedy_select(0.5, [A, B, Cc], lambda s: r2b[s], max_size=6, min_drop=0.03)
    assert g2b["selected"] == [[0, 0]] and g2b["stopped_reason"] == "min_drop", g2b  # 0.03 edge added; 0.01 rejected

    # ---- B1 greedy: WEAK (no candidate clears min_drop) -> empty subset, zero total drop ----
    r3 = {frozenset([A]): 0.99, frozenset([B]): 0.985, frozenset([Cc]): 0.98}
    g3 = greedy_select(1.0, [A, B, Cc], lambda s: r3[s], max_size=6, min_drop=0.03)
    assert g3["selected"] == [] and abs(g3["total_drop"]) < 1e-9 and g3["stopped_reason"] == "min_drop", g3

    # ---- Part B deterministic even/odd split (by sorted question string) ----
    fam = [{"q": c, "correct": "x", "Wstar": "y"} for c in ["d", "a", "f", "c", "b", "e"]]
    d_set, e_set = derive_eval_split(fam)
    assert [it["q"] for it in d_set] == ["a", "c", "e"], d_set     # sorted a..f, even idx
    assert [it["q"] for it in e_set] == ["b", "d", "f"], e_set     # odd idx
    assert len(d_set) == 3 and len(e_set) == 3

    # ---- B2 write directions: unit-normalize + per-layer diff-of-means + cosine (report only) ----
    assert abs(np.linalg.norm(unit_normalize([3.0, 4.0])) - 1.0) < 1e-9
    assert abs(cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0, 0], [0, 1, 0])) < 1e-9
    assert abs(cosine([1, 0, 0], [-1, 0, 0]) + 1.0) < 1e-9
    # per item the array is (n_band=2, d=2): layer0 lies on the x-axis, layer1 on the y-axis, so the
    # per-layer diff-of-means directions come out [1,0] and [0,1] (orthogonal).
    pos = {0: np.array([[2.0, 0.0], [0.0, 3.0]]), 1: np.array([[4.0, 0.0], [0.0, 5.0]])}
    neg = {0: np.array([[0.0, 0.0], [0.0, 0.0]]), 1: np.array([[0.0, 0.0], [0.0, 0.0]])}
    w = fit_write_directions(pos, neg)
    assert w["dirs"].shape == (2, 2) and w["n_items"] == 2, w
    assert np.allclose(w["dirs"][0], [1.0, 0.0]) and np.allclose(w["dirs"][1], [0.0, 1.0]), w["dirs"]
    assert abs(cosine(w["dirs"][0], w["dirs"][1])) < 1e-9                # orthogonal write dirs
    assert fit_write_directions({}, neg) is None                        # no shared items -> None

    # ---- handle-freeze decision (FROZEN / WEAK flag / UNDERPOWERED) ----
    hf_ok = handle_freeze_decision(0.9, 0.45, 0.30, 10, 10)
    assert hf_ok["category"] == "FROZEN" and hf_ok["read_side_fold"] == "OK" and hf_ok["read_side_listen"] == "OK", hf_ok
    hf_weak = handle_freeze_decision(0.9, 0.05, 0.30, 10, 10)
    assert hf_weak["category"] == "FROZEN" and hf_weak["read_side_fold"] == "WEAK_AT_DERIVE", hf_weak
    assert handle_freeze_decision(0.4, 0.45, 0.30, 10, 10)["category"] == "UNDERPOWERED"      # base < min
    assert handle_freeze_decision(0.9, 0.45, 0.30, 0, 10)["category"] == "UNDERPOWERED"       # no candidates
    assert handle_freeze_decision(None, 0.45, 0.30, 10, 10)["category"] == "UNDERPOWERED"     # base None
    assert handle_freeze_decision(0.9, None, 0.30, 10, 10)["read_side_fold"] == "WEAK_AT_DERIVE"
    assert handle_freeze_decision(0.5, 0.45, 0.30, 10, 10)["category"] == "FROZEN"                # base == MIN_BASE_RATE edge
    assert handle_freeze_decision(0.9, 0.10, 0.30, 10, 10)["read_side_fold"] == "OK"              # drop == HANDLE_WEAK_DROP edge

    # ---- arm counts / rate route through interpret() (W*-stated neutral arm scores the LISTEN cell) ----
    recs = [{"arm": "neutral_wstar_mask", "cell": "listen", "commit_elicit": "correct"},   # moved off W*
            {"arm": "neutral_wstar_mask", "cell": "listen", "commit_elicit": "wrong"},      # held W*
            {"arm": "neutral_wstar_mask", "cell": "listen", "commit_elicit": "other"},      # abstain
            {"arm": "fold_nomask", "cell": "fold", "commit_elicit": "wrong"}]               # folded
    assert arm_counts(recs, "neutral_wstar_mask") == {"moved": 1, "held": 1, "abstain": 1}
    assert _rate(arm_counts(recs, "neutral_wstar_mask")) == 0.5
    assert _rate(arm_counts(recs, "fold_nomask")) == 1.0
    assert _rate({"moved": 0, "held": 0, "abstain": 3}) is None

    print("[selftest] span/stability, DLA rank+verdict, listen_ko_reread, head-rank, greedy(min_drop/"
          "max_size/weak), even-odd split, write dirs+cosine, handle_freeze, arm_counts -- all PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="model-free pure-logic tests (default action)")
    ap.add_argument("--run", action="store_true", help="GPU pass: Part A patches + Part B handle derivation")
    ap.add_argument("--family", default="mechanism_family_9bit.json")
    ap.add_argument("--p2-summary", dest="p2_summary", default=None,
                    help="committed Phase-2 summary JSON (cited, never recomputed): listen_mask rate + "
                         "fold/listen top-5 |attn_delta| layers")
    ap.add_argument("--name", default="google/gemma-2-9b-it")
    ap.add_argument("--tag", default="p3a_9bit")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--n", type=int, default=0, help="cap items (0 = all; smoke only)")
    ap.add_argument("--write-band-lo", dest="wb_lo", type=int, default=None,
                    help="explicit write-band lower layer, inclusive (default = frozen 9b L28-37). "
                         "Scale-transport relative-depth analogue: 2b->17, 27b->31.")
    ap.add_argument("--write-band-hi", dest="wb_hi", type=int, default=None,
                    help="explicit write-band upper layer, EXCLUSIVE (default = frozen 9b L28-37). "
                         "Scale-transport relative-depth analogue: 2b->24, 27b->42.")
    a = ap.parse_args()
    if a.run and not a.selftest:
        run(a.family, a.name, a.tag, a.device, a.chat, a.n, a.p2_summary, a.wb_lo, a.wb_hi)
    else:
        selftest()
