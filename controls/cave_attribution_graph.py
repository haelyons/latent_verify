"""FOUNDING-METHOD attribution-graph for caving on gemma-2-2b (base, Q/A) -- the NODE (feature) basis, the
opposite of the direction-based controls. (Step B of the RESEARCH_QUESTIONS attribution-graph route; sibling
of cave_direction_sae_decomp.py / cave_suppress_vs_install.py / faithful_caving.py / headset_direction.py.)

CONTEXT (neutral). Every prior caving control measures a DIRECTION: a diff-of-means residual cave-vector
u_cave(L), its necessity/sufficiency (headset_direction / cave_direction_heldout), where suppression sends
the emitted answer (cave_suppress_vs_install), whether u_cave moves the REALIZED token (faithful_caving),
and how u_cave decomposes into a frozen SAE dictionary (cave_direction_sae_decomp). This control measures the
OPPOSITE basis: instead of one fitted residual direction, it builds the feature-level ATTRIBUTION GRAPH whose
NODES are cross-layer-transcoder FEATURES and whose EDGES are direct attributions, for the realized caving
logit-difference at the answer slot, using the circuit-tracer library + GemmaScope transcoders. It introduces
no new caving mechanism and attaches no hypothesis to any node, layer, or number -- it only constructs the
graph, prunes to the influential subgraph, and reports the influence concentration / completeness / ablation.

GRAPH INSTANCE. ONE faithful caving instance, selected by the EXACT cave_suppress_vs_install logic: on the
wide misconception pool (misconception_pool.ITEMS_WIDE), under NEUTRAL and COUNTER (W* asserted) prompts
(job_truthful_flip PUSH/NEUTRAL turns; qa template for base), restrict to CAVING items (counter lowers the
first-token margin M = logp(C) - logp(W*) from neutral by >= MIN_EFFECT_NET) whose COUNTER argmax IS the
W*-first-token (argmax == W*-first-tok) -- the items where the model would ACTUALLY emit W* under pushback, so
"trace the realized cave" is a well-posed question. Of those, pick the argmax-W* item with the LARGEST
in-sample cave magnitude (the M drop M_neu - M_ctr), deterministically (the argmax-W* caving-item selection
of cave_suppress_vs_install). If no such item exists -> status="NO_FAITHFUL_INSTANCE".

TARGET. The realized caving logit-difference at the answer slot on the COUNTER prompt:
    target = logit[W*-first-tok] - logit[C-first-tok]   (the unpushed competitor C; the realized-answer
    readout of faithful_caving._readout -- W* is the emitted token, C is what it caved AWAY from).
The two logit nodes are the graph's output side; the COUNTER prompt's input tokens are the input side.

WHAT IT BUILDS (circuit-tracer + GemmaScope-2b transcoders, real run):
  (a) ReplacementModel for google/gemma-2-2b with the gemma cross-layer transcoders (the tracer's bundled /
      GemmaScope-2b transcoder set; the symbol used is recorded in the output).
  (b) attribution for the COUNTER prompt -> the graph: NODES = transcoder FEATURES (layer, position, feature
      index) + ERROR/residual nodes + input TOKEN nodes + the two LOGIT nodes; EDGES = direct attributions.
  (c) node INFLUENCE on the target logit-diff (the tracer's influence / direct-effect on the logit node), and
      PRUNE to the influential subgraph.
  REPORTED: the top-TOPK feature nodes (layer, position, feature index, activation, influence/attribution to
  the target); the input-token -> ... -> W*-logit path (the highest-influence path through the pruned graph);
  the COMPLETENESS (fraction of the target logit-diff explained by the FEATURE subgraph vs the transcoder
  ERROR/residual nodes); and a VALIDATION ablation -- clamp the top-k feature nodes OFF and measure the drop
  in the realized caving metric (the same logit-diff), vs a MATCHED-RANDOM feature set of equal size.

This is claim-blind: it measures the influence concentration, completeness, and ablation effect of the
feature-level caving circuit. It attaches no hypothesis to any node, layer, sign, or whether the circuit is
sparse vs broad.

TOOLING. circuit-tracer (decoderesearch/circuit-tracer; `pip install circuit-tracer`) with its GemmaScope
transcoder support for google/gemma-2-2b (~15GB; the tracer FULLY supports 2b -- bundled GemmaScope-2b
transcoders, PLT + CLT; per RESEARCH_QUESTIONS PART 7). The API is accessed through a small adapter
(_attribute_graph) that tries the documented circuit-tracer symbols (ReplacementModel.from_pretrained,
attribute / Graph, node influence / prune) and adapts across versions. If circuit-tracer OR the transcoders
are UNAVAILABLE / unloadable, _attribute_graph prints a clear diagnostic, the run writes a JSON with
status="TOOLING_UNAVAILABLE" (+ what was tried) and exits 0 -- it does NOT crash. The box needs
`pip install circuit-tracer` + the GemmaScope-2b transcoders for the real run (NOT for --selftest).

NEUTRAL DECISION (module constants TOPK=15, SPARSE_FRAC=0.5, COMPLETE_TOL=0.5, ABLATE_THR=0.2; numbers +
categories only, no hypothesis). Once the graph + influences exist:
  INCOMPLETE         iff the FEATURE subgraph explains < COMPLETE_TOL of the target logit-diff (the
                         transcoder ERROR/residual nodes dominate -> the transcoders do not capture caving;
                         no sparse/broad call is meaningful). Checked FIRST.
  SPARSE_CIRCUIT     iff completeness >= COMPLETE_TOL AND the top-TOPK feature nodes carry >= SPARSE_FRAC of
                         the total feature-node influence on the target AND clamping them OFF drops the
                         realized caving metric by >= ABLATE_THR (relative) while the matched-RANDOM feature
                         set does NOT (its drop < ABLATE_THR).
  BROAD_DISTRIBUTED  iff completeness >= COMPLETE_TOL but influence is spread (top-TOPK < SPARSE_FRAC) OR no
                         such small set's ablation reaches ABLATE_THR (or the random set matches it).
Reported: influence concentration (top-TOPK influence fraction), completeness, ablation drop (top-k vs
matched-random), and the top feature nodes. Numbers + categories only; no claim attached to any node or
verdict.

Model-free --selftest (CPU, NO model load, NO circuit-tracer import) builds a SYNTHETIC attribution graph
(node influences + a target logit-diff + an error-node share + a planted ablation response) and verifies the
concentration / completeness / ablation decision logic and its boundaries:
  (i)   a few high-influence feature nodes carrying >= SPARSE_FRAC + their ablation crosses ABLATE_THR while
        matched-random does not + completeness high -> SPARSE_CIRCUIT;
  (ii)  influence spread over many nodes (top-TOPK < SPARSE_FRAC) -> BROAD_DISTRIBUTED;
  (iii) error/residual nodes dominate (completeness < COMPLETE_TOL) -> INCOMPLETE (checked first);
plus the concentration / completeness / ablation math and the SPARSE_FRAC / COMPLETE_TOL / ABLATE_THR
boundaries. Writes out/cave_attribution_graph_<tag>.json.

  python controls/cave_attribution_graph.py --selftest
  python controls/cave_attribution_graph.py --name google/gemma-2-2b --tag 2b --device cuda
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured numbers only.
TOPK = 15            # #top feature nodes for the concentration / completeness / ablation decision
SPARSE_FRAC = 0.5    # top-TOPK feature-node influence fraction >= this -> concentrated (SPARSE candidate)
COMPLETE_TOL = 0.5   # feature subgraph must explain >= this fraction of the target logit-diff (else INCOMPLETE)
ABLATE_THR = 0.2     # relative drop in the caving metric from clamping the top-k features that counts as causal
RAND_SEED = 0        # deterministic matched-random feature set (same convention as the direction controls)
MIN_EFFECT_NET = 0.5 # counter-vs-neutral M-gap that counts as a real cave (fallback; the real run defers the
                     # reference rlhf_differential.MIN_EFFECT_NET, same value). --selftest needs no import.

MODEL_DEFAULT = "google/gemma-2-2b"   # base; circuit-tracer fully supports 2b w/ GemmaScope-2b transcoders

DECISION_RULE = (
    "On the wide misconception pool, build NEUTRAL and COUNTER (W* asserted) prompts (job_truthful_flip "
    "turns; qa template for base gemma-2-2b). Restrict to CAVING items (counter lowers M=logp(C)-logp(W*) "
    "from neutral by >= MIN_EFFECT_NET) whose COUNTER argmax IS the W*-first-token; pick the argmax-W* item "
    "with the largest M-drop as the graph instance (else NO_FAITHFUL_INSTANCE). Build a circuit-tracer "
    "attribution graph for google/gemma-2-2b with GemmaScope transcoders on that COUNTER prompt; NODES = "
    "transcoder features + error/residual + input-tokens + the two logit nodes, EDGES = direct attributions; "
    "TARGET = logit[W*-first-tok] - logit[C-first-tok] at the answer slot. Compute node influence on the "
    "target, prune to the influential subgraph. Report: top-TOPK(15) feature nodes (layer, pos, feature, "
    "activation, influence); the input-token->...->W*-logit highest-influence path; COMPLETENESS = fraction "
    "of the target logit-diff explained by the FEATURE subgraph vs the ERROR/residual nodes; and a VALIDATION "
    "ablation clamping the top-k features OFF (relative drop in the caving logit-diff) vs a matched-RANDOM "
    "feature set. Decision (numbers+categories only): INCOMPLETE iff completeness < COMPLETE_TOL(0.5) (error "
    "nodes dominate); else SPARSE_CIRCUIT iff top-TOPK influence fraction >= SPARSE_FRAC(0.5) AND top-k "
    "ablation drop >= ABLATE_THR(0.2) AND matched-random drop < ABLATE_THR; else BROAD_DISTRIBUTED. If "
    "circuit-tracer / the GemmaScope-2b transcoders are unavailable -> TOOLING_UNAVAILABLE (exits 0, no "
    "crash). No claim attached to any node, layer, sign, or verdict."
)


# --------------------------------------------------------------------------- pure graph-statistic math
def influence_fraction(influences, topk=TOPK):
    """Fraction of the TOTAL absolute node influence carried by the top-`topk` nodes by |influence|.
    `influences` is a list of per-feature-node influences (signed; concentration is over magnitudes). Returns
    (frac, top_indices, total): frac in [0,1], the indices of the top-topk nodes (by |influence|, descending),
    and the total |influence|. Pure. Empty / all-zero -> (0.0, [], 0.0). The top-topk fraction is monotone
    nondecreasing in topk and bounded to [0,1] (a subset's |sum| share cannot exceed the whole)."""
    if not influences:
        return 0.0, [], 0.0
    mags = [abs(float(x)) for x in influences]
    total = float(sum(mags))
    order = sorted(range(len(mags)), key=lambda i: mags[i], reverse=True)
    top = order[:topk]
    if total <= 0.0:
        return 0.0, top, 0.0
    frac = float(sum(mags[i] for i in top)) / total
    if frac < 0.0:
        frac = 0.0
    if frac > 1.0:
        frac = 1.0
    return frac, top, total


def completeness(feature_influence_sum, error_influence_sum):
    """Fraction of the target logit-diff explained by the FEATURE subgraph vs the transcoder ERROR/residual
    nodes: |feature| / (|feature| + |error|). High -> features capture the target; low -> error nodes dominate
    (the transcoders do not reconstruct the caving computation). Pure; in [0,1]. Both-zero -> 0.0 (nothing
    explained)."""
    f = abs(float(feature_influence_sum))
    e = abs(float(error_influence_sum))
    denom = f + e
    if denom <= 0.0:
        return 0.0
    return f / denom


def relative_drop(base_metric, ablated_metric):
    """Relative drop in the (signed) caving metric from base to ablated, in [0, 1+]: (base - ablated)/|base|,
    clamped at 0 below (an ablation that INCREASES the metric is a 0 drop, not a negative one -- the ablation
    'did not attenuate'). |base| ~ 0 -> 0.0 (no metric to drop). Pure. The caving metric here is the realized
    logit-diff target (positive = caved toward W*); clamping the causal features OFF should LOWER it."""
    b = float(base_metric)
    if abs(b) < 1e-9:
        return 0.0
    d = (b - float(ablated_metric)) / abs(b)
    return d if d > 0.0 else 0.0


# --------------------------------------------------------------------------- pure decision
def decide_graph(complete_frac, topk_influence_frac, topk_ablate_drop, rand_ablate_drop,
                 complete_tol=COMPLETE_TOL, sparse_frac=SPARSE_FRAC, ablate_thr=ABLATE_THR):
    """Pure decision over the measured graph numbers only (no hypothesis attached). Resolution order:
      UNAVAILABLE       iff any required number is None (graph/influences/ablation not produced).
      INCOMPLETE        iff complete_frac < complete_tol (the transcoder error/residual nodes dominate the
                            target logit-diff -> the feature subgraph does not capture caving; checked FIRST,
                            no sparse/broad call meaningful).
      SPARSE_CIRCUIT    iff complete_frac >= complete_tol AND topk_influence_frac >= sparse_frac AND
                            topk_ablate_drop >= ablate_thr AND rand_ablate_drop < ablate_thr (a small feature
                            set concentrates the influence AND clamping it causally attenuates the caving
                            metric while a matched-random set does not).
      BROAD_DISTRIBUTED otherwise (complete but influence is spread, or no small set's ablation is causal /
                            specific)."""
    nums = (complete_frac, topk_influence_frac, topk_ablate_drop, rand_ablate_drop)
    if any(x is None for x in nums):
        return {"category": "UNAVAILABLE", "sparse_circuit": False,
                "completeness": (round(complete_frac, 4) if complete_frac is not None else None),
                "topk_influence_frac": (round(topk_influence_frac, 4) if topk_influence_frac is not None else None),
                "topk_ablate_drop": (round(topk_ablate_drop, 4) if topk_ablate_drop is not None else None),
                "rand_ablate_drop": (round(rand_ablate_drop, 4) if rand_ablate_drop is not None else None),
                "msg": "graph / influences / ablation not produced -- no sparse/broad/incomplete call."}

    complete = complete_frac >= complete_tol
    concentrated = topk_influence_frac >= sparse_frac
    ablation_causal = topk_ablate_drop >= ablate_thr
    rand_clean = rand_ablate_drop < ablate_thr

    if not complete:
        cat = "INCOMPLETE"
        msg = (f"feature subgraph completeness {complete_frac:.4f} < {complete_tol}: the transcoder "
               f"ERROR/residual nodes dominate the target logit-diff -- the transcoders do not capture the "
               f"caving computation, so no sparse/broad call is meaningful.")
    elif concentrated and ablation_causal and rand_clean:
        cat = "SPARSE_CIRCUIT"
        msg = (f"completeness {complete_frac:.4f} >= {complete_tol}; the top-{TOPK} feature nodes carry "
               f"{topk_influence_frac:.4f} >= {sparse_frac} of the feature-node influence AND clamping them "
               f"OFF drops the caving metric by {topk_ablate_drop:.4f} >= {ablate_thr} while a matched-random "
               f"feature set drops only {rand_ablate_drop:.4f} < {ablate_thr}: a small, specific feature "
               f"circuit carries the realized cave.")
    else:
        cat = "BROAD_DISTRIBUTED"
        reasons = []
        if not concentrated:
            reasons.append(f"top-{TOPK} influence frac {topk_influence_frac:.4f} < {sparse_frac} (spread)")
        if not ablation_causal:
            reasons.append(f"top-k ablation drop {topk_ablate_drop:.4f} < {ablate_thr} (not causal)")
        if ablation_causal and not rand_clean:
            reasons.append(f"matched-random drop {rand_ablate_drop:.4f} >= {ablate_thr} (not specific)")
        msg = (f"completeness {complete_frac:.4f} >= {complete_tol} but " + "; ".join(reasons) +
               ": the realized cave is not carried by a small, specific feature set.")
    return {"category": cat, "sparse_circuit": cat == "SPARSE_CIRCUIT",
            "incomplete": cat == "INCOMPLETE", "broad_distributed": cat == "BROAD_DISTRIBUTED",
            "complete": bool(complete), "concentrated": bool(concentrated),
            "ablation_causal": bool(ablation_causal), "rand_clean": bool(rand_clean),
            "completeness": round(complete_frac, 4),
            "topk_influence_frac": round(topk_influence_frac, 4),
            "topk_ablate_drop": round(topk_ablate_drop, 4),
            "rand_ablate_drop": round(rand_ablate_drop, 4), "msg": msg}


# --------------------------------------------------------------------------- real-run helpers
def _full_softmax(logits):
    """Full next-token probability vector at the LAST position. gemma-2's final softcap is applied inside the
    forward (cave_suppress_vs_install._full_softmax / faithful_caving._full_softmax). 1-D float tensor."""
    return torch.softmax(logits[0, -1].float(), dim=-1)


def _logp_diff_local(logits, cid, aid):
    """First-token margin M = logp(C) - logp(W*) at the last position (faithful_caving._logp_diff_local /
    rlhf_differential._logp_diff). The cave metric. Caving lowers M (drives it negative)."""
    lp = torch.log_softmax(logits[0, -1].float(), -1)
    return float(lp[cid] - lp[aid])


def _readout(P, cid, aid):
    """Realized readout from a full softmax P: argmax token id, P(C first-tok), P(W* first-tok). Pure.
    (cave_suppress_vs_install._readout / faithful_caving._readout.)"""
    return {"argmax": int(P.argmax()), "p_c": float(P[cid]), "p_w": float(P[aid])}


def _select_instance(model, pool, device, min_effect_net):
    """Pick ONE faithful caving instance with the EXACT cave_suppress_vs_install selection logic: per pool
    item build NEUTRAL/COUNTER, read M + the realized next-token softmax, keep CAVING items (M_neu - M_ctr >=
    min_effect_net) whose COUNTER argmax IS the W*-first-tok; of those, return the one with the LARGEST M-drop
    (M_neu - M_ctr). Returns (instance_dict | None, diagnostics). Forward-only. The TransformerLens forward is
    only used to SELECT the instance (the same _helpers prompt builders the tracer prompt is built from);
    circuit-tracer then re-runs the prompt itself."""
    from rlhf_differential import _helpers
    from job_truthful_flip import PUSH, NEUTRAL
    raw, single, push, first, num_lp = _helpers(model, device, False)   # base, qa (Q/A template)
    n_ok = n_cave = n_argmaxW = 0
    best = None
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        if cid == aid:                                  # first-token collision -> readout degenerate, skip
            continue
        n_ok += 1
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        with torch.no_grad():
            lg_n = model(neutral)
            lg_c = model(counter)
        Pc = _full_softmax(lg_c)
        ctr = _readout(Pc, cid, aid)
        M_neu = _logp_diff_local(lg_n, cid, aid)
        M_ctr = _logp_diff_local(lg_c, cid, aid)
        drop = M_neu - M_ctr
        is_cave = drop >= min_effect_net
        if is_cave:
            n_cave += 1
        argmaxW = is_cave and (ctr["argmax"] == aid)
        if argmaxW:
            n_argmaxW += 1
            print(f"  [select] argmax-W* cave item {i}: M_neu={M_neu:+.2f} M_ctr={M_ctr:+.2f} "
                  f"drop={drop:+.2f} amx_ctr={ctr['argmax']}==W* q={q[:36]!r}", flush=True)
            if best is None or drop > best["m_drop"]:
                best = {"i": i, "q": q, "correct": C, "Wstar": W, "cid": cid, "aid": aid,
                        "counter": counter, "neutral": neutral, "M_neu": M_neu, "M_ctr": M_ctr,
                        "m_drop": drop, "ctr_argmax": ctr["argmax"], "ctr_p_w": ctr["p_w"],
                        "ctr_p_c": ctr["p_c"]}
    diag = {"n_ok": n_ok, "n_cave": n_cave, "n_argmaxW_cave": n_argmaxW}
    return best, diag


def _attribute_graph(name, device, instance):
    """Adapter around circuit-tracer. Tries the documented API to (a) load a ReplacementModel for
    google/gemma-2-2b with the gemma GemmaScope transcoders, (b) run attribution on the COUNTER prompt to get
    the graph (feature + error + token + logit nodes; attribution edges), (c) compute node influence on the
    TARGET = logit[W*-first-tok] - logit[C-first-tok] at the answer slot, prune to the influential subgraph,
    and (d) run the VALIDATION ablation (clamp the top-k feature nodes OFF -> measure the drop in the realized
    caving logit-diff, vs a matched-random equal-size feature set). circuit-tracer's exact symbols differ
    across versions; this adapter tries the known entry points and adapts. On ANY failure (import / load /
    attribute / transcoders unavailable) it returns ("TOOLING_UNAVAILABLE", info-with-`tried`) -- the caller
    writes that status and exits 0; it never crashes. On success it returns ("OK", graph-stats dict).

    The graph-stats dict (status OK) carries the raw numbers the pure decision consumes:
      all_feature_influence: FULL per-feature-node influence list on the target (for the concentration frac)
      feature_nodes        : top {layer, pos, feature, activation, influence} pruned feature nodes (reporting)
      feature_influence_sum: signed sum of feature-node influence on the target
      error_influence_sum  : signed sum of error/residual-node influence on the target
      target_logit_diff    : the realized caving logit-diff at the answer slot (W* - C)
      topk_ablate_metric   : the caving logit-diff after clamping the top-k feature nodes OFF
      rand_ablate_metric   : the caving logit-diff after clamping a matched-random equal-size feature set OFF
      input_to_logit_path  : the highest-influence input-token -> ... -> W*-logit path (list of node labels)
    """
    tried = []
    target_desc = ("logit[W*-first-tok] - logit[C-first-tok] at the answer slot of the COUNTER prompt")

    # ---- import circuit-tracer (robust to its module layout across versions) ----
    ct = None
    for modpath, attrs in (("circuit_tracer", ("ReplacementModel", "attribute")),
                           ("circuit_tracer.replacement_model", ("ReplacementModel",)),
                           ("circuit_tracer.attribution", ("attribute",))):
        try:
            import importlib
            m = importlib.import_module(modpath)
            tried.append(f"import {modpath}: ok ({[a for a in attrs if hasattr(m, a)]})")
            ct = ct or {}
            for a in attrs:
                if hasattr(m, a) and a not in ct:
                    ct[a] = getattr(m, a)
        except Exception as e:
            tried.append(f"import {modpath}: FAILED ({type(e).__name__}: {e})")
    if not ct or "ReplacementModel" not in ct:
        print("  [tracer] circuit-tracer not importable / ReplacementModel missing; "
              "`pip install circuit-tracer` + GemmaScope-2b transcoders on the box.", flush=True)
        return "TOOLING_UNAVAILABLE", {"tried": tried, "target": target_desc,
                                       "reason": "circuit_tracer import / ReplacementModel unavailable"}

    ReplacementModel = ct["ReplacementModel"]

    # ---- (a) load ReplacementModel for gemma-2-2b with gemma transcoders ----
    rmodel = None
    transcoder_set = None
    for ts in ("gemma", "gemma-2-2b", "gemma_scope", "gemmascope-2b"):
        try:
            rmodel = ReplacementModel.from_pretrained(name, ts, device=device)
            transcoder_set = ts
            tried.append(f"ReplacementModel.from_pretrained({name!r}, {ts!r}): ok")
            break
        except TypeError:
            try:
                rmodel = ReplacementModel.from_pretrained(name, transcoder_set=ts)
                transcoder_set = ts
                tried.append(f"ReplacementModel.from_pretrained({name!r}, transcoder_set={ts!r}): ok")
                break
            except Exception as e:
                tried.append(f"ReplacementModel.from_pretrained({name!r}, {ts!r}): FAILED "
                             f"({type(e).__name__}: {e})")
        except Exception as e:
            tried.append(f"ReplacementModel.from_pretrained({name!r}, {ts!r}): FAILED "
                         f"({type(e).__name__}: {e})")
    if rmodel is None:
        print("  [tracer] ReplacementModel + GemmaScope-2b transcoders failed to load (see tried).",
              flush=True)
        return "TOOLING_UNAVAILABLE", {"tried": tried, "target": target_desc,
                                       "reason": "ReplacementModel / GemmaScope-2b transcoders unloadable"}

    # ---- (b) attribution on the COUNTER prompt -> graph ----
    # circuit-tracer's `attribute` takes the prompt (string or ids) + the ReplacementModel and returns a Graph
    # (nodes: features/error/tokens/logits; edges: attributions). The prompt is the SAME COUNTER text the
    # instance was selected on (Q/A template), passed through to the tracer.
    # A bare attribute(prompt, model) materializes the full feature-gradient backward graph in ONE shot (OOM'd a
    # 40GB A100 trying to alloc 22GB for a 2b model). Pass the library's memory knobs -- CPU-offload the
    # transcoder cache + a small backward batch_size -- FILTERED to whatever the installed version's signature
    # actually accepts (so an unknown kwarg can never TypeError), stepping batch_size down on OOM; first success
    # wins. NOT capping node count (that would bias the concentration/completeness read toward "sparse"); only
    # the backward pass is batched, the graph still forms fully.
    import inspect as _inspect
    attribute = ct.get("attribute", None)
    use_fn = attribute is not None
    fn = attribute if use_fn else getattr(rmodel, "attribute", None)
    if fn is None:
        return "TOOLING_UNAVAILABLE", {"tried": tried, "target": target_desc,
                                       "transcoder_set": transcoder_set,
                                       "reason": "no attribute() function or method found"}
    prompt = instance["counter_text"]
    sig_params = set()
    try:
        sig_params = set(_inspect.signature(fn).parameters)
        tried.append(f"attribute signature params: {sorted(sig_params)}")
    except Exception as e:
        tried.append(f"attribute signature introspection failed ({type(e).__name__}: {e})")

    def _memkw(bs):   # memory-frugal kwargs, filtered to the real signature
        d = {"offload": "cpu", "batch_size": bs, "verbose": False}
        return {k: v for k, v in d.items() if k in sig_params}

    graph = None
    for bs in (16, 8, 4, None):    # None -> bare call (last resort)
        kw = _memkw(bs) if bs is not None else {}
        try:
            graph = fn(prompt, rmodel, **kw) if use_fn else fn(prompt, **kw)
            tried.append(f"attribute(prompt, model, **{kw}): ok")
            break
        except TypeError as e:
            tried.append(f"attribute(**{kw}): TypeError ({str(e)[:100]}) -> next")
        except Exception as e:
            tried.append(f"attribute(**{kw}): {type(e).__name__} ({str(e)[:100]}) -> next")
            if device == "cuda":
                torch.cuda.empty_cache()
    if graph is None:
        print("  [tracer] attribute() failed across the batch-size ladder (see tried).", flush=True)
        return "TOOLING_UNAVAILABLE", {"tried": tried, "target": target_desc,
                                       "transcoder_set": transcoder_set,
                                       "reason": "circuit-tracer attribute() OOM/failed across batch-size ladder"}

    # ---- introspect the produced Graph + ReplacementModel API (captured on EVERY path) ----
    # circuit-tracer's Graph is tensor-based and its exact field/method names vary across versions; this dump
    # (attrs, tensor shapes, key-tensor heads, intervention method names) is returned in BOTH the OK and the
    # failure diagnostic so a read-off mismatch reports the real API instead of just an AttributeError.
    introspection = _introspect_graph(graph, rmodel)

    # ---- (c)+(d) influence / prune / ablation ----
    # These read off the produced `graph` and re-run clamped forwards through `rmodel`. Any accessor mismatch ->
    # TOOLING_UNAVAILABLE (never a crash), with the introspection attached.
    try:
        stats = _graph_stats_from_tracer(graph, rmodel, instance, transcoder_set)
        tried.append("graph influence / prune / ablation: ok")
    except Exception as e:
        tried.append(f"graph influence / prune / ablation: FAILED ({type(e).__name__}: {e})")
        print(f"  [tracer] influence/prune/ablation read-off failed ({type(e).__name__}: {e}).", flush=True)
        return "TOOLING_UNAVAILABLE", {"tried": tried, "target": target_desc,
                                       "transcoder_set": transcoder_set,
                                       "graph_introspection": introspection,
                                       "reason": f"circuit-tracer graph influence/prune/ablation unavailable "
                                                 f"({type(e).__name__}: {str(e)[:120]})"}

    stats["tried"] = tried
    stats["transcoder_set"] = transcoder_set
    stats["target"] = target_desc
    stats["graph_introspection"] = introspection
    return "OK", stats


def _introspect_graph(graph, rmodel):
    """Dump the circuit-tracer Graph + ReplacementModel API so a read-off mismatch reports the REAL structure
    (field names, tensor shapes, key-tensor heads, intervention method names) instead of a bare AttributeError.
    Pure inspection; never raises (best-effort, errors captured inline)."""
    info = {}
    try:
        info["graph_type"] = type(graph).__name__
        attrs = [a for a in dir(graph) if not a.startswith("_")]
        info["graph_attrs"] = attrs
        fields = {}
        for a in attrs:
            try:
                v = getattr(graph, a)
            except Exception as e:
                fields[a] = f"<getattr ERR {type(e).__name__}>"
                continue
            if callable(v):
                fields[a] = "<callable>"
            elif hasattr(v, "shape"):
                fields[a] = {"shape": list(v.shape), "dtype": str(getattr(v, "dtype", ""))}
            elif isinstance(v, (int, float, bool, str)):
                fields[a] = v if not isinstance(v, str) else v[:80]
            elif isinstance(v, (list, tuple)):
                fields[a] = {"type": type(v).__name__, "len": len(v),
                             "head": [str(x)[:50] for x in list(v)[:4]]}
            else:
                fields[a] = type(v).__name__
        info["graph_fields"] = fields
        for a in ("active_features", "selected_features", "logit_tokens", "logit_probabilities",
                  "input_tokens", "activation_values"):
            if hasattr(graph, a):
                try:
                    t = getattr(graph, a)
                    info[f"{a}_head"] = t[:8].tolist() if hasattr(t, "tolist") else str(t)[:200]
                except Exception as e:
                    info[f"{a}_head"] = f"<ERR {type(e).__name__}: {e}>"
        info["rmodel_type"] = type(rmodel).__name__
        info["rmodel_methods"] = [m for m in dir(rmodel) if not m.startswith("_") and any(
            k in m.lower() for k in ("feature", "ablat", "interven", "clamp", "hook", "run", "forward",
                                     "logit", "activation"))]
    except Exception as e:
        info["introspection_error"] = f"{type(e).__name__}: {e}"
    return info


def _graph_stats_from_tracer(graph, rmodel, instance, transcoder_set):
    """Pull the decision-relevant numbers off a circuit-tracer Graph + ReplacementModel. Isolated so the
    version-specific accessor names live in ONE place; raises on any mismatch (the caller converts that to
    TOOLING_UNAVAILABLE). Returns the graph-stats dict documented in _attribute_graph.

    The procedure (documented circuit-tracer flow):
      1. Identify the two LOGIT nodes (W*-first-tok and C-first-tok at the answer slot) and form the TARGET
         logit-diff. 2. Compute per-node INFLUENCE on the target (the tracer's influence / direct-effect
         scores) and PRUNE to the influential subgraph. 3. Split node influence into FEATURE-node influence
         and ERROR/residual-node influence (for completeness). 4. Rank feature nodes by |influence|, take the
         top-TOPK. 5. VALIDATION ablation: clamp the top-k feature activations OFF and re-read the target
         logit-diff; repeat for a matched-random equal-size feature set. 6. Extract the highest-influence
         input-token -> ... -> W*-logit path.

    Real circuit-tracer Graph (tensor-based, confirmed by introspection): adjacency_matrix [N,N] with
    A[target,source] = direct attribution; active_features [n_feat,3]=(layer,pos,feat); activation_values
    [n_feat]; logit_token_ids [n_logit]. Node layout (confirmed by arithmetic N - n_feat - n_pos - n_logit =
    n_layers*n_pos): [features | error (n_layers*n_pos) | tokens (n_pos) | logits (n_logit)]. The attribution
    graph is feedforward/acyclic, so the row-normalized adjacency is nilpotent and the geometric-path influence
    series converges (early-stopped)."""
    import random as _r
    GRAPH_HOPS = 32          # cap; nilpotent DAG -> early-stops well before this
    aid, cid = instance["aid"], instance["cid"]

    A = graph.adjacency_matrix
    N = int(A.shape[0])
    active = graph.active_features
    acts = graph.activation_values
    n_feat = int(active.shape[0])
    n_pos = int(getattr(graph, "n_pos", graph.input_tokens.shape[0]))
    logit_ids = [int(x) for x in graph.logit_token_ids.tolist()]
    n_logit = len(logit_ids)
    rem = N - n_feat - n_pos - n_logit            # = n_layers * n_pos
    if rem <= 0 or rem % n_pos != 0:
        raise RuntimeError(f"node-layout arithmetic failed: N={N} n_feat={n_feat} n_pos={n_pos} "
                           f"n_logit={n_logit} rem={rem}")
    n_layers = rem // n_pos
    feat_lo, feat_hi = 0, n_feat
    err_lo, err_hi = n_feat, n_feat + n_layers * n_pos
    logit_lo = N - n_logit
    # locate the W* and C logit node ROWS (logit nodes are the last n_logit rows, ordered as logit_token_ids)
    if aid not in logit_ids:
        raise RuntimeError(f"W* token {aid} not among graph logit nodes {logit_ids}")
    row_w = logit_lo + logit_ids.index(aid)
    row_c = logit_lo + logit_ids.index(cid) if cid in logit_ids else None

    # row-normalized |adjacency| (in place; graph discarded after) -> geometric-path influence on a logit row
    absA = A.abs_().float()
    rowsum = absA.sum(dim=1, keepdim=True).clamp_min(1e-12)
    absA.div_(rowsum)                              # now absA = row-normalized transition matrix
    norm = absA

    def _influence(row):
        infl = norm[row].clone()                   # [N] direct sources of `row`
        cur = infl.clone()
        for _ in range(GRAPH_HOPS):
            cur = cur @ norm                        # propagate one more hop back through the DAG
            infl = infl + cur
            if float(cur.abs().sum()) < 1e-9:       # nilpotent -> converges; stop early
                break
        return infl

    infl_w = _influence(row_w)
    infl = (infl_w - _influence(row_c)) if row_c is not None else infl_w
    infl = infl.detach().cpu()
    feat_infl = infl[feat_lo:feat_hi]
    err_infl = infl[err_lo:err_hi]
    feature_influence_sum = float(feat_infl.sum())
    error_influence_sum = float(err_infl.sum())
    all_feature_influence = [float(x) for x in feat_infl.tolist()]   # FULL list -> concentration frac

    # rank feature nodes by |influence|, take top (report up to max(TOPK,30))
    order = torch.argsort(feat_infl.abs(), descending=True).tolist()
    active_l = active.cpu().tolist()
    acts_l = acts.cpu().tolist()
    feat_nodes = []
    for idx in order[:max(TOPK, 30)]:
        L, p, f = active_l[idx]
        feat_nodes.append({"layer": int(L), "pos": int(p), "feature": int(f),
                           "activation": float(acts_l[idx]), "influence": float(feat_infl[idx]), "_idx": idx})
    if not feat_nodes:
        raise RuntimeError("no feature nodes on the attribution graph")
    top_idx = order[:TOPK]

    # VALIDATION ablation: clamp the top-k feature activations OFF via ReplacementModel.feature_intervention,
    # re-read the realized caving logit-diff; repeat for a matched-random equal-size feature set. Best-effort:
    # feature_intervention's signature/return vary, so introspect + try call forms; on failure leave the
    # ablation metrics None (the decision becomes UNAVAILABLE for SPARSE but concentration/completeness still
    # report). The matched-random set is an equal-size sample of the OTHER feature nodes (by influence rank).
    import inspect as _inspect
    base_diff = float(instance.get("target_logit_diff_base"))
    prompt = instance["counter_text"]
    topk_ablate_metric = rand_ablate_metric = None
    ablation_note = None
    fi_sig = None
    try:
        fi = rmodel.feature_intervention
        fi_sig = list(_inspect.signature(fi).parameters)

        def _logit_diff_after(idxs):
            interventions = [(int(active_l[i][0]), int(active_l[i][1]), int(active_l[i][2]), 0.0) for i in idxs]
            out = fi(prompt, interventions)
            lg = getattr(out, "logits", out[0] if isinstance(out, (tuple, list)) else out)
            v = (lg[0, -1] if lg.dim() == 3 else lg[-1]).float()
            return float(v[aid] - v[cid])

        topk_ablate_metric = _logit_diff_after(top_idx)
        others = order[TOPK:]
        rng = _r.Random(RAND_SEED)
        k = len(top_idx)
        rand_idx = others if len(others) <= k else rng.sample(others, k)
        rand_ablate_metric = _logit_diff_after(rand_idx) if rand_idx else base_diff
    except Exception as e:
        ablation_note = f"{type(e).__name__}: {str(e)[:180]}"

    t0 = feat_nodes[0]
    path = ["input_tokens", f"L{t0['layer']}.feat{t0['feature']}@pos{t0['pos']}", f"logit[W*={aid}]"]

    return {
        "feature_nodes": [{k2: v for k2, v in f.items() if k2 != "_idx"} for f in feat_nodes],
        "all_feature_influence": all_feature_influence,
        "n_feature_nodes": n_feat,
        "feature_influence_sum": feature_influence_sum,
        "error_influence_sum": error_influence_sum,
        "target_logit_diff": base_diff,
        "topk_ablate_metric": topk_ablate_metric,
        "rand_ablate_metric": rand_ablate_metric,
        "ablation_note": ablation_note,
        "feature_intervention_sig": fi_sig,
        "node_layout": {"n_nodes": N, "n_feat": n_feat, "n_layers": n_layers, "n_pos": n_pos,
                        "n_logit": n_logit, "logit_ids": logit_ids, "row_w": row_w, "row_c": row_c},
        "input_to_logit_path": path,
        "topk": TOPK, "n_topk_used": len(top_idx),
    }


def _stats_to_decision(stats):
    """Turn the raw graph-stats dict (status OK) into the reported numbers + the pure decision. Pure over the
    dict (the model + tracer are gone by now). Concentration is computed over the FULL feature-node influence
    list (`all_feature_influence` when present -- the real run reports only the top feature_nodes for brevity;
    the synthetic selftest stores every node in feature_nodes so the fallback is exact there)."""
    infl = stats.get("all_feature_influence")
    if infl is None:
        infl = [f["influence"] for f in stats["feature_nodes"]]
    topk_frac, top_idx, total = influence_fraction(infl, TOPK)
    comp = completeness(stats["feature_influence_sum"], stats["error_influence_sum"])
    tk_metric = stats.get("topk_ablate_metric")
    rd_metric = stats.get("rand_ablate_metric")
    have_ablation = (tk_metric is not None) and (rd_metric is not None)
    if have_ablation:
        topk_drop = relative_drop(stats["target_logit_diff"], tk_metric)
        rand_drop = relative_drop(stats["target_logit_diff"], rd_metric)
        dec = decide_graph(comp, topk_frac, topk_drop, rand_drop)
    else:
        # causal clamp unavailable -> geometry-only call (completeness + concentration); no SPARSE_CIRCUIT
        # claim is possible without the ablation, so it cannot pass for the full SPARSE category.
        topk_drop = rand_drop = None
        if comp < COMPLETE_TOL:
            cat = "INCOMPLETE"
        elif topk_frac >= SPARSE_FRAC:
            cat = "SPARSE_GEOMETRY_CAUSAL_UNCONFIRMED"
        else:
            cat = "BROAD_DISTRIBUTED"
        dec = {"category": cat, "causal_confirmed": False, "sparse_circuit": False,
               "incomplete": cat == "INCOMPLETE", "broad_distributed": cat == "BROAD_DISTRIBUTED",
               "complete": comp >= COMPLETE_TOL,
               "msg": (f"ablation unavailable ({stats.get('ablation_note')}); geometry-only: "
                       f"completeness={comp:.3f}, top{TOPK}_influence_frac={topk_frac:.3f} "
                       f"(causal clamp NOT confirmed)")}
    return {
        "completeness": round(comp, 6),
        "topk_influence_frac": round(topk_frac, 6),
        "topk_ablate_drop": (round(topk_drop, 6) if topk_drop is not None else None),
        "rand_ablate_drop": (round(rand_drop, 6) if rand_drop is not None else None),
        "target_logit_diff": round(float(stats["target_logit_diff"]), 6),
        "topk_ablate_metric": (round(float(tk_metric), 6) if tk_metric is not None else None),
        "rand_ablate_metric": (round(float(rd_metric), 6) if rd_metric is not None else None),
        "feature_influence_sum": round(float(stats["feature_influence_sum"]), 6),
        "error_influence_sum": round(float(stats["error_influence_sum"]), 6),
        "n_feature_nodes": stats.get("n_feature_nodes"),
        "top_feature_nodes": stats["feature_nodes"][:TOPK],
        "input_to_logit_path": stats.get("input_to_logit_path"),
        "ablation_note": stats.get("ablation_note"),
        "feature_intervention_sig": stats.get("feature_intervention_sig"),
        "node_layout": stats.get("node_layout"),
        "decision": dec,
    }


def run(name, tag, device, pool):
    """Real run: select ONE faithful caving instance (cave_suppress_vs_install logic) on gemma-2-2b base
    (Q/A), build the circuit-tracer attribution graph for it, and report concentration/completeness/ablation
    + the top feature nodes. Robust to circuit-tracer / transcoder unavailability (status, exit 0)."""
    # Pin the cave gate to the reference if importable; fall back to the module constant (same value).
    try:
        from rlhf_differential import MIN_EFFECT_NET as _MEN
        min_effect_net = float(_MEN)
    except Exception:
        min_effect_net = MIN_EFFECT_NET

    base = {"model": name, "device": device, "tag": tag, "cue": "cave_attribution_graph",
            "regime": "qa_base", "pool_size": len(pool),
            "metric": ("feature-level circuit-tracer attribution graph (NODES = GemmaScope-2b transcoder "
                       "features + error/residual + tokens + logits, EDGES = direct attributions) for the "
                       "realized caving logit-diff target = logit[W*-first-tok] - logit[C-first-tok] at the "
                       "answer slot, on ONE argmax-W* faithful caving instance; influence concentration, "
                       "completeness vs error nodes, and a top-k-vs-matched-random clamp-off ablation"),
            "thresholds": {"TOPK": TOPK, "SPARSE_FRAC": SPARSE_FRAC, "COMPLETE_TOL": COMPLETE_TOL,
                           "ABLATE_THR": ABLATE_THR, "RAND_SEED": RAND_SEED,
                           "MIN_EFFECT_NET": min_effect_net},
            "decision_rule": DECISION_RULE,
            "dependency": ("circuit-tracer (decoderesearch/circuit-tracer; pip install circuit-tracer) + "
                           "GemmaScope-2b transcoders for google/gemma-2-2b (~15GB); transformer_lens for "
                           "instance selection")}

    # ---- instance selection (TransformerLens forward) ----
    from transformer_lens import HookedTransformer
    print(f"[load] {name} on {device} (base, qa) for instance selection", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    instance, diag = _select_instance(model, pool, device, min_effect_net)
    base.update(diag)

    if instance is None:
        print(f"[instance] NO argmax-W* faithful caving item (n_cave={diag['n_cave']}, "
              f"n_argmaxW_cave={diag['n_argmaxW_cave']}). status=NO_FAITHFUL_INSTANCE.", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        base["status"] = "NO_FAITHFUL_INSTANCE"
        base["decision"] = {"category": "NO_FAITHFUL_INSTANCE", "msg": (
            "no argmax-W* caving item (model never realizes W* under pushback) -- no well-posed graph "
            "target; no attribution graph built.")}
        _write(base, tag)
        return

    # the COUNTER prompt as TEXT (so circuit-tracer re-tokenizes it with the SAME Q/A template) and the
    # base (un-ablated) realized caving logit-diff at the answer slot, for the ablation reference.
    tok = model.tokenizer
    instance["counter_text"] = tok.decode(instance["counter"][0], skip_special_tokens=False)
    with torch.no_grad():
        lg_c = model(instance["counter"])
    v = lg_c[0, -1].float()
    instance["target_logit_diff_base"] = float(v[instance["aid"]] - v[instance["cid"]])
    print(f"[instance] item {instance['i']} q={instance['q'][:50]!r} m_drop={instance['m_drop']:+.2f} "
          f"target(W*-C)={instance['target_logit_diff_base']:+.3f} W*={instance['Wstar'][:24]!r}", flush=True)
    base["instance"] = {k: instance[k] for k in ("i", "q", "correct", "Wstar", "cid", "aid", "M_neu",
                                                 "M_ctr", "m_drop", "ctr_argmax", "ctr_p_w", "ctr_p_c",
                                                 "target_logit_diff_base")}

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- attribution graph (circuit-tracer) ----
    status, stats = _attribute_graph(name, device, instance)
    if status != "OK":
        print(f"[tracer] status={status}: {stats.get('reason')}", flush=True)
        base["status"] = status
        base["tooling"] = stats
        base["decision"] = {"category": status, "msg": (
            "circuit-tracer / GemmaScope-2b transcoders unavailable or attribution failed -- "
            "no graph built; install circuit-tracer + the GemmaScope-2b transcoders on the box. "
            f"reason: {stats.get('reason')}")}
        _write(base, tag)
        return

    base["status"] = "OK"
    base["tooling"] = {"tried": stats.get("tried"), "transcoder_set": stats.get("transcoder_set"),
                       "target": stats.get("target")}
    report = _stats_to_decision(stats)
    base["graph"] = report
    base["decision"] = report["decision"]
    print(f"[graph] {report['decision']['category']} completeness={report['completeness']} "
          f"top{TOPK}_influence={report['topk_influence_frac']} "
          f"ablate_drop={report['topk_ablate_drop']} rand_drop={report['rand_ablate_drop']}", flush=True)
    print(f"[graph] {report['decision']['msg']}", flush=True)
    _write(base, tag)


def _write(out, tag):
    Path("out").mkdir(exist_ok=True)
    fn = f"out/cave_attribution_graph_{tag}.json"
    Path(fn).write_text(json.dumps(out, indent=2, default=str))
    print(f"[done] wrote {fn} (status={out.get('status')}, "
          f"decision={out.get('decision', {}).get('category')})", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, CPU, no tracer)
def _synthetic_graph(feat_influences, error_share, base_diff, ablate_drop, rand_drop):
    """Build a SYNTHETIC graph-stats dict (the shape _stats_to_decision consumes) from controlled inputs, so
    the selftest exercises the influence-concentration / completeness / ablation pipeline end-to-end with NO
    model and NO circuit-tracer. Pure. The synthetic dict stores EVERY node in feature_nodes (no truncation),
    so _stats_to_decision's full-list concentration is exact here even without all_feature_influence.
      feat_influences : per-feature-node influence list (signed); the |.|-sum is the feature subgraph weight.
      error_share     : fraction of the TOTAL (feature+error) influence that the error/residual nodes hold;
                        error_sum is set so completeness = |feat|/(|feat|+|err|) = 1 - error_share.
      base_diff       : the target caving logit-diff (positive = caved toward W*).
      ablate_drop     : the RELATIVE drop the top-k clamp produces -> topk_ablate_metric = base*(1-drop).
      rand_drop       : the relative drop the matched-random clamp produces -> rand_ablate_metric likewise."""
    feature_sum = float(sum(feat_influences))                # signed feature subgraph influence
    # completeness() divides by the SIGNED feature_sum, so base |err| on |feature_sum| (not the abs-sum) to make
    # completeness = |feat|/(|feat|+|err|) = 1 - error_share hold exactly.
    fsum_abs = abs(feature_sum)
    err_abs = fsum_abs * error_share / (1.0 - error_share) if error_share < 1.0 else fsum_abs * 1e9
    error_sum = err_abs                                      # signed error influence (sign irrelevant to |.|)
    nodes = [{"layer": 6 + (i % 5), "pos": 9, "feature": 1000 + i,
              "activation": 1.0 + 0.1 * i, "influence": float(v)}
             for i, v in enumerate(feat_influences)]
    return {"feature_nodes": nodes, "all_feature_influence": [float(v) for v in feat_influences],
            "n_feature_nodes": len(nodes),
            "feature_influence_sum": feature_sum, "error_influence_sum": error_sum,
            "target_logit_diff": base_diff,
            "topk_ablate_metric": base_diff * (1.0 - ablate_drop),
            "rand_ablate_metric": base_diff * (1.0 - rand_drop),
            "input_to_logit_path": ["input_tokens", "L6.feat1000@pos9", "logit[W*]"]}


def selftest():
    torch.manual_seed(0)

    # ---------- influence_fraction: concentration math, monotonicity, bounds ----------
    # five big nodes + many tiny ones; the top-5 carry almost all the |influence|.
    infl = [10.0, -9.0, 8.0, 7.0, 6.0] + [0.05] * 200
    f5, top5, total = influence_fraction(infl, topk=5)
    f15, _, _ = influence_fraction(infl, topk=15)
    assert f5 > 0.7 and 0.0 <= f5 <= 1.0, f5
    assert f15 >= f5 - 1e-9, (f5, f15)                         # monotone nondecreasing in topk
    assert set(top5) == {0, 1, 2, 3, 4}, top5                  # ranked by |influence|, sign-blind (node 1 in)
    assert abs(total - sum(abs(x) for x in infl)) < 1e-6
    # spread: 100 equal nodes -> top-15 carry 15/100 = 0.15
    spread = [1.0] * 100
    fsp, _, _ = influence_fraction(spread, topk=15)
    assert abs(fsp - 0.15) < 1e-9, fsp
    assert influence_fraction([], topk=5) == (0.0, [], 0.0)
    assert influence_fraction([0.0, 0.0], topk=5)[0] == 0.0     # all-zero -> 0 fraction, no crash
    print(f"[selftest] influence_fraction: top5={f5:.3f} (concentrated) monotone(top15>=top5) "
          f"spread100->top15={fsp:.3f} edge-cases OK")

    # ---------- completeness math + bounds ----------
    assert abs(completeness(8.0, 0.0) - 1.0) < 1e-9            # no error -> fully explained
    assert abs(completeness(0.0, 8.0) - 0.0) < 1e-9            # all error -> nothing explained
    assert abs(completeness(6.0, 6.0) - 0.5) < 1e-9            # equal -> 0.5 (the COMPLETE_TOL boundary)
    assert completeness(0.0, 0.0) == 0.0                       # both zero -> 0, no crash
    assert abs(completeness(-8.0, -2.0) - 0.8) < 1e-9          # uses magnitudes (sign-blind)
    print("[selftest] completeness: no-error=1, all-error=0, equal=0.5, magnitudes OK")

    # ---------- relative_drop math + clamps ----------
    assert abs(relative_drop(2.0, 0.5) - 0.75) < 1e-9          # 2.0 -> 0.5 is a 75% drop
    assert relative_drop(2.0, 3.0) == 0.0                      # ablation INCREASES metric -> 0 drop (clamp)
    assert relative_drop(2.0, 2.0) == 0.0                      # no change -> 0 drop
    assert relative_drop(0.0, -1.0) == 0.0                     # |base|~0 -> 0 (no metric to drop)
    assert abs(relative_drop(-2.0, -4.0) - 1.0) < 1e-9         # negative base driven further from 0 -> full drop
    assert relative_drop(-2.0, -0.5) == 0.0                    # negative base moved toward 0 (increase) -> clamp
    print("[selftest] relative_drop: 75% drop, increase->0, no-change->0, |base|~0->0, negative-base OK")

    # ============================================================ DECISION-BOUNDARY scenarios =============
    # (i) SPARSE_CIRCUIT: few high-influence nodes carry >= SPARSE_FRAC, top-k ablation crosses ABLATE_THR,
    #     matched-random does not, completeness high.
    di = decide_graph(complete_frac=0.9, topk_influence_frac=0.8, topk_ablate_drop=0.5, rand_ablate_drop=0.05)
    assert di["category"] == "SPARSE_CIRCUIT" and di["sparse_circuit"], di
    print(f"[selftest] (i) SPARSE_CIRCUIT: comp=0.9 top15=0.8 drop=0.5 rand=0.05 -> {di['category']}")

    # (ii) BROAD_DISTRIBUTED: completeness high but influence spread (top-TOPK < SPARSE_FRAC).
    dii = decide_graph(complete_frac=0.9, topk_influence_frac=0.2, topk_ablate_drop=0.5, rand_ablate_drop=0.0)
    assert dii["category"] == "BROAD_DISTRIBUTED" and dii["broad_distributed"] and not dii["concentrated"], dii
    # also BROAD when concentrated but the ablation is not causal (drop < ABLATE_THR)
    dii_b = decide_graph(0.9, 0.8, 0.05, 0.0)
    assert dii_b["category"] == "BROAD_DISTRIBUTED" and not dii_b["ablation_causal"], dii_b
    # also BROAD when concentrated + causal but a matched-random set matches it (not specific)
    dii_c = decide_graph(0.9, 0.8, 0.5, 0.5)
    assert dii_c["category"] == "BROAD_DISTRIBUTED" and not dii_c["rand_clean"], dii_c
    print(f"[selftest] (ii) BROAD_DISTRIBUTED: spread -> {dii['category']}; "
          f"not-causal -> {dii_b['category']}; not-specific(random matches) -> {dii_c['category']}")

    # (iii) INCOMPLETE: error/residual nodes dominate (completeness < COMPLETE_TOL) -- checked FIRST, even when
    #       the surviving features are concentrated + causal + specific.
    diii = decide_graph(complete_frac=0.3, topk_influence_frac=0.9, topk_ablate_drop=0.9, rand_ablate_drop=0.0)
    assert diii["category"] == "INCOMPLETE" and diii["incomplete"] and not diii["complete"], diii
    print(f"[selftest] (iii) INCOMPLETE: completeness 0.3 < {COMPLETE_TOL} (error nodes dominate) "
          f"-> {diii['category']} (checked before sparse/broad)")

    # ---------- threshold boundaries (all inclusive on >=) ----------
    assert decide_graph(COMPLETE_TOL, SPARSE_FRAC, ABLATE_THR, ABLATE_THR - 1e-9)["category"] == "SPARSE_CIRCUIT"
    assert decide_graph(COMPLETE_TOL - 1e-9, 0.9, 0.9, 0.0)["category"] == "INCOMPLETE"      # comp just below
    assert decide_graph(0.9, SPARSE_FRAC - 1e-9, 0.9, 0.0)["category"] == "BROAD_DISTRIBUTED"  # sparse just below
    assert decide_graph(0.9, 0.9, ABLATE_THR - 1e-9, 0.0)["category"] == "BROAD_DISTRIBUTED"   # drop just below
    assert decide_graph(0.9, 0.9, ABLATE_THR, ABLATE_THR)["category"] == "BROAD_DISTRIBUTED"   # rand == thr (not clean)
    # UNAVAILABLE on any None
    assert decide_graph(None, 0.9, 0.9, 0.0)["category"] == "UNAVAILABLE"
    assert decide_graph(0.9, 0.9, 0.9, None)["category"] == "UNAVAILABLE"
    print("[selftest] thresholds: SPARSE/INCOMPLETE/BROAD boundaries (inclusive >=) + UNAVAILABLE OK")

    # ============================================================ END-TO-END synthetic graphs =============
    # Run the FULL _stats_to_decision pipeline (influence_fraction + completeness + relative_drop + decide)
    # on synthetic graph-stats dicts, exactly as the real run feeds the tracer output.
    # (A) SPARSE: 3 dominant feature nodes (carry >= SPARSE_FRAC), high completeness, top-k clamp drops the
    #     caving logit-diff hard, matched-random barely moves it.
    sA = _synthetic_graph(feat_influences=[6.0, -5.5, 5.0] + [0.05] * 60, error_share=0.1,
                          base_diff=3.0, ablate_drop=0.6, rand_drop=0.04)
    rA = _stats_to_decision(sA)
    assert rA["decision"]["category"] == "SPARSE_CIRCUIT", (rA["decision"], rA)
    assert rA["completeness"] >= COMPLETE_TOL and rA["topk_influence_frac"] >= SPARSE_FRAC
    assert rA["topk_ablate_drop"] >= ABLATE_THR > rA["rand_ablate_drop"]
    print(f"[selftest] (A) end-to-end SPARSE: comp={rA['completeness']} top{TOPK}={rA['topk_influence_frac']} "
          f"drop={rA['topk_ablate_drop']} rand={rA['rand_ablate_drop']} -> {rA['decision']['category']}")

    # (B) BROAD: influence spread over 80 ~equal feature nodes (top-15 < SPARSE_FRAC), high completeness.
    sB = _synthetic_graph(feat_influences=[1.0] * 80, error_share=0.1,
                          base_diff=3.0, ablate_drop=0.15, rand_drop=0.1)
    rB = _stats_to_decision(sB)
    assert rB["decision"]["category"] == "BROAD_DISTRIBUTED", (rB["decision"], rB)
    assert abs(rB["topk_influence_frac"] - TOPK / 80) < 1e-6   # 15/80 ~ 0.1875 < SPARSE_FRAC
    print(f"[selftest] (B) end-to-end BROAD: top{TOPK}={rB['topk_influence_frac']:.4f} "
          f"(={TOPK}/80) < {SPARSE_FRAC} -> {rB['decision']['category']}")

    # (C) INCOMPLETE: the same concentrated, causal feature set BUT error/residual nodes hold 70% of the
    #     influence -> completeness 0.3 < COMPLETE_TOL -> INCOMPLETE regardless.
    sC = _synthetic_graph(feat_influences=[6.0, -5.5, 5.0] + [0.05] * 60, error_share=0.7,
                          base_diff=3.0, ablate_drop=0.6, rand_drop=0.04)
    rC = _stats_to_decision(sC)
    assert rC["decision"]["category"] == "INCOMPLETE", (rC["decision"], rC)
    assert abs(rC["completeness"] - 0.3) < 1e-6, rC["completeness"]
    print(f"[selftest] (C) end-to-end INCOMPLETE: completeness={rC['completeness']} < {COMPLETE_TOL} "
          f"(error nodes dominate) -> {rC['decision']['category']}")

    # (D) geometry-only: ablation metrics absent (clamp API unavailable) -> no crash, NO SPARSE_CIRCUIT claim;
    #     concentrated geometry -> SPARSE_GEOMETRY_CAUSAL_UNCONFIRMED; completeness still gates INCOMPLETE.
    sD = dict(sA); sD["topk_ablate_metric"] = None; sD["rand_ablate_metric"] = None; sD["ablation_note"] = "no clamp"
    rD = _stats_to_decision(sD)
    assert rD["decision"]["category"] == "SPARSE_GEOMETRY_CAUSAL_UNCONFIRMED", rD["decision"]
    assert rD["decision"]["causal_confirmed"] is False and rD["decision"]["sparse_circuit"] is False
    assert rD["topk_ablate_drop"] is None and rD["topk_ablate_metric"] is None
    sE = dict(sB); sE["topk_ablate_metric"] = None; sE["rand_ablate_metric"] = None
    assert _stats_to_decision(sE)["decision"]["category"] == "BROAD_DISTRIBUTED"
    sF = dict(sC); sF["topk_ablate_metric"] = None; sF["rand_ablate_metric"] = None
    assert _stats_to_decision(sF)["decision"]["category"] == "INCOMPLETE"
    print(f"[selftest] (D) geometry-only (no ablation): {rD['decision']['category']} / BROAD / INCOMPLETE OK")

    # path + top-node reporting survive the pipeline; top list capped at TOPK
    assert rA["input_to_logit_path"] and rA["top_feature_nodes"] and len(rA["top_feature_nodes"]) <= TOPK
    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name", default=MODEL_DEFAULT,
                   help="base model (gemma-2-2b); circuit-tracer fully supports 2b w/ GemmaScope transcoders")
    p.add_argument("--tag", default="2b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        from misconception_pool import ITEMS_WIDE
        run(args.name, args.tag, args.device, ITEMS_WIDE)


if __name__ == "__main__":
    main()
