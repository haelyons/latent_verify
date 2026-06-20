"""Attribution of the early-layer logit-lens margin gap (it vs base, neutral turn) to WEIGHTS vs two instrument confounds, on the matched both-models-know set.

CONTEXT. The sibling controls/logit_lens_margin_matched.py found a large it-minus-base early-layer logit-lens
margin gap on the NEUTRAL turn (it favors the correct first token early in the stack, base does not), on the
matched intersection where BOTH gemma-2-9b base and -it are single-turn correct (neutral final-layer margin>0).
That early gap could be a property of the model WEIGHTS, or an artifact of the instrument/setup. This control
holds the readout fixed (the same ln_final + W_U + b_U + gemma-2 final-logit softcap last-position logit-lens,
reused verbatim) and crosses the early gap against two confounds, plus characterizes the trajectory, all on the
matched set, reporting numbers + neutral categories only and asserting nothing about which factor should win:
  A. FORMAT CROSSOVER (neutral turn). _helpers takes an is_chat flag that controls prompt FORMAT only (chat
     template vs raw Q/A). For EACH model build the neutral prompt in BOTH formats -- native (base->Q/A,
     it->chat) and cross (base->chat, it->Q/A) -- and read the L0-excluded early-third margin under each. If the
     early gap is FORMAT-driven the cross-format conditions converge; if WEIGHT-driven each model keeps its own
     early margin regardless of format.
  B. CROSS-LENS SWAP (neutral, native format). Hold both models' final transforms (ln_final, W_U, b_U, softcap).
     Recompute each model's early margin from its OWN native-format resid_post projected through (i) its own
     transform and (ii) the OTHER model's transform. If the gap is a lens-calibration artifact it shrinks when
     both residuals are read through the SAME lens; if it is residual-stream geometry it persists.
  C. FAITHFULNESS / TRAJECTORY (neutral, native, own lens). Per layer, fraction of matched items whose full-vocab
     logit-lens argmax equals the model's OWN final-layer greedy argmax token (does the early lens already point
     at the eventual output?), plus the full mean margin trajectory per layer and the layer of max mean margin.
  D. FRACTIONAL EROSION (both turns, native). Per item frac_erosion(model) = (late_margin(neutral) -
     late_margin(challenge)) / max(late_margin(neutral), eps); paired bootstrap CI of frac_erosion(base) -
     frac_erosion(it) (normalizes the baseline-magnitude difference between models).

NEUTRAL DECISION (module-constant thresholds; numbers + categories only, no claim attached to any sign):
  FORMAT: report early_margin per (model, format); the it-base early_diff under NATIVE, under BOTH-CHAT (it
    native vs base cross) and under BOTH-QA (it cross vs base native), each a paired-bootstrap CI + ci_excludes_zero.
    Per same-format comparison: GAP_PERSISTS if its CI excludes 0 and keeps the native sign, else GAP_COLLAPSES.
  CROSS-LENS: it-base early_diff with BOTH residuals read through base's lens, and through it's lens;
    GAP_PERSISTS / GAP_COLLAPSES per the CI (excludes 0 and keeps native sign).
  FAITHFULNESS: per model the early-third mean argmax-match fraction vs the final layer (1.0 by construction);
    EARLY_FAITHFUL if early argmax-match >= FAITH_TOL else EARLY_UNFAITHFUL.
  FRACTIONAL EROSION: frac_erosion_diff (base-it) mean + CI + SIGNIFICANT/NULL + sign.

Run model-free selftest (no model load, CPU):
    python controls/logit_lens_attribution.py --selftest
Run the measurement (9b; needs the GPU box):
    python controls/logit_lens_attribution.py --device cuda --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

# Repo-internal imports (item pool, turn builders) are deferred into the functions that use them so
# --selftest runs standalone from controls/ without the rest of the repo on sys.path; on the box the
# files are scp'd flat into latent_verify/, where these resolve.

# Pre-registered thresholds (see module docstring). Neutral: stated on the measured margin only.
KNOW_GATE = 0.0       # keep an item iff its NEUTRAL final-layer margin > this FOR BOTH MODELS (both know the fact)
FAITH_TOL = 0.5       # early-third mean full-vocab argmax-match fraction >= this -> EARLY_FAITHFUL, else EARLY_UNFAITHFUL
EROSION_EPS = 0.1     # floor on the neutral late_margin denominator of frac_erosion (avoids div-by-tiny)
N_BOOT = 1000         # paired bootstrap resamples
SEED = 0              # fixed seed -> reproducible CIs
CI_LO, CI_HI = 2.5, 97.5   # 95 percent percentile interval

DECISION_RULE = (
    "matched set = items with NEUTRAL final-layer margin > 0 for BOTH base and it. Early metric = L0-excluded "
    "first-third mean margin. (A) FORMAT: it-base early_diff under NATIVE (base QA, it chat), BOTH-CHAT (it "
    "native vs base cross), BOTH-QA (it cross vs base native); per same-format comparison GAP_PERSISTS if its 95 "
    "percent paired-bootstrap CI (B=1000, seed=0) excludes 0 and keeps the native sign, else GAP_COLLAPSES. (B) "
    "CROSS-LENS: it-base early_diff with both native residuals through base's lens, and through it's lens; "
    "GAP_PERSISTS / GAP_COLLAPSES per the CI. (C) FAITHFULNESS: per model early-third mean full-vocab argmax-match "
    "fraction vs final layer (1.0); EARLY_FAITHFUL if >=0.5 else EARLY_UNFAITHFUL. (D) FRACTIONAL EROSION: "
    "frac_erosion(model)=(late_neutral-late_challenge)/max(late_neutral,0.1); frac_erosion_diff(base-it) mean + "
    "CI + SIGNIFICANT/NULL + sign. Numbers + categories only; no claim attached to any sign."
)

TURNS = ("neutral", "challenge")
MODELS = ("base", "it")
FORMATS = ("native", "cross")          # native: base->QA(is_chat=False), it->chat(is_chat=True); cross flips both
LENSES = ("base", "it")


# --------------------------------------------------------------------------- pure margin / readout logic (verbatim from sibling)
def softcap(logits, cap):
    """gemma-2 final-logit softcap, matching the last-position readout in rlhf_differential._atp_net.M_last:
    cap * tanh(logits / cap). cap falsy (None/0) -> identity. Pure (tensor, scalar -> tensor)."""
    if cap:
        return cap * torch.tanh(logits / cap)
    return logits


def early_margin_no_l0(margins):
    """L0-EXCLUDED early metric: mean of margin_L over the first third of layers EXCLUDING layer 0.
    Layer 0 is dominated by the embedding/unembedding bias, not computation, so it is dropped. Pure
    (list -> float). The first third is taken over the FULL layer count, then layer 0 is removed; if only
    layer 0 falls in the first third, fall back to the next available layer so the metric is never empty."""
    n = len(margins)
    third = max(1, n // 3)
    window = margins[1:third]                              # first third, layer 0 excluded
    if not window:
        window = margins[1:2] if n > 1 else margins[:1]    # degenerate-short fallback (still L0-excluded if possible)
    return statistics.mean(window)


def late_margin(margins):
    """late_margin = the final-layer margin. Pure (list -> float)."""
    return margins[-1]


def frac_erosion(late_neutral, late_challenge, eps=EROSION_EPS):
    """(late_margin(neutral) - late_margin(challenge)) / max(late_margin(neutral), eps). Normalizes the
    erosion by the baseline neutral magnitude so a model with a larger absolute neutral margin is not
    counted as eroding more just for being bigger. Pure (float, float -> float)."""
    return (late_neutral - late_challenge) / max(late_neutral, eps)


def sign_of(mean, tol=1e-9):
    """POS / NEG / ZERO sign of a mean. Pure."""
    if mean > tol:
        return "POS"
    if mean < -tol:
        return "NEG"
    return "ZERO"


def early_fraction(frac_curve):
    """Early-third (L0-excluded) mean of a per-layer fraction curve (e.g. argmax-match). Reuses the exact
    L0-excluded first-third window of early_margin_no_l0. Pure (list -> float)."""
    return early_margin_no_l0(frac_curve)


def faithful_tag(early_frac, tol=FAITH_TOL):
    """EARLY_FAITHFUL if the early-third mean argmax-match fraction clears tol, else EARLY_UNFAITHFUL. Pure."""
    return "EARLY_FAITHFUL" if early_frac >= tol else "EARLY_UNFAITHFUL"


def argmax_layer(margins):
    """Index into the layer list of the maximum mean margin (the peak of the trajectory). Pure (list -> int)."""
    return max(range(len(margins)), key=lambda j: margins[j])


# --------------------------------------------------------------------------- matched intersection (verbatim from sibling)
def matched(base_neutral_late, it_neutral_late, know_gate=KNOW_GATE):
    """Item indices the matched set keeps: items present for BOTH models whose NEUTRAL final-layer margin
    exceeds know_gate for BOTH (both models know the fact single-turn). Pure (dict, dict -> sorted list)."""
    return sorted(i for i in base_neutral_late
                  if i in it_neutral_late and base_neutral_late[i] > know_gate and it_neutral_late[i] > know_gate)


# --------------------------------------------------------------------------- paired bootstrap CI (verbatim from sibling)
def paired_bootstrap_ci(values, seed=SEED, n_boot=N_BOOT, lo=CI_LO, hi=CI_HI):
    """Percentile paired-bootstrap CI of the mean of per-item paired differentials. The values list is
    ALREADY per-item differences (aligned by item), so resampling rows = a paired bootstrap. Pure
    (list -> dict); deterministic via random.Random(seed)."""
    import random as _r
    n = len(values)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = _r.Random(seed)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    def pct(p):
        return means[min(n_boot - 1, max(0, int(round(p / 100 * (n_boot - 1)))))]
    return {"mean": round(sum(values) / n, 4), "lo": round(pct(lo), 4), "hi": round(pct(hi), 4), "n": n}


def ci_excludes_zero(ci):
    """True iff the CI lies entirely above or entirely below 0. Pure (dict -> bool)."""
    if ci["lo"] is None or ci["hi"] is None:
        return False
    return (ci["lo"] > 0 and ci["hi"] > 0) or (ci["lo"] < 0 and ci["hi"] < 0)


def classify_diff(values):
    """Build a differential's CI + SIGNIFICANT/NULL + sign report from per-item paired values. Pure."""
    ci = paired_bootstrap_ci(values)
    exc = ci_excludes_zero(ci)
    return {"mean": ci["mean"], "ci_lo": ci["lo"], "ci_hi": ci["hi"], "n": ci["n"],
            "ci_excludes_zero": exc, "significance": "SIGNIFICANT" if exc else "NULL",
            "sign": sign_of(ci["mean"]) if ci["mean"] is not None else "ZERO"}


def gap_category(values, native_sign):
    """GAP_PERSISTS if the paired-bootstrap CI of these per-item it-base early diffs excludes 0 AND keeps the
    native sign (i.e. the early gap survives the format-matched / cross-lens condition), else GAP_COLLAPSES.
    Returns the classify_diff report augmented with the category. Pure."""
    rep = classify_diff(values)
    persists = rep["ci_excludes_zero"] and rep["sign"] == native_sign and native_sign != "ZERO"
    rep["native_sign"] = native_sign
    rep["category"] = "GAP_PERSISTS" if persists else "GAP_COLLAPSES"
    return rep


# --------------------------------------------------------------------------- real per-layer readout
def _layer_readout(model, ids, cid, aid, layers, cap, lens=None):
    """One forward pass; capture the last-position resid_post per layer; then for each layer L in `layers`
    project it through a LENS (ln_final, W_U, b_U, cap) -- the model's OWN by default, or a supplied
    (ln_final_callable, W_U, b_U, cap) tuple -- with the gemma softcap, and return for each layer:
      margin_L = logit_L(cid) - logit_L(aid)   and   argmax_L = argmax over the FULL vocab.
    The full-vocab argmax is needed for the faithfulness curve; the scalar margin path is identical to the
    sibling's _layer_margins. Memory-lean: caches only the last-position resid per layer (no [seq, d_vocab]).
    Returns (margins, argmaxes, final_argmax)."""
    nL = model.cfg.n_layers
    resid = {}

    def grab(r, hook, _resid=resid):
        _resid[hook.layer()] = r[0, -1].detach()
        return r

    names = [f"blocks.{L}.hook_resid_post" for L in range(nL)]
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(n, grab) for n in names], return_type=None)

    if lens is None:
        ln_final, W_U, b_U, lens_cap = model.ln_final, model.W_U, model.b_U, cap
    else:
        ln_final, W_U, b_U, lens_cap = lens

    margins, argmaxes, final_argmax = [], [], None
    for L in layers:
        h = ln_final(resid[L].unsqueeze(0).to(W_U.device))[0]   # [d_model], final LN exactly as applied
        logits = h @ W_U + b_U                                  # [d_vocab]
        logits = softcap(logits.float(), lens_cap)
        margins.append(float(logits[cid] - logits[aid]))
        am = int(torch.argmax(logits))
        argmaxes.append(am)
        final_argmax = am                                       # last layer in `layers` is the final layer
    return margins, argmaxes, final_argmax


def _extract_lens(model, cap):
    """Pull a portable lens (ln_final callable, W_U, b_U, cap) for cross-lens projection, moved to CPU so
    BOTH models' unembeds can be held simultaneously. The ln_final module is deep-copied to CPU; W_U/b_U are
    detached CPU clones. PEAK-MEMORY NOTE: this materializes a SECOND [d_model, d_vocab] unembed on CPU per
    model (~0.5GB bf16 each on 9b), held for the duration of block B -- budget for two unembeds plus the live
    GPU model. Returns the lens tuple."""
    import copy
    lf = copy.deepcopy(model.ln_final).to("cpu").eval()
    W_U = model.W_U.detach().to("cpu").clone()
    b_U = model.b_U.detach().to("cpu").clone()
    return (lf, W_U, b_U, cap)


def _model_pass(name, is_chat, device, layers_stride, pool, want_cross_format, keep_lens, chat_template=None):
    """Load one model. For its NATIVE format (is_chat) build the neutral + challenge prompts (verbatim repo
    construction) and run the per-layer readout (margins + full-vocab argmax + final argmax). If
    want_cross_format, ALSO build the NEUTRAL prompt in the OPPOSITE format (is_chat flipped) and run the
    neutral readout there too (block A cross condition). If keep_lens, extract a CPU lens (block B). Stores
    per-item native resids' margins/argmaxes; the matched gate is taken AFTER both models run.
    Returns per-item readouts + the layer list + (optional) the cross-format neutral margins + lens."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import _helpers
    from job_truthful_flip import PUSH, NEUTRAL
    tag = "it" if is_chat else "base"
    print(f"[load] {name} on {device} (native chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    if chat_template and not getattr(model.tokenizer, "chat_template", None):
        model.tokenizer.chat_template = chat_template   # base tokenizer lacks one; borrow so the cross (chat) format builds
    nL = model.cfg.n_layers
    cap = getattr(model.cfg, "final_logit_softcap", None)
    layers = list(range(0, nL, max(1, layers_stride)))
    if layers[-1] != nL - 1:
        layers.append(nL - 1)                              # always keep the final layer (late_margin / gate / final argmax)

    # native-format builders
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    # cross-format builders (is_chat flipped) -- prompt FORMAT only; same items, same readout
    raw_x, single_x, push_x, first_x, num_lp_x = _helpers(model, device, not is_chat)

    per_item = {}                                          # pool idx -> dict of native readouts
    cross_neutral_early = {}                               # pool idx -> cross-format neutral margins (block A)
    neutral_late = {}                                      # pool idx -> NATIVE neutral final-layer margin (the gate)
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)          # first-token correct-vs-misconception pair (native tokenizer)
        if cid == aid:                                     # first-token collision -> margin meaningless, skip
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        m_neu, am_neu, fa_neu = _layer_readout(model, neutral, cid, aid, layers, cap)
        m_chal, _, _ = _layer_readout(model, counter, cid, aid, layers, cap)
        per_item[i] = {"neutral": m_neu, "challenge": m_chal, "neutral_argmax": am_neu, "final_argmax": fa_neu}
        neutral_late[i] = m_neu[-1]
        if want_cross_format:
            neutral_x = push_x(q, C, NEUTRAL)              # SAME content, OPPOSITE prompt format
            m_neu_x, _, _ = _layer_readout(model, neutral_x, cid, aid, layers, cap)
            cross_neutral_early[i] = m_neu_x
        print(f"  [{tag}] item {i} margin_final neu={m_neu[-1]:+.2f} chal={m_chal[-1]:+.2f} "
              f"q={q[:40]!r}", flush=True)

    lens = _extract_lens(model, cap) if keep_lens else None
    # capture per-item native resids ONLY if a cross-lens reprojection is needed; we re-run forward in block B
    # to avoid holding [n_items, n_layers, d_model] resids. Store nothing heavy here.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"per_item": per_item, "neutral_late": neutral_late, "n_layers": nL, "layers": layers,
            "cross_neutral_early": cross_neutral_early, "lens": lens,
            "name": name, "is_chat": is_chat}


def _crosslens_pass(name, is_chat, device, layers, cap, idxs, pool, base_lens, it_lens):
    """Block B: reload ONE model, and for each matched item re-run its NATIVE neutral readout TWICE -- once
    through base's lens and once through it's lens (both held on CPU) -- returning per-item early margins under
    each lens. Reloading (vs caching resids) keeps peak memory at one live GPU model + two CPU unembeds.
    Returns {idx: {"base_lens": early, "it_lens": early}}."""
    from transformer_lens import HookedTransformer
    from rlhf_differential import _helpers
    from job_truthful_flip import NEUTRAL
    tag = "it" if is_chat else "base"
    print(f"[crosslens-load] {name} (chat={is_chat}) re-run for own-vs-other lens", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    out = {}
    for i in idxs:
        it = pool[i]
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)
        neutral = push(q, C, NEUTRAL)
        m_b, _, _ = _layer_readout(model, neutral, cid, aid, layers, cap, lens=base_lens)
        m_i, _, _ = _layer_readout(model, neutral, cid, aid, layers, cap, lens=it_lens)
        out[i] = {"base_lens": early_margin_no_l0(m_b), "it_lens": early_margin_no_l0(m_i)}
        print(f"  [{tag} xlens] item {i} early(base-lens)={out[i]['base_lens']:+.2f} "
              f"early(it-lens)={out[i]['it_lens']:+.2f}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _mean_curve(per_item, idxs, key, nseg):
    """Mean per-layer curve (margins or 0/1 argmax-match) over the matched indices. Pure over per_item."""
    if not idxs:
        return [0.0] * nseg
    acc = [0.0] * nseg
    for i in idxs:
        v = per_item[i][key]
        for j in range(nseg):
            acc[j] += v[j]
    return [x / len(idxs) for x in acc]


def _argmatch_curve(per_item, idxs, nseg):
    """Per-layer fraction of matched items whose NEUTRAL full-vocab logit-lens argmax == the model's OWN
    final-layer greedy argmax token. The final layer is 1.0 by construction. Pure over per_item."""
    if not idxs:
        return [0.0] * nseg
    curve = [0.0] * nseg
    for i in idxs:
        ams = per_item[i]["neutral_argmax"]
        fa = per_item[i]["final_argmax"]
        for j in range(nseg):
            curve[j] += 1.0 if ams[j] == fa else 0.0
    return [x / len(idxs) for x in curve]


def run(name_base, name_it, tag, device, layers_stride, pool, skip_crosslens=True):
    from transformers import AutoTokenizer
    chat_template = AutoTokenizer.from_pretrained(name_it).chat_template   # gemma-2 chat template (base tokenizer lacks one)
    # block A needs cross-format neutral; block B needs both lenses extracted
    res = {"base": _model_pass(name_base, False, device, layers_stride, pool,
                               want_cross_format=True, keep_lens=True, chat_template=chat_template),
           "it": _model_pass(name_it, True, device, layers_stride, pool,
                             want_cross_format=True, keep_lens=True, chat_template=chat_template)}
    nL = res["it"]["n_layers"]
    layers = res["it"]["layers"]
    nseg = len(layers)
    assert res["base"]["layers"] == layers, "base/it layer index lists differ (stride mismatch)"

    idxs = matched(res["base"]["neutral_late"], res["it"]["neutral_late"])
    n_matched = len(idxs)
    would_keep_alone = {m: sum(1 for v in res[m]["neutral_late"].values() if v > KNOW_GATE) for m in MODELS}
    print(f"[matched] {n_matched}/{len(pool)} items known single-turn by BOTH (neutral final margin>"
          f"{KNOW_GATE}): {idxs}", flush=True)

    # ---- block A: FORMAT CROSSOVER (per-item early margins under native & cross format) ----
    fmt_early = {m: {"native": {}, "cross": {}} for m in MODELS}
    for m in MODELS:
        for i in idxs:
            fmt_early[m]["native"][i] = early_margin_no_l0(res[m]["per_item"][i]["neutral"])
            fmt_early[m]["cross"][i] = early_margin_no_l0(res[m]["cross_neutral_early"][i])
    fmt_early_mean = {m: {f: round(statistics.mean([fmt_early[m][f][i] for i in idxs]), 4) if idxs else None
                          for f in FORMATS} for m in MODELS}
    # native it-base early diff (the original gap), then format-matched comparisons
    native_diff_vals = [fmt_early["it"]["native"][i] - fmt_early["base"]["native"][i] for i in idxs]
    native_rep = classify_diff(native_diff_vals)
    native_sign = native_rep["sign"]
    both_chat_vals = [fmt_early["it"]["native"][i] - fmt_early["base"]["cross"][i] for i in idxs]   # both chat
    both_qa_vals = [fmt_early["it"]["cross"][i] - fmt_early["base"]["native"][i] for i in idxs]      # both QA
    format_block = {
        "early_margin_mean": fmt_early_mean,
        "native_it_minus_base": {"definition": "early(it native) - early(base native) [the original gap]", **native_rep},
        "both_chat_it_minus_base": {"definition": "early(it native=chat) - early(base cross=chat) [format matched on chat]",
                                    **gap_category(both_chat_vals, native_sign)},
        "both_qa_it_minus_base": {"definition": "early(it cross=QA) - early(base native=QA) [format matched on QA]",
                                  **gap_category(both_qa_vals, native_sign)},
    }

    # ---- block B: CROSS-LENS SWAP (re-run each model through base's lens and it's lens) ----
    # GATED: the full-vocab CPU re-projection is ~3 min/item (reload + 256k-wide CPU matmul per layer).
    # Default-skipped; a separate run with --crosslens enables it. (An 11-item partial preview already
    # showed base-lens approx it-lens, so calibration is not the driver of the early gap.)
    if skip_crosslens:
        crosslens_block = {"status": "SKIPPED",
                           "note": "cross-lens full-vocab CPU re-projection skipped for speed; enable with --crosslens"}
    else:
        base_lens, it_lens = res["base"]["lens"], res["it"]["lens"]
        cap_base = base_lens[3]
        cap_it = it_lens[3]
        base_xl = _crosslens_pass(name_base, False, device, layers, cap_base, idxs, pool, base_lens, it_lens)
        it_xl = _crosslens_pass(name_it, True, device, layers, cap_it, idxs, pool, base_lens, it_lens)
        # it-base early diff with BOTH residuals read through the SAME lens
        base_lens_vals = [it_xl[i]["base_lens"] - base_xl[i]["base_lens"] for i in idxs]   # both through base's lens
        it_lens_vals = [it_xl[i]["it_lens"] - base_xl[i]["it_lens"] for i in idxs]          # both through it's lens
        crosslens_block = {
            "through_base_lens_it_minus_base": {"definition": "early(it resid|base lens) - early(base resid|base lens)",
                                                **gap_category(base_lens_vals, native_sign)},
            "through_it_lens_it_minus_base": {"definition": "early(it resid|it lens) - early(base resid|it lens)",
                                              **gap_category(it_lens_vals, native_sign)},
            "own_lens_means": {
                "base": {"own": round(statistics.mean([base_xl[i]["base_lens"] for i in idxs]), 4) if idxs else None,
                         "other": round(statistics.mean([base_xl[i]["it_lens"] for i in idxs]), 4) if idxs else None},
                "it": {"own": round(statistics.mean([it_xl[i]["it_lens"] for i in idxs]), 4) if idxs else None,
                       "other": round(statistics.mean([it_xl[i]["base_lens"] for i in idxs]), 4) if idxs else None},
            },
        }

    # ---- block C: FAITHFULNESS / TRAJECTORY ----
    faith = {}
    traj = {}
    for m in MODELS:
        argcurve = _argmatch_curve(res[m]["per_item"], idxs, nseg)
        margin_traj = _mean_curve(res[m]["per_item"], idxs, "neutral", nseg)
        early_f = early_fraction(argcurve)
        peak_j = argmax_layer(margin_traj) if margin_traj else 0
        faith[m] = {"argmatch_curve": [round(x, 4) for x in argcurve],
                    "early_third_argmatch": round(early_f, 4),
                    "final_layer_argmatch": round(argcurve[-1], 4),
                    "faithfulness": faithful_tag(early_f)}
        traj[m] = {"mean_margin_trajectory": [round(x, 4) for x in margin_traj],
                   "layer_of_max_mean_margin": layers[peak_j], "peak_index": peak_j,
                   "max_mean_margin": round(margin_traj[peak_j], 4) if margin_traj else None}

    # ---- block D: FRACTIONAL EROSION ----
    raw_margins = {m: {"neutral": {}, "challenge": {}} for m in MODELS}
    for m in MODELS:
        for i in idxs:
            raw_margins[m]["neutral"][i] = late_margin(res[m]["per_item"][i]["neutral"])
            raw_margins[m]["challenge"][i] = late_margin(res[m]["per_item"][i]["challenge"])
    frac_diff_vals = []
    frac_by_model = {m: [] for m in MODELS}
    for i in idxs:
        fb = frac_erosion(raw_margins["base"]["neutral"][i], raw_margins["base"]["challenge"][i])
        fi = frac_erosion(raw_margins["it"]["neutral"][i], raw_margins["it"]["challenge"][i])
        frac_by_model["base"].append(fb)
        frac_by_model["it"].append(fi)
        frac_diff_vals.append(fb - fi)                                  # frac_erosion(base) - frac_erosion(it)
    erosion_block = {
        "frac_erosion_mean": {m: round(statistics.mean(frac_by_model[m]), 4) if idxs else None for m in MODELS},
        "frac_erosion_diff_base_minus_it": {"definition": "frac_erosion(base) - frac_erosion(it) [normalized by neutral magnitude]",
                                            **classify_diff(frac_diff_vals)},
    }

    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "logit_lens_attribution", "pool_size": len(pool),
        "n_layers": nL, "layers": layers, "layers_stride": layers_stride,
        "thresholds": {"KNOW_GATE": KNOW_GATE, "FAITH_TOL": FAITH_TOL, "EROSION_EPS": EROSION_EPS,
                       "N_BOOT": N_BOOT, "SEED": SEED, "CI_LO": CI_LO, "CI_HI": CI_HI},
        "decision_rule": DECISION_RULE,
        "n_matched": n_matched, "matched_idxs": idxs, "would_keep_alone": would_keep_alone,
        "A_format_crossover": format_block,
        "B_cross_lens": crosslens_block,
        "C_faithfulness_trajectory": {"faithfulness": faith, "trajectory": traj},
        "D_fractional_erosion": erosion_block,
        "per_item": {
            "native_early": {m: [round(fmt_early[m]["native"][i], 4) for i in idxs] for m in MODELS},
            "cross_early": {m: [round(fmt_early[m]["cross"][i], 4) for i in idxs] for m in MODELS},
            "frac_erosion": {m: [round(x, 4) for x in frac_by_model[m]] for m in MODELS},
        },
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/logit_lens_attribution_{tag}.json").write_text(json.dumps(out, indent=2))

    # ---- summary lines (one per block) ----
    print(f"[matched] n_matched={n_matched} | native it-base early_diff mean={native_rep['mean']} "
          f"CI=[{native_rep['ci_lo']}, {native_rep['ci_hi']}] sign={native_sign}", flush=True)
    bc, bq = format_block["both_chat_it_minus_base"], format_block["both_qa_it_minus_base"]
    print(f"[A format] early_means={fmt_early_mean} | both-chat {bc['mean']} CI=[{bc['ci_lo']},{bc['ci_hi']}] "
          f"-> {bc['category']} | both-qa {bq['mean']} CI=[{bq['ci_lo']},{bq['ci_hi']}] -> {bq['category']}", flush=True)
    if crosslens_block.get("status") == "SKIPPED":
        print("[B crosslens] SKIPPED (run with --crosslens to enable)", flush=True)
    else:
        lb, li = crosslens_block["through_base_lens_it_minus_base"], crosslens_block["through_it_lens_it_minus_base"]
        print(f"[B crosslens] base-lens {lb['mean']} CI=[{lb['ci_lo']},{lb['ci_hi']}] -> {lb['category']} | "
              f"it-lens {li['mean']} CI=[{li['ci_lo']},{li['ci_hi']}] -> {li['category']}", flush=True)
    print(f"[C faithful] base early_argmatch={faith['base']['early_third_argmatch']} -> {faith['base']['faithfulness']} "
          f"(peak L{traj['base']['layer_of_max_mean_margin']}) | it early_argmatch={faith['it']['early_third_argmatch']} "
          f"-> {faith['it']['faithfulness']} (peak L{traj['it']['layer_of_max_mean_margin']})", flush=True)
    fr = erosion_block["frac_erosion_diff_base_minus_it"]
    print(f"[D frac_erosion] base-it mean={fr['mean']} CI=[{fr['ci_lo']},{fr['ci_hi']}] -> {fr['significance']} "
          f"sign={fr['sign']}", flush=True)
    print(f"[done] wrote out/logit_lens_attribution_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free, synthetic, CPU)
def selftest():
    # softcap helper matches cap*tanh(logits/cap); falsy cap -> identity
    v = torch.tensor([0.0, 30.0, -30.0, 5.0])
    assert torch.allclose(softcap(v, 10.0), 10.0 * torch.tanh(v / 10.0)), softcap(v, 10.0)
    assert torch.allclose(softcap(v, None), v) and torch.allclose(softcap(v, 0), v)
    print("[selftest] softcap matches cap*tanh(logits/cap), identity on falsy cap")

    # matched intersection reused (kept iff present AND above gate FOR BOTH models)
    base_late = {0: 2.0, 1: 0.5, 2: -0.3, 3: 1.0, 4: 3.0}   # item 2 below gate for base
    it_late = {0: 1.5, 1: -0.1, 2: 0.8, 3: 2.0, 5: 4.0}     # item 1 below gate for it; 5 base-absent; 4 it-absent
    assert matched(base_late, it_late) == [0, 3], matched(base_late, it_late)
    print(f"[selftest] matched intersection reused = {matched(base_late, it_late)}")

    # CROSS-LENS projection arithmetic: a residual read through two synthetic unembeds gives expected margins.
    # Two-token vocab; cid=0, aid=1. ln_final = identity. Lens A favors token0 (margin>0), lens B favors token1.
    ident = lambda x: x                                     # ln_final identity for the synthetic check
    d_model, d_vocab = 3, 2
    resid = torch.tensor([[1.0, 0.0, 0.0]])                 # last-position resid (already [1, d_model])
    W_A = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])   # token0 logit = 2*x0, token1 = 0
    W_B = torch.tensor([[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])   # token0 = 0, token1 = 2*x0
    b0 = torch.zeros(d_vocab)
    cid, aid = 0, 1
    # project manually through each lens (mirror _layer_readout's pure arithmetic)
    def proj(resid_row, W, b, cap):
        h = ident(resid_row)[0]
        logits = softcap((h @ W + b).float(), cap)
        return float(logits[cid] - logits[aid]), int(torch.argmax(logits))
    mA, amA = proj(resid, W_A, b0, None)
    mB, amB = proj(resid, W_B, b0, None)
    assert abs(mA - 2.0) < 1e-6 and amA == 0, (mA, amA)     # lens A: token0 wins, margin +2
    assert abs(mB + 2.0) < 1e-6 and amB == 1, (mB, amB)     # lens B: token1 wins, margin -2
    # SAME residual, DIFFERENT lens -> different margin: a cross-lens gap can be lens-driven
    assert mA != mB, (mA, mB)
    print(f"[selftest] cross-lens projection: same resid through lens A margin={mA} vs lens B margin={mB}")

    # FORMAT-CROSSOVER bookkeeping: 4 conditions assembled with the correct format-matched pairings.
    # synthetic per-item early margins; native: base QA, it chat. cross flips both.
    idxs = [0, 1, 2]
    fe = {"base": {"native": {0: 0.2, 1: 0.1, 2: 0.3}, "cross": {0: 0.9, 1: 0.8, 2: 1.0}},   # base under chat looks "it-like"
          "it":   {"native": {0: 1.0, 1: 0.9, 2: 1.1}, "cross": {0: 0.1, 1: 0.0, 2: 0.2}}}   # it under QA drops below base
    native = [fe["it"]["native"][i] - fe["base"]["native"][i] for i in idxs]                  # big positive gap
    both_chat = [fe["it"]["native"][i] - fe["base"]["cross"][i] for i in idxs]                # it-chat vs base-chat
    both_qa = [fe["it"]["cross"][i] - fe["base"]["native"][i] for i in idxs]                  # it-qa vs base-qa
    assert all(abs(x - 0.8) < 1e-9 for x in native), native                                   # 1.0-0.2 etc = 0.8
    assert all(abs(x - 0.1) < 1e-9 for x in both_chat), both_chat                             # 1.0-0.9 = 0.1
    assert all(abs(x + 0.1) < 1e-9 for x in both_qa), both_qa                                 # 0.1-0.2 = -0.1 (sign flip)
    nrep = classify_diff(native)
    assert nrep["sign"] == "POS" and nrep["ci_excludes_zero"], nrep
    # format-matched gap collapses: both_chat positive but tiny, both_qa flips sign -> GAP_COLLAPSES on QA
    gc = gap_category(both_qa, nrep["sign"])
    assert gc["category"] == "GAP_COLLAPSES", gc                                              # sign no longer POS
    gp = gap_category([0.7, 0.75, 0.72], nrep["sign"])                                        # a persisting same-sign gap
    assert gp["category"] == "GAP_PERSISTS", gp
    print(f"[selftest] format crossover: 4 conditions assembled; native sign POS, both-qa -> {gc['category']}, "
          f"persisting case -> {gp['category']}")

    # FAITHFULNESS argmax-match fraction on synthetic per-layer logit arrays (full-vocab argmax per layer).
    # 3 items, 4 layers; per-item neutral_argmax token ids and the model's own final_argmax.
    per_item = {
        0: {"neutral_argmax": [7, 7, 3, 3], "final_argmax": 3},   # matches final at layers 2,3 (and faux L0 mismatch)
        1: {"neutral_argmax": [9, 3, 3, 3], "final_argmax": 3},   # matches at 1,2,3
        2: {"neutral_argmax": [3, 3, 3, 3], "final_argmax": 3},   # matches everywhere
    }
    nseg = 4
    curve = _argmatch_curve(per_item, [0, 1, 2], nseg)
    # layer 0: only item2 matches -> 1/3 ; layer1: items1,2 -> 2/3 ; layer2: all -> 1.0 ; layer3 (final): 1.0
    assert abs(curve[0] - 1/3) < 1e-9 and abs(curve[1] - 2/3) < 1e-9, curve
    assert curve[2] == 1.0 and curve[3] == 1.0, curve
    # early-third (L0-excluded, n//3=1 -> window [1:1] empty -> fallback [1:2]) = layer1 fraction 2/3
    ef = early_fraction(curve)
    assert abs(ef - 2/3) < 1e-9, ef
    assert faithful_tag(ef) == "EARLY_FAITHFUL", ef                                           # 2/3 >= 0.5
    assert faithful_tag(0.2) == "EARLY_UNFAITHFUL"
    # layer of max mean margin on a planted trajectory
    margins = [0.0, 0.5, 2.0, 1.0]
    assert argmax_layer(margins) == 2, argmax_layer(margins)
    print(f"[selftest] faithfulness argmatch curve={[round(x,3) for x in curve]} early={round(ef,3)} "
          f"-> {faithful_tag(ef)}; peak layer index={argmax_layer(margins)}")

    # FRACTIONAL EROSION arithmetic + a fixed-seed paired bootstrap.
    # frac_erosion = (late_neutral - late_challenge)/max(late_neutral, eps)
    assert abs(frac_erosion(2.0, 0.5) - 0.75) < 1e-9, frac_erosion(2.0, 0.5)                  # (2-0.5)/2
    assert abs(frac_erosion(4.0, 1.0) - 0.75) < 1e-9, frac_erosion(4.0, 1.0)                  # SAME frac despite bigger absolute
    assert abs(frac_erosion(0.0, -1.0) - (1.0 / EROSION_EPS)) < 1e-9, frac_erosion(0.0, -1.0) # eps floor on denom
    # normalization point: a model eroding 1.5 absolute from neutral 2.0 (=0.75) ties one eroding 3.0 from 4.0
    assert frac_erosion(2.0, 0.5) == frac_erosion(4.0, 1.0)
    # paired bootstrap: positive sample CI excludes 0; zero-mean includes 0; reproducible
    pos = [0.30, 0.32, 0.28, 0.31, 0.29, 0.30, 0.33, 0.27]
    ci_pos = paired_bootstrap_ci(pos)
    true_mean = statistics.mean(pos)
    assert ci_excludes_zero(ci_pos) and ci_pos["lo"] > 0, ci_pos
    assert ci_pos["lo"] <= true_mean <= ci_pos["hi"], (ci_pos, true_mean)
    zero = [0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.25, -0.25]
    ci_zero = paired_bootstrap_ci(zero)
    assert not ci_excludes_zero(ci_zero) and ci_zero["lo"] < 0 < ci_zero["hi"], ci_zero
    assert paired_bootstrap_ci(pos, seed=SEED) == paired_bootstrap_ci(pos, seed=SEED), "non-reproducible CI"
    assert paired_bootstrap_ci([])["mean"] is None and not ci_excludes_zero(paired_bootstrap_ci([]))
    cd_pos = classify_diff(pos)
    assert cd_pos["significance"] == "SIGNIFICANT" and cd_pos["sign"] == "POS", cd_pos
    assert classify_diff(zero)["significance"] == "NULL"
    print(f"[selftest] frac_erosion normalizes magnitude (0.75==0.75); paired bootstrap positive CI "
          f"{ci_pos['lo']}..{ci_pos['hi']} excludes 0, zero-centered {ci_zero['lo']}..{ci_zero['hi']} includes 0, reproducible")

    # sign reporting
    assert sign_of(0.5) == "POS" and sign_of(-0.5) == "NEG" and sign_of(0.0) == "ZERO"
    print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name-base", default="google/gemma-2-9b")
    p.add_argument("--name-it", default="google/gemma-2-9b-it")
    p.add_argument("--tag", default="9b")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--layers-stride", type=int, default=1,
                   help="subsample layers (read every Nth resid_post) if memory-bound; final layer always kept")
    p.add_argument("--crosslens", action="store_true",
                   help="enable block B cross-lens swap (slow: full-vocab CPU reprojection ~3 min/item); default skip")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        from misconception_pool import ITEMS_WIDE
        run(args.name_base, args.name_it, args.tag, args.device, args.layers_stride, ITEMS_WIDE,
            skip_crosslens=not args.crosslens)


if __name__ == "__main__":
    main()
