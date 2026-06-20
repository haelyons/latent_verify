"""Per-layer logit-lens margin trajectory, base vs -it, neutral vs challenge, on a MATCHED item set with paired CIs.

CONTEXT. This is a matched-item de-confound of controls/logit_lens_margin_trajectory.py. That sibling gates
each model on its OWN single-turn-known items (base kept one subset, -it another), so the two per-layer
margin trajectories are aggregated over DIFFERENT items and the cross-model (base-vs-it) differential is
item-mismatch confounded. This control keeps an item only if BOTH models know the fact single-turn (each
model's NEUTRAL final-layer logit-lens margin > KNOW_GATE), aggregates everything over that shared
intersection, stores per-item trajectories so the base-vs-it differential can be a PAIRED statistic, and
reports paired percentile-bootstrap CIs (fixed seed, B=1000, 95 percent) on the per-item paired differentials.
The readout is reused verbatim from the sibling: for each model x turn x item it runs one forward pass on the
repo's neutral / counter prompt (built with the rlhf_differential / job_truthful_flip turn construction), and
at the readout position (last token) it logit-lenses every layer -- resid_post after layer L through the
model's final LayerNorm then @ W_U (+ b_U) with the gemma-2 final-logit softcap -- recording
  margin_L = logit_L(correct_first_token) - logit_L(misconception_first_token).
The early metric EXCLUDES layer 0 (embedding/unembedding-bias dominated, not computation): early_margin is the
mean of margin_L over the first third of layers with layer 0 removed; late_margin is the final-layer margin.

NEUTRAL DECISION (module-constant thresholds; reports numbers + categories only, asserts nothing about which
sign is expected or favored):
  For each of the three paired differentials -- early_diff = early_margin(it)-early_margin(base) [neutral],
  late_diff = late_margin(it)-late_margin(base) [neutral], erosion_diff = challenge_erosion(base)-
  challenge_erosion(it) where challenge_erosion(model)=late_margin(neutral)-late_margin(challenge) per item --
  classify SIGNIFICANT if the 95 percent paired-bootstrap CI excludes 0 else NULL, and report the sign of the
  mean (POS / NEG / ZERO). Per model on the matched set, report formation EARLY_FORMED if the L0-excluded
  early_margin >= FORM_TOL else LATE_FORMED. No claim is attached to any category or sign.

Run model-free selftest (no model load, CPU):
    python controls/logit_lens_margin_matched.py --selftest
Run the measurement (9b; needs the GPU box):
    python controls/logit_lens_margin_matched.py --device cuda --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
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
FORM_TOL = 0.5        # L0-excluded early_margin >= this (logits) -> EARLY_FORMED, else LATE_FORMED
KNOW_GATE = 0.0       # keep an item iff its NEUTRAL final-layer margin > this FOR BOTH MODELS (both know the fact)
N_BOOT = 1000         # paired bootstrap resamples
SEED = 0              # fixed seed -> reproducible CIs
CI_LO, CI_HI = 2.5, 97.5   # 95 percent percentile interval

DECISION_RULE = (
    "matched set = items with NEUTRAL final-layer margin > 0 for BOTH base and it. Over that set, for each of "
    "early_diff = early_margin(it)-early_margin(base) [neutral, L0-excluded first-third], late_diff = "
    "late_margin(it)-late_margin(base) [neutral final layer], erosion_diff = challenge_erosion(base)-"
    "challenge_erosion(it) [challenge_erosion = late_margin(neutral)-late_margin(challenge) per item]: "
    "SIGNIFICANT if the 95 percent paired-bootstrap CI (B=1000, seed=0) excludes 0 else NULL, plus sign of mean "
    "(POS/NEG/ZERO). Per model formation EARLY_FORMED if L0-excluded early_margin>=0.5 else LATE_FORMED. "
    "Numbers + categories only; no claim attached to any sign."
)

TURNS = ("neutral", "challenge")
MODELS = ("base", "it")
DIFFS = ("early_diff", "late_diff", "erosion_diff")


# --------------------------------------------------------------------------- pure margin / readout logic
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


def challenge_erosion(late_neutral, late_challenge):
    """late_margin(neutral) - late_margin(challenge): positive = the challenge eroded the margin. Pure."""
    return late_neutral - late_challenge


def formation(early, form_tol=FORM_TOL):
    """EARLY_FORMED if the L0-excluded first-third mean margin already clears form_tol, else LATE_FORMED. Pure."""
    return "EARLY_FORMED" if early >= form_tol else "LATE_FORMED"


def sign_of(mean, tol=1e-9):
    """POS / NEG / ZERO sign of a mean. Pure."""
    if mean > tol:
        return "POS"
    if mean < -tol:
        return "NEG"
    return "ZERO"


# --------------------------------------------------------------------------- matched intersection
def matched(base_neutral_late, it_neutral_late, know_gate=KNOW_GATE):
    """Item indices the matched set keeps: items present for BOTH models whose NEUTRAL final-layer margin
    exceeds know_gate for BOTH (both models know the fact single-turn). Pure (dict, dict -> sorted list)."""
    return sorted(i for i in base_neutral_late
                  if i in it_neutral_late and base_neutral_late[i] > know_gate and it_neutral_late[i] > know_gate)


# --------------------------------------------------------------------------- paired bootstrap CI
def paired_bootstrap_ci(values, seed=SEED, n_boot=N_BOOT, lo=CI_LO, hi=CI_HI):
    """Percentile paired-bootstrap CI of the mean of per-item paired differentials. The values list is
    ALREADY per-item differences (it minus base, aligned by item), so resampling rows = a paired bootstrap.
    Pure (list -> dict); deterministic via random.Random(seed)."""
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


# --------------------------------------------------------------------------- real per-layer readout
def _layer_margins(model, ids, cid, aid, layers, cap):
    """One forward pass; for each layer L in `layers`, logit-lens resid_post[L] at the last position
    through ln_final then @ W_U (+ b_U) with the gemma softcap, and return margin_L = logit(cid)-logit(aid).
    Memory-lean: caches only the last-position resid_post per layer (no full [seq, d_vocab] logits).
    Reused verbatim from controls/logit_lens_margin_trajectory.py."""
    nL = model.cfg.n_layers
    resid = {}

    def grab(r, hook, _resid=resid):
        _resid[hook.layer()] = r[0, -1].detach()
        return r

    names = [f"blocks.{L}.hook_resid_post" for L in range(nL)]
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(n, grab) for n in names], return_type=None)
    margins = []
    for L in layers:
        h = model.ln_final(resid[L].unsqueeze(0))[0]      # [d_model], final LN exactly as the model applies it
        logits = h @ model.W_U + model.b_U                # [d_vocab], bf16, no transpose copy
        logits = softcap(logits.float(), cap)
        margins.append(float(logits[cid] - logits[aid]))
    return margins


def _model_pass(name, is_chat, device, layers_stride, pool):
    """Load one model; per item, build the neutral and counter prompts (verbatim repo construction) and run
    the per-layer logit-lens margin on each. NO per-model gate here: keep every non-collision item, store
    its per-item neutral + challenge trajectory and its NEUTRAL final-layer margin keyed by pool index, so
    the matched intersection can be taken AFTER both models have run. Returns per-item trajectories + the
    layer index list. (The matched gate replaces the sibling's per-model KNOW_GATE gate.)"""
    from transformer_lens import HookedTransformer
    from rlhf_differential import _helpers
    from job_truthful_flip import PUSH, NEUTRAL
    print(f"[load] {name} on {device} (chat={is_chat})", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=device)
    model.eval()
    nL = model.cfg.n_layers
    cap = getattr(model.cfg, "final_logit_softcap", None)
    layers = list(range(0, nL, max(1, layers_stride)))
    if layers[-1] != nL - 1:
        layers.append(nL - 1)                              # always keep the final layer (late_margin / gate)
    raw, single, push, first, num_lp = _helpers(model, device, is_chat)
    per_item = {}                                          # pool index -> {"neutral": traj, "challenge": traj}
    neutral_late = {}                                      # pool index -> NEUTRAL final-layer margin (the gate value)
    tag = "it" if is_chat else "base"
    for i, it in enumerate(pool):
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)          # first-token correct-vs-misconception pair
        if cid == aid:                                     # first-token collision -> margin meaningless, skip
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        m_neutral = _layer_margins(model, neutral, cid, aid, layers, cap)
        m_challenge = _layer_margins(model, counter, cid, aid, layers, cap)
        per_item[i] = {"neutral": m_neutral, "challenge": m_challenge}
        neutral_late[i] = m_neutral[-1]
        print(f"  [{tag}] item {i} margin_final neu={m_neutral[-1]:+.2f} chal={m_challenge[-1]:+.2f} "
              f"q={q[:42]!r}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"per_item": per_item, "neutral_late": neutral_late, "n_layers": nL, "layers": layers}


def _mean_traj(per_item, idxs, turn, n_layers_list):
    """Mean margin_L trajectory over the matched item indices for one turn. Pure over the per-item dict."""
    if not idxs:
        return [0.0] * n_layers_list
    acc = [0.0] * n_layers_list
    for i in idxs:
        traj = per_item[i][turn]
        for j in range(n_layers_list):
            acc[j] += traj[j]
    return [x / len(idxs) for x in acc]


def run(name_base, name_it, tag, device, layers_stride, pool):
    res = {"base": _model_pass(name_base, False, device, layers_stride, pool),
           "it": _model_pass(name_it, True, device, layers_stride, pool)}
    nL = res["it"]["n_layers"]
    layers = res["it"]["layers"]
    assert res["base"]["layers"] == layers, "base/it layer index lists differ (stride mismatch)"

    # MATCHED INTERSECTION: keep an item only if BOTH models' NEUTRAL final-layer margin clears the gate
    idxs = matched(res["base"]["neutral_late"], res["it"]["neutral_late"])
    n_matched = len(idxs)
    would_keep_alone = {m: sum(1 for v in res[m]["neutral_late"].values() if v > KNOW_GATE) for m in MODELS}
    print(f"[matched] {n_matched}/{len(pool)} items known single-turn by BOTH (neutral final margin>"
          f"{KNOW_GATE}): {idxs}", flush=True)
    print(f"[matched] would-keep-alone base/it = {would_keep_alone['base']}/{would_keep_alone['it']}", flush=True)

    nseg = len(layers)
    mean_traj = {m: {t: [round(x, 4) for x in _mean_traj(res[m]["per_item"], idxs, t, nseg)] for t in TURNS}
                 for m in MODELS}

    # per-model formation on the matched set (from the L0-excluded early metric of the NEUTRAL trajectory)
    form = {}
    for m in MODELS:
        em = early_margin_no_l0(_mean_traj(res[m]["per_item"], idxs, "neutral", nseg))
        form[m] = {"early_margin_no_l0": round(em, 4),
                   "late_margin": round(late_margin(_mean_traj(res[m]["per_item"], idxs, "neutral", nseg)), 4),
                   "formation": formation(em)}

    # PER-ITEM paired differentials over the matched set (it minus base, aligned by item index)
    early_diff_vals, late_diff_vals, erosion_diff_vals = [], [], []
    for i in idxs:
        b, it = res["base"]["per_item"][i], res["it"]["per_item"][i]
        b_early = early_margin_no_l0(b["neutral"])
        it_early = early_margin_no_l0(it["neutral"])
        early_diff_vals.append(it_early - b_early)
        b_late_n, it_late_n = late_margin(b["neutral"]), late_margin(it["neutral"])
        late_diff_vals.append(it_late_n - b_late_n)
        b_eros = challenge_erosion(b_late_n, late_margin(b["challenge"]))
        it_eros = challenge_erosion(it_late_n, late_margin(it["challenge"]))
        erosion_diff_vals.append(b_eros - it_eros)         # erosion_diff = challenge_erosion(base)-challenge_erosion(it)

    differentials = {
        "early_diff": {"definition": "early_margin(it) - early_margin(base) [neutral, L0-excluded first-third]",
                       **classify_diff(early_diff_vals)},
        "late_diff": {"definition": "late_margin(it) - late_margin(base) [neutral final layer]",
                      **classify_diff(late_diff_vals)},
        "erosion_diff": {"definition": "challenge_erosion(base) - challenge_erosion(it) [late neutral minus late challenge]",
                         **classify_diff(erosion_diff_vals)},
    }

    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "logit_lens_margin_matched", "pool_size": len(pool),
        "n_layers": nL, "layers": layers, "layers_stride": layers_stride,
        "thresholds": {"FORM_TOL": FORM_TOL, "KNOW_GATE": KNOW_GATE, "N_BOOT": N_BOOT, "SEED": SEED,
                       "CI_LO": CI_LO, "CI_HI": CI_HI},
        "decision_rule": DECISION_RULE,
        "n_matched": n_matched, "matched_idxs": idxs, "would_keep_alone": would_keep_alone,
        "mean_trajectory_matched": mean_traj,
        "per_item_diff": {"early_diff": [round(x, 4) for x in early_diff_vals],
                          "late_diff": [round(x, 4) for x in late_diff_vals],
                          "erosion_diff": [round(x, 4) for x in erosion_diff_vals]},
        "formation": form,
        "differentials": differentials,
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/logit_lens_matched_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[matched] n_matched={n_matched} | formation base/it="
          f"{form['base']['formation']}/{form['it']['formation']}", flush=True)
    for d in DIFFS:
        r = differentials[d]
        print(f"[{d}] mean={r['mean']} CI=[{r['ci_lo']}, {r['ci_hi']}] n={r['n']} -> "
              f"{r['significance']} sign={r['sign']}", flush=True)
    print(f"[done] wrote out/logit_lens_matched_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    # softcap helper matches its definition cap*tanh(logits/cap); falsy cap -> identity
    v = torch.tensor([0.0, 30.0, -30.0, 5.0])
    cap = 10.0
    assert torch.allclose(softcap(v, cap), cap * torch.tanh(v / cap)), softcap(v, cap)
    assert torch.allclose(softcap(v, None), v) and torch.allclose(softcap(v, 0), v)
    print("[selftest] softcap matches cap*tanh(logits/cap), identity on falsy cap")

    # matched intersection: kept iff present AND above gate FOR BOTH models; aggregation uses only the intersection
    base_late = {0: 2.0, 1: 0.5, 2: -0.3, 3: 1.0, 4: 3.0}   # item 2 below gate for base
    it_late = {0: 1.5, 1: -0.1, 2: 0.8, 3: 2.0, 5: 4.0}     # item 1 below gate for it; item 5 base-absent; item 4 it-absent
    mset = matched(base_late, it_late)
    assert mset == [0, 3], mset                              # only 0 and 3 clear the gate on BOTH and exist in both
    # aggregation uses ONLY the intersection items
    per_item = {0: {"neutral": [9.0, 1.0, 1.0]}, 3: {"neutral": [9.0, 2.0, 2.0]},
                1: {"neutral": [9.0, 99.0, 99.0]}, 2: {"neutral": [9.0, -99.0, -99.0]}}
    mt = _mean_traj(per_item, mset, "neutral", 3)
    assert mt == [9.0, 1.5, 1.5], mt                         # only items 0,3 averaged; 1,2 excluded
    print(f"[selftest] matched intersection = {mset}; mean over intersection only = {mt}")

    # L0-excluded early metric: a huge L0 spike must NOT enter early_margin
    spike = [1000.0, 0.1, 0.2, 0.3, 0.4, 0.5]               # 6 layers; first third (n//3)=2 -> indices [0,1]
    with_l0 = statistics.mean(spike[:2])                    # what a naive first-third mean would give
    no_l0 = early_margin_no_l0(spike)                       # layer 0 dropped
    assert no_l0 == 0.1, no_l0                               # only layer 1 in first-third after dropping L0
    assert abs(with_l0 - no_l0) > 100.0, (with_l0, no_l0)   # L0 spike is excluded -> wildly different value
    # late_margin is the final layer regardless
    assert late_margin(spike) == 0.5, late_margin(spike)
    print(f"[selftest] L0-excluded early: with_l0={with_l0} vs no_l0={no_l0} (L0 spike excluded); late={late_margin(spike)}")

    # erosion / differential arithmetic on synthetic per-item margins
    assert abs(challenge_erosion(2.0, 0.5) - 1.5) < 1e-9    # challenge eroded the neutral margin by 1.5
    assert abs(challenge_erosion(2.0, 2.3) - (-0.3)) < 1e-9 # challenge raised it (negative erosion)
    # erosion_diff = challenge_erosion(base) - challenge_erosion(it)
    base_eros = challenge_erosion(2.0, 0.5)                 # 1.5
    it_eros = challenge_erosion(2.0, 1.7)                   # 0.3
    assert abs((base_eros - it_eros) - 1.2) < 1e-9, (base_eros, it_eros)
    print("[selftest] erosion + differential arithmetic OK")

    # sign reporting
    assert sign_of(0.5) == "POS" and sign_of(-0.5) == "NEG" and sign_of(0.0) == "ZERO"

    # paired bootstrap: tight positive mean difference -> CI excludes 0 AND brackets the true mean
    pos = [0.30, 0.32, 0.28, 0.31, 0.29, 0.30, 0.33, 0.27]
    ci_pos = paired_bootstrap_ci(pos)
    true_mean = statistics.mean(pos)
    assert ci_excludes_zero(ci_pos) and ci_pos["lo"] > 0, ci_pos
    assert ci_pos["lo"] <= true_mean <= ci_pos["hi"], (ci_pos, true_mean)   # CI brackets the true mean
    # zero-mean noise -> CI includes 0
    zero = [0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.25, -0.25]
    ci_zero = paired_bootstrap_ci(zero)
    assert not ci_excludes_zero(ci_zero) and ci_zero["lo"] < 0 < ci_zero["hi"], ci_zero
    # empty -> degenerate, does not exclude zero
    assert paired_bootstrap_ci([])["mean"] is None and not ci_excludes_zero(paired_bootstrap_ci([]))
    print(f"[selftest] paired bootstrap: positive CI {ci_pos['lo']}..{ci_pos['hi']} excludes 0 and brackets "
          f"mean {round(true_mean, 4)}; zero-centered {ci_zero['lo']}..{ci_zero['hi']} includes 0")

    # reproducibility: same seed -> identical CI; classify_diff wires CI -> significance + sign
    assert paired_bootstrap_ci(pos, seed=SEED) == paired_bootstrap_ci(pos, seed=SEED), "non-reproducible CI"
    cd_pos = classify_diff(pos)
    assert cd_pos["significance"] == "SIGNIFICANT" and cd_pos["sign"] == "POS", cd_pos
    cd_zero = classify_diff(zero)
    assert cd_zero["significance"] == "NULL", cd_zero
    print("[selftest] same seed -> same CI (reproducible); classify_diff SIGNIFICANT/NULL + sign wired")

    # formation from the L0-excluded early metric
    early_traj = [99.0, 0.8, 1.0, 1.5, 2.0, 2.5]           # big L0, but first-third (no L0) mean 0.8 >= FORM_TOL
    assert formation(early_margin_no_l0(early_traj)) == "EARLY_FORMED", early_margin_no_l0(early_traj)
    late_traj = [99.0, -1.0, -0.5, 0.0, 1.0, 3.0]          # first-third (no L0) mean -1.0 < FORM_TOL
    assert formation(early_margin_no_l0(late_traj)) == "LATE_FORMED", early_margin_no_l0(late_traj)
    print("[selftest] formation EARLY_FORMED / LATE_FORMED from L0-excluded early metric")
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
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        from misconception_pool import ITEMS_WIDE
        run(args.name_base, args.name_it, args.tag, args.device, args.layers_stride, ITEMS_WIDE)


if __name__ == "__main__":
    main()
