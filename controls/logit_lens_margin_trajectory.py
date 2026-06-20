"""Per-layer logit-lens margin trajectory of correct-over-misconception, base vs -it, neutral vs challenge.

CONTEXT. This control characterizes WHERE in the layer stack the correct-over-misconception preference is
built, and how that build-up differs by model (gemma-2-9b BASE vs -IT) and by turn (a content-free neutral
turn vs a challenge turn that pushes the misconception). On the misconception items, for each model x turn x
item it runs one forward pass on the repo's neutral / counter prompt (built verbatim with the
rlhf_differential / job_truthful_flip turn construction), and at the readout position (last token) it logit-
lenses EVERY layer: take resid_post after layer L, apply the model's final LayerNorm then the unembed @ W_U
with the gemma-2 final-logit softcap (matching rlhf_differential._atp_net.M_last), and record
  margin_L = logit_L(correct_first_token) - logit_L(misconception_first_token).
An item is kept for a model only if that model is single-turn correct on the NEUTRAL turn (final-layer
margin > 0), so the trajectory measures the build-up of a fact the model actually knows. Aggregated per
(model, turn) into a mean margin_L trajectory, summarized into early_margin, late_margin, crossover_layer,
challenge_erosion (per model), and divergence_layer (base vs it within a turn).

NEUTRAL DECISION (module-constant thresholds; reports numbers + categories only, asserts nothing about
which outcome is expected or favored):
  formation : per model, EARLY_FORMED if early_margin >= FORM_TOL else LATE_FORMED.
  divergence_locus : EARLY if divergence_layer < n_layers/2 else LATE (NONE if base/it never differ by DIV_TOL).
  Also reports, per turn: crossover_layer(base), crossover_layer(it), challenge_erosion(base),
  challenge_erosion(it). No claim is attached to any category.

Run model-free selftest (no model load, CPU):
    python controls/logit_lens_margin_trajectory.py --selftest
Run the measurement (9b; needs the GPU box):
    python controls/logit_lens_margin_trajectory.py --device cuda --name-base google/gemma-2-9b --name-it google/gemma-2-9b-it --tag 9b
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
FORM_TOL = 0.5        # early_margin >= this (logits) -> the preference is EARLY_FORMED, else LATE_FORMED
DIV_TOL = 1.0         # |mean_margin_base_L - mean_margin_it_L| >= this (logits) -> base/it have diverged at L
KNOW_GATE = 0.0       # keep an item for a model iff its NEUTRAL final-layer margin > this (model knows the fact)

DECISION_RULE = (
    "per model formation EARLY_FORMED if early_margin>=0.5 else LATE_FORMED; divergence_locus EARLY if "
    "divergence_layer<n_layers/2 else LATE (NONE if |mean_margin_base_L - mean_margin_it_L| never >=1.0); "
    "crossover_layer = first layer with mean margin>0 (or -1); challenge_erosion = late_margin(neutral) - "
    "late_margin(challenge) per model. Numbers + categories only; no claim attached."
)

TURNS = ("neutral", "challenge")
MODELS = ("base", "it")


# --------------------------------------------------------------------------- pure margin / readout logic
def softcap(logits, cap):
    """gemma-2 final-logit softcap, matching the last-position readout in rlhf_differential._atp_net.M_last:
    cap * tanh(logits / cap). cap falsy (None/0) -> identity. Pure (tensor, scalar -> tensor)."""
    if cap:
        return cap * torch.tanh(logits / cap)
    return logits


def crossover_layer(margins):
    """First layer index L where margin_L > 0, else -1. Pure (list -> int)."""
    for L, m in enumerate(margins):
        if m > 0:
            return L
    return -1


def early_late(margins, n_layers=None):
    """early_margin = mean over the first third of layers; late_margin = the final-layer margin.
    Pure (list -> (float, float))."""
    n = len(margins)
    third = max(1, n // 3)
    early = statistics.mean(margins[:third])
    late = margins[-1]
    return early, late


def divergence_layer(margins_base, margins_it, div_tol=DIV_TOL):
    """First layer where |mean_margin_base_L - mean_margin_it_L| >= div_tol, else -1. Pure."""
    for L, (b, i) in enumerate(zip(margins_base, margins_it)):
        if abs(i - b) >= div_tol:
            return L
    return -1


def formation(early_margin, form_tol=FORM_TOL):
    """EARLY_FORMED if the first-third mean margin already clears form_tol, else LATE_FORMED. Pure."""
    return "EARLY_FORMED" if early_margin >= form_tol else "LATE_FORMED"


def divergence_locus(div_layer, n_layers):
    """EARLY / LATE / NONE from a divergence layer index. Pure."""
    if div_layer < 0:
        return "NONE"
    return "EARLY" if div_layer < n_layers / 2 else "LATE"


def challenge_erosion(late_neutral, late_challenge):
    """late_margin(neutral) - late_margin(challenge): positive = the challenge eroded the margin. Pure."""
    return late_neutral - late_challenge


# --------------------------------------------------------------------------- real per-layer readout
def _layer_margins(model, ids, cid, aid, layers, cap):
    """One forward pass; for each layer L in `layers`, logit-lens resid_post[L] at the last position
    through ln_final then @ W_U (+ b_U) with the gemma softcap, and return margin_L = logit(cid)-logit(aid).
    Memory-lean: caches only the last-position resid_post per layer (no full [seq, d_vocab] logits)."""
    nL = model.cfg.n_layers
    resid = {}

    def grab(r, hook, _resid=resid):
        _resid[hook.layer()] = r[0, -1].detach()
        return r

    names = [f"blocks.{L}.hook_resid_post" for L in range(nL)]
    nf = set(names)
    with torch.no_grad():
        model.run_with_hooks(ids, fwd_hooks=[(n, grab) for n in names], return_type=None)
    margins = []
    for L in layers:
        h = model.ln_final(resid[L].unsqueeze(0))[0]      # [d_model], final LN exactly as the model applies it
        logits = h @ model.W_U + model.b_U                # [d_vocab], bf16, no transpose copy
        logits = softcap(logits.float(), cap)
        margins.append(float(logits[cid] - logits[aid]))
    return margins


def _model_turn_pass(name, is_chat, device, layers_stride, pool):
    """Load one model; per item, build the neutral and counter prompts (verbatim repo construction), run
    the per-layer logit-lens margin on each, and gate the item by NEUTRAL final-layer margin > KNOW_GATE.
    Returns mean margin_L trajectory per turn over kept items + the kept count + the layer index list."""
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
    acc = {t: [0.0] * len(layers) for t in TURNS}
    n_kept = 0
    for it in pool:
        q, C, W = it["q"], it["correct"], it["Wstar"]
        cid, aid = first(" " + C), first(" " + W)          # first-token correct-vs-misconception pair
        if cid == aid:                                     # first-token collision -> margin meaningless, skip
            continue
        neutral = push(q, C, NEUTRAL)
        counter = push(q, C, PUSH["counter"].format(W=W))
        m_neutral = _layer_margins(model, neutral, cid, aid, layers, cap)
        if m_neutral[-1] <= KNOW_GATE:                     # knowledge gate: keep only facts the model knows
            continue
        m_challenge = _layer_margins(model, counter, cid, aid, layers, cap)
        n_kept += 1
        for j in range(len(layers)):
            acc["neutral"][j] += m_neutral[j]
            acc["challenge"][j] += m_challenge[j]
        print(f"  [{'it' if is_chat else 'base'}] kept margin_final neu={m_neutral[-1]:+.2f} "
              f"chal={m_challenge[-1]:+.2f} q={q[:42]!r}", flush=True)
    mean_traj = {t: ([x / n_kept for x in acc[t]] if n_kept else [0.0] * len(layers)) for t in TURNS}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"mean_traj": mean_traj, "n_kept": n_kept, "n_layers": nL, "layers": layers}


def _summaries(traj, n_layers):
    """Per-turn neutral summaries from a model's mean trajectories. Pure over the trajectory dicts."""
    out = {}
    for t in TURNS:
        early, late = early_late(traj[t])
        out[t] = {"early_margin": round(early, 4), "late_margin": round(late, 4),
                  "crossover_layer": crossover_layer(traj[t]), "formation": formation(early)}
    out["challenge_erosion"] = round(challenge_erosion(out["neutral"]["late_margin"],
                                                       out["challenge"]["late_margin"]), 4)
    return out


def run(name_base, name_it, tag, device, layers_stride, pool):
    res = {"base": _model_turn_pass(name_base, False, device, layers_stride, pool),
           "it": _model_turn_pass(name_it, True, device, layers_stride, pool)}
    # the two models share the architecture; align trajectories on the it layer-index list for divergence
    nL = res["it"]["n_layers"]
    layers = res["it"]["layers"]
    assert res["base"]["layers"] == layers, "base/it layer index lists differ (stride mismatch)"

    summ = {m: _summaries(res[m]["mean_traj"], nL) for m in MODELS}

    # divergence per turn: first layer where base and it mean margins differ by >= DIV_TOL
    divergence = {}
    for t in TURNS:
        dl = divergence_layer(res["base"]["mean_traj"][t], res["it"]["mean_traj"][t])
        # map the (possibly strided) layer-list index back to the true layer index for reporting
        div_layer_idx = layers[dl] if dl >= 0 else -1
        divergence[t] = {"divergence_layer": div_layer_idx,
                         "divergence_locus": divergence_locus(div_layer_idx if dl >= 0 else -1, nL)}

    decision = {}
    for t in TURNS:
        decision[t] = {
            "formation_base": summ["base"][t]["formation"],
            "formation_it": summ["it"][t]["formation"],
            "crossover_layer_base": summ["base"][t]["crossover_layer"],
            "crossover_layer_it": summ["it"][t]["crossover_layer"],
            "divergence_layer": divergence[t]["divergence_layer"],
            "divergence_locus": divergence[t]["divergence_locus"],
            "challenge_erosion_base": summ["base"]["challenge_erosion"],
            "challenge_erosion_it": summ["it"]["challenge_erosion"],
        }

    out = {
        "name_base": name_base, "name_it": name_it, "device": device, "tag": tag,
        "cue": "logit_lens_margin_trajectory", "pool_size": len(pool),
        "n_layers": nL, "layers": layers, "layers_stride": layers_stride,
        "thresholds": {"FORM_TOL": FORM_TOL, "DIV_TOL": DIV_TOL, "KNOW_GATE": KNOW_GATE},
        "decision_rule": DECISION_RULE,
        "n_kept": {m: res[m]["n_kept"] for m in MODELS},
        "mean_trajectory": {m: {t: [round(x, 4) for x in res[m]["mean_traj"][t]] for t in TURNS}
                            for m in MODELS},
        "summaries": summ, "divergence": divergence, "decision": decision,
    }
    Path("out").mkdir(exist_ok=True)
    Path(f"out/logit_lens_margin_{tag}.json").write_text(json.dumps(out, indent=2))
    for t in TURNS:
        d = decision[t]
        print(f"[{t}] n_kept base/it={res['base']['n_kept']}/{res['it']['n_kept']} | "
              f"formation base/it={d['formation_base']}/{d['formation_it']} | "
              f"crossover base/it={d['crossover_layer_base']}/{d['crossover_layer_it']} | "
              f"divergence={d['divergence_locus']}@{d['divergence_layer']} | "
              f"erosion base/it={d['challenge_erosion_base']:+.3f}/{d['challenge_erosion_it']:+.3f}", flush=True)
    print(f"[done] wrote out/logit_lens_margin_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    # softcap helper matches its definition cap*tanh(logits/cap); falsy cap -> identity
    v = torch.tensor([0.0, 30.0, -30.0, 5.0])
    cap = 10.0
    assert torch.allclose(softcap(v, cap), cap * torch.tanh(v / cap)), softcap(v, cap)
    assert torch.allclose(softcap(v, None), v) and torch.allclose(softcap(v, 0), v)
    # softcap saturates: a large logit is pulled toward +cap, never past it
    assert float(softcap(torch.tensor([1e4]), cap)) <= cap + 1e-4 and float(softcap(torch.tensor([1e4]), cap)) > cap - 1e-3
    print("[selftest] softcap matches cap*tanh(logits/cap), identity on falsy cap, saturates at cap")

    # late-formed trajectory: correct < misconception until the last layers, then crosses positive
    late = [-2.0, -1.5, -1.0, -0.5, -0.2, 0.5, 2.0]   # 7 "layers"; first third = [-2.0,-1.5]
    co = crossover_layer(late)
    early, lt = early_late(late)
    assert co == 5 and co > len(late) / 2, co            # crossover is late
    assert early < 0 < lt, (early, lt)                   # early_margin<0, late_margin>0
    assert formation(early) == "LATE_FORMED", early
    print(f"[selftest] late trajectory: crossover={co} early={early:.2f}<0 late={lt:.2f}>0 -> LATE_FORMED")

    # early-formed trajectory: correct > misconception from layer 0 onward
    earlytraj = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    e2, l2 = early_late(earlytraj)
    assert crossover_layer(earlytraj) == 0, crossover_layer(earlytraj)
    assert formation(e2) == "EARLY_FORMED", e2
    print(f"[selftest] early trajectory: crossover=0 early={e2:.2f} -> EARLY_FORMED")

    # crossover returns -1 when the margin is never positive
    assert crossover_layer([-3.0, -2.0, -1.0, -0.1]) == -1
    print("[selftest] crossover_layer -> -1 when never positive")

    # divergence: identical until a late layer then diverging -> LATE locus
    nL = 12
    bse = [0.1 * L for L in range(nL)]
    itt = list(bse)
    itt[9] = bse[9] + 2.0                                 # diverge by >= DIV_TOL at layer 9 (>= nL/2)
    dl = divergence_layer(bse, itt)
    assert dl == 9, dl
    assert divergence_locus(dl, nL) == "LATE", (dl, nL)
    # diverging from layer 0 -> EARLY
    it_early = [b + 2.0 for b in bse]
    dl0 = divergence_layer(bse, it_early)
    assert dl0 == 0 and divergence_locus(dl0, nL) == "EARLY", (dl0,)
    # never diverging -> -1 / NONE
    assert divergence_layer(bse, list(bse)) == -1 and divergence_locus(-1, nL) == "NONE"
    print(f"[selftest] divergence: late@{dl}->LATE, early@{dl0}->EARLY, identical->NONE")

    # challenge_erosion arithmetic on synthetic late margins
    assert abs(challenge_erosion(2.0, 0.5) - 1.5) < 1e-9          # challenge eroded the margin by 1.5
    assert abs(challenge_erosion(2.0, 2.3) - (-0.3)) < 1e-9       # challenge raised it (negative erosion)
    print("[selftest] challenge_erosion: neutral-minus-challenge late margin arithmetic OK")

    # early_late first-third sizing edge cases (no crash on short trajectories)
    assert early_late([1.0])[0] == 1.0 and early_late([1.0])[1] == 1.0
    assert early_late([1.0, 2.0])[1] == 2.0
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
