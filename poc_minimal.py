"""Hyper-minimal PoC: do the causal claims of the Dallas->Austin attribution
graph survive (i) single-vs-joint intervention and (ii) task-preserving
paraphrase?

Two stages, independently re-runnable, each writing JSON to ./out:

  t0  Redundancy positive control on the seed prompt. Clamp each of the six
      published "Texas" supernode features individually, then jointly, then
      matched-random control sets. Sweep clamp multipliers.
  t1  Paraphrase transport. For each checked-in paraphrase that preserves the
      behaviour (Austin top-1), measure (a) recruitment of the seed features
      and (b) the joint-clamp effect normalized by the seed effect; tag each
      survivor with a regime: A transported / B backup-candidate / C
      non-recruited.

Out of scope by design: refinement loop, candidate adoption, mediation-share
estimation, error-node accounting, QK checks, LLM paraphrase generation.

Pre-registered success criteria (evaluated and printed by each stage):
  S1  joint clamp on seed removes Austin from top-1 OR drops its logit > 1.0
  S2  max(single drop) / joint drop < 0.3 confirms PLT redundancy;
      > 0.7 falsifies it; in between is reported as ambiguous
  S3  Texas joint drop > 3x mean matched-random-control joint drop
  S4  >= 5 paraphrases survive the behaviour filter, regimes non-degenerate

VERIFY BEFORE FIRST RUN -- three API assumptions, isolated in the three
wrappers marked  # API  below. Pin the circuit-tracer commit and check:
  1. ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma") loads
     Gemma-2-2B with GemmaScope per-layer transcoders.
  2. model.get_activations(prompt) returns (logits, activations) with
     logits[seq, vocab] (or [vocab]) and activations[n_layers, seq, d].
  3. model.feature_intervention(prompt, [(layer, pos, feat, value), ...])
     returns intervened logits with each feature set to `value` at `pos`.
If any signature differs, fix only the wrappers; nothing else touches the
library.
"""

import argparse
import json
import random
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Frozen experimental constants (changing these is a different experiment)
# --------------------------------------------------------------------------

SEED_PROMPT = "Fact: The capital of the state containing Dallas is"
TARGET_STR = " Austin"
MODEL_NAME = "google/gemma-2-2b"
TRANSCODERS = "gemma"  # GemmaScope per-layer transcoders

# Texas supernode, pinned from the canonical Neuronpedia gemma-fact-dallas-
# austin graph: (layer, feature_idx). Published position suffix is _9 (the
# Dallas token region on the seed tokenization); positions are re-derived
# per-prompt here because paraphrases retokenize.
TEXAS = [(20, 15589), (19, 7477), (16, 25), (14, 2268), (7, 6861), (4, 13154)]

OUT_DIR = Path("./out")


# --------------------------------------------------------------------------
# Thin wrappers around circuit-tracer -- the ONLY schematic code (see header)
# --------------------------------------------------------------------------

def load_model():  # API (assumption 1)
    from transformers import AutoModelForCausalLM
    from transformer_lens.weight_processing import ProcessWeights
    from circuit_tracer import ReplacementModel

    # Memory-constrained-host path; every measure below is bit-identical to
    # the default load for THIS model and flag set:
    # 1. hf_model= pre-loads the weights in bf16 so TransformerLens does not
    #    re-materialize the fp32 shards itself.
    # 2. lazy_encoder streams W_enc from the cached safetensors per use
    #    (same file, same cast) instead of holding ~2 GB resident.
    # 3. circuit-tracer loads with fold_ln/center_writing/center_unembed all
    #    False; fold_value_biases=False is additionally passed because
    #    gemma-2 has no biases anywhere, making the fold a numerical no-op.
    #    With all flags False, ProcessWeights.process_weights performs no
    #    transformation but still round-trips the entire state dict through
    #    fp32 (~10 GB transient -- OOM on a 16 GB host); the guard skips the
    #    call when it would be the identity (bf16->fp32->bf16 is exact).
    _orig = ProcessWeights.process_weights

    def _guarded(state_dict, cfg, fold_ln=True, center_writing_weights=True,
                 center_unembed=True, fold_value_biases=True,
                 refactor_factored_attn_matrices=False, adapter=None):
        if not any((fold_ln, center_writing_weights, center_unembed,
                    fold_value_biases, refactor_factored_attn_matrices)):
            return state_dict
        return _orig(state_dict, cfg, fold_ln, center_writing_weights,
                     center_unembed, fold_value_biases,
                     refactor_factored_attn_matrices, adapter)

    ProcessWeights.process_weights = staticmethod(_guarded)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return ReplacementModel.from_pretrained(MODEL_NAME, TRANSCODERS,
                                            dtype=torch.bfloat16,
                                            lazy_encoder=True,
                                            hf_model=hf_model,
                                            fold_value_biases=False)


def logits_and_acts(model, prompt):  # API (assumption 2)
    """Return (last_position_logits [vocab], activations [n_layers, seq, d])."""
    logits, acts = model.get_activations(prompt)
    # circuit-tracer @041a9b2 returns logits [batch, seq, vocab] (3-D); also
    # tolerate [seq, vocab] and [vocab]. acts is [n_layers, seq, d] (squeeze(0)).
    last = (logits[0, -1] if logits.dim() == 3
            else logits[-1] if logits.dim() == 2 else logits)
    assert acts.dim() == 3, f"unexpected activations shape {tuple(acts.shape)}"
    return last.float(), acts


def intervened_logits(model, prompt, interventions):  # API (assumption 3)
    """interventions: list of (layer, pos, feature_idx, new_value)."""
    logits, _ = model.feature_intervention(prompt, interventions)
    # feature_intervention returns logits [batch, seq, vocab] (3-D); match wrapper 2.
    last = (logits[0, -1] if logits.dim() == 3
            else logits[-1] if logits.dim() == 2 else logits)
    return last.float()


# --------------------------------------------------------------------------
# Measurement helpers (plain functions, no library coupling)
# --------------------------------------------------------------------------

def target_token_id(model):
    ids = model.tokenizer.encode(TARGET_STR, add_special_tokens=False)
    assert len(ids) == 1, (
        f"{TARGET_STR!r} tokenizes to {ids}; the single-token assumption "
        "fails -- stop and re-derive the target before trusting any number.")
    return ids[0]


def active_positions(acts, layer, feat):
    """Positions where the feature fires on this prompt, with values."""
    vals = acts[layer, :, feat]
    pos = torch.nonzero(vals > 0).flatten().tolist()
    return [(p, float(vals[p])) for p in pos]


def clamp_interventions(acts, features, multiplier):
    """Clamp each feature to multiplier * its value at every active position.

    multiplier = 0.0 is zero-ablation; negative multipliers reproduce the
    Anthropic-style inhibition. Features with no active positions contribute
    no intervention (a no-op, recorded by the caller via recruitment stats).
    """
    ivs = []
    for (layer, feat) in features:
        for (p, v) in active_positions(acts, layer, feat):
            ivs.append((layer, p, feat, multiplier * v))
    return ivs


def rank_of(logits, token_id):
    return int((logits > logits[token_id]).sum().item())  # 0 = top-1


def measure_clamp(model, prompt, acts, features, multiplier, tid, base_logits):
    """Effect of clamping `features` on the target logit for this prompt."""
    ivs = clamp_interventions(acts, features, multiplier)
    if not ivs:  # nothing fired -- intervention is vacuous
        return {"drop": 0.0, "rank_after": rank_of(base_logits, tid),
                "n_interventions": 0, "vacuous": True}
    abl = intervened_logits(model, prompt, ivs)
    return {"drop": float(base_logits[tid] - abl[tid]),
            "rank_after": rank_of(abl, tid),
            "n_interventions": len(ivs), "vacuous": False}


def matched_random_features(acts, rng):
    """Petrova-style matched null: for each Texas feature, sample a random
    non-Texas feature in the SAME layer with seed max-activation within
    [0.5x, 2x] of it (widening if empty). Controls for layer and magnitude."""
    texas_set = set(TEXAS)
    control = []
    for (layer, feat) in TEXAS:
        target_mag = float(acts[layer, :, feat].max())
        layer_max = acts[layer].max(dim=0).values  # [d] max over positions
        for lo, hi in [(0.5, 2.0), (0.25, 4.0), (0.0, float("inf"))]:
            cand = torch.nonzero(
                (layer_max > max(lo * target_mag, 1e-6))
                & (layer_max < hi * target_mag)).flatten().tolist()
            cand = [c for c in cand if (layer, c) not in texas_set]
            if cand:
                control.append((layer, rng.choice(cand)))
                break
    return control


# --------------------------------------------------------------------------
# Stage t0: redundancy positive control on the seed prompt
# --------------------------------------------------------------------------

def stage_t0(model, args):
    tid = target_token_id(model)
    base, acts = logits_and_acts(model, SEED_PROMPT)
    rng = random.Random(args.seed)

    # Audit record: where/how strongly each pinned feature fires on the seed.
    firing = {f"L{l}/{f}": active_positions(acts, l, f) for (l, f) in TEXAS}
    dead = [k for k, v in firing.items() if not v]
    if dead:
        print(f"[t0][WARN] pinned features inactive on seed: {dead} -- "
              "feature list or transcoder set likely wrong; results suspect.")

    results = {"seed_prompt": SEED_PROMPT, "austin_id": tid,
               "austin_base_logit": float(base[tid]),
               "austin_base_rank": rank_of(base, tid),
               "firing_positions": firing, "by_multiplier": {}}

    for m in args.multipliers:  # response curve, not a point estimate
        singles = {f"L{l}/{f}": measure_clamp(model, SEED_PROMPT, acts,
                                              [(l, f)], m, tid, base)
                   for (l, f) in TEXAS}
        joint = measure_clamp(model, SEED_PROMPT, acts, TEXAS, m, tid, base)
        controls = [measure_clamp(model, SEED_PROMPT, acts,
                                  matched_random_features(acts, rng),
                                  m, tid, base)["drop"]
                    for _ in range(args.n_controls)]
        ctrl_mean = sum(controls) / len(controls)
        max_single = max(s["drop"] for s in singles.values())
        ratio = max_single / joint["drop"] if joint["drop"] > 0 else float("inf")

        results["by_multiplier"][str(m)] = {
            "singles": singles, "joint": joint,
            "control_drops": controls, "control_mean": ctrl_mean,
            "max_single_over_joint": ratio,
            "joint_over_control": (joint["drop"] / ctrl_mean
                                   if ctrl_mean > 0 else float("inf")),
        }
        # Pre-registered criteria, evaluated in-line
        s1 = joint["rank_after"] > 0 or joint["drop"] > 1.0
        s2 = ("redundant" if ratio < 0.3 else
              "NOT redundant" if ratio > 0.7 else "ambiguous")
        s3 = ctrl_mean <= 0 or joint["drop"] > 3 * ctrl_mean
        print(f"[t0][m={m}] joint drop={joint['drop']:.3f} "
              f"(rank {results['austin_base_rank']}->{joint['rank_after']}) | "
              f"max single={max_single:.3f} | ratio={ratio:.2f} ({s2}) | "
              f"control mean={ctrl_mean:.3f} | S1={'PASS' if s1 else 'FAIL'} "
              f"S3={'PASS' if s3 else 'FAIL'}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "t0.json").write_text(json.dumps(results, indent=2))
    print(f"[t0] written to {OUT_DIR / 't0.json'}")


# --------------------------------------------------------------------------
# Stage t1: paraphrase transport with recruitment tagging
# --------------------------------------------------------------------------

def stage_t1(model, args):
    tid = target_token_id(model)
    m = args.transport_multiplier

    # Seed baseline recomputed here so t1 never depends on t0's artifact.
    seed_base, seed_acts = logits_and_acts(model, SEED_PROMPT)
    seed_max = {(l, f): float(seed_acts[l, :, f].max()) for (l, f) in TEXAS}
    seed_joint = measure_clamp(model, SEED_PROMPT, seed_acts, TEXAS, m,
                               tid, seed_base)
    assert seed_joint["drop"] > 0, (
        "seed joint clamp has no effect; run t0 first and resolve S1 "
        "before interpreting any paraphrase result.")

    data = json.loads(Path(args.paraphrases).read_text())
    records = []
    for entry in data["paraphrases"]:
        prompt, structure = entry["prompt"], entry["structure"]
        base, acts = logits_and_acts(model, prompt)
        margin = float(base[tid]
                       - base[torch.argsort(base, descending=True)[
                           1 if rank_of(base, tid) == 0 else 0]])
        rec = {"prompt": prompt, "structure": structure,
               "austin_rank": rank_of(base, tid), "margin": margin}
        if rec["austin_rank"] != 0:  # behaviour-preservation filter (top-1)
            rec["regime"] = "filtered"
            records.append(rec)
            continue

        recruited = [(l, f) for (l, f) in TEXAS
                     if float(acts[l, :, f].max())
                     >= args.recruit_frac * max(seed_max[(l, f)], 1e-6)]
        eff = measure_clamp(model, prompt, acts, TEXAS, m, tid, base)
        rec.update({
            "n_recruited": len(recruited),
            "recruited": [f"L{l}/{f}" for (l, f) in recruited],
            "drop": eff["drop"],
            "drop_normalized": eff["drop"] / seed_joint["drop"],
            "rank_after": eff["rank_after"],
        })
        if len(recruited) < args.min_recruited:
            rec["regime"] = "C_non_recruited"
        elif rec["drop_normalized"] >= args.transport_frac:
            rec["regime"] = "A_transported"
        else:
            rec["regime"] = "B_backup_candidate"
        records.append(rec)

    survivors = [r for r in records if r["regime"] != "filtered"]
    print(f"[t1] {len(survivors)}/{len(records)} paraphrases preserve "
          f"behaviour (S4 {'PASS' if len(survivors) >= 5 else 'FAIL'}); "
          f"seed joint drop={seed_joint['drop']:.3f} at m={m}")
    for structure in ("minimal", "syntactic", "reordered"):
        regs = [r["regime"][0] for r in survivors
                if r["structure"] == structure]
        print(f"[t1]   {structure:>9}: " +
              " ".join(f"{k}={regs.count(k)}" for k in "ABC"))

    OUT_DIR.mkdir(exist_ok=True)
    out = {"transport_multiplier": m, "seed_joint_drop": seed_joint["drop"],
           "records": records}
    (OUT_DIR / "t1.json").write_text(json.dumps(out, indent=2))
    print(f"[t1] written to {OUT_DIR / 't1.json'}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["t0", "t1", "all"], default="all")
    p.add_argument("--multipliers", nargs="+", type=float, default=[0.0, -2.0],
                   help="t0 clamp multipliers (0 = zero-ablation)")
    p.add_argument("--transport-multiplier", type=float, default=-2.0,
                   help="single multiplier used for t1 (calibrate via t0)")
    p.add_argument("--recruit-frac", type=float, default=0.25,
                   help="paraphrase max-activation as fraction of seed "
                        "max-activation to count a feature as recruited")
    p.add_argument("--min-recruited", type=int, default=3,
                   help="fewer recruited features than this => regime C")
    p.add_argument("--transport-frac", type=float, default=0.5,
                   help="normalized drop threshold separating regimes A/B")
    p.add_argument("--n-controls", type=int, default=5)
    p.add_argument("--paraphrases", default="paraphrases.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = load_model()
    if args.stage in ("t0", "all"):
        stage_t0(model, args)
    if args.stage in ("t1", "all"):
        stage_t1(model, args)


if __name__ == "__main__":
    main()
