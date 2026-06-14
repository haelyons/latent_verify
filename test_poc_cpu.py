"""CPU validation harness for poc_minimal.py -- no GPU, no HF access needed.

Drives stage_t0 and stage_t1 end-to-end against a deterministic mock
ReplacementModel whose causal ground truth is fixed by construction, so every
branch of the measurement logic can be asserted against known-correct values:

  - intervention construction (positions re-derived per prompt; the mock
    raises if an intervention targets a position where the feature is not
    active, so position bookkeeping is checked on every call)
  - S1 (rank flip / logit drop), S2 classification at both extremes
    (redundant vs dominant single feature), S3 (matched-null specificity)
  - the matched-random control sampler (band widening fallback included)
  - t1 behaviour filter, margin sign, recruitment counting, and all four
    regimes (filtered / A_transported / B_backup_candidate / C_non_recruited)
  - vacuous-intervention handling and JSON artifact shape

The mock exposes exactly the three # API surfaces poc_minimal touches
(tokenizer.encode, get_activations, feature_intervention) with the 3-D
[batch, seq, vocab] logits shape of circuit-tracer @041a9b2.

Run:  .venv/bin/python test_poc_cpu.py
"""

import argparse
import json
from pathlib import Path

import torch

import poc_minimal as poc

N_LAYERS, SEQ, D, VOCAB = 26, 12, 16384, 32
AUSTIN_ID, RIVAL_ID = 7, 8
RIVAL_LOGIT = 10.0
JOINT_EFFECT = 5.0      # ground-truth Austin-logit drop when the rule fires
DECOY_EFFECT = 0.1      # per clamped decoy feature (exercises S3's 3x branch)


class MockTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert text == poc.TARGET_STR and not add_special_tokens
        return [AUSTIN_ID]


class PromptSpec:
    """acts: {(layer, feat): (pos, value)}; rule decides the Austin logit.

    rule = "redundant": effect fires only when ALL six TEXAS features are
           clamped (any single clamp is fully compensated)
    rule = "dominant":  effect fires iff TEXAS[0] is clamped (one feature
           carries the whole effect)
    rule = "backup":    TEXAS clamps never matter (parallel path carries
           the behaviour)
    """

    def __init__(self, acts, rule, austin_base=12.0):
        self.acts, self.rule, self.austin_base = acts, rule, austin_base


class MockModel:
    def __init__(self, world):
        self.world = world  # {prompt: PromptSpec}
        self.tokenizer = MockTokenizer()

    def _logits(self, spec, clamped):
        n_texas = sum(1 for lf in clamped if lf in set(poc.TEXAS))
        n_decoy = len(clamped) - n_texas
        if spec.rule == "redundant":
            drop = JOINT_EFFECT if n_texas == len(poc.TEXAS) else 0.0
        elif spec.rule == "dominant":
            drop = JOINT_EFFECT if poc.TEXAS[0] in clamped else 0.0
        elif spec.rule == "backup":
            drop = 0.0
        else:
            raise ValueError(spec.rule)
        logits = torch.zeros(1, SEQ, VOCAB)
        logits[0, -1, AUSTIN_ID] = spec.austin_base - drop - DECOY_EFFECT * n_decoy
        logits[0, -1, RIVAL_ID] = RIVAL_LOGIT
        return logits

    def get_activations(self, prompt):
        spec = self.world[prompt]
        acts = torch.zeros(N_LAYERS, SEQ, D)
        for (layer, feat), (pos, val) in spec.acts.items():
            acts[layer, pos, feat] = val
        return self._logits(spec, clamped=set()), acts

    def feature_intervention(self, prompt, interventions):
        spec = self.world[prompt]
        clamped = set()
        for (layer, pos, feat, value) in interventions:
            key = (layer, feat)
            assert key in spec.acts and spec.acts[key][0] == pos, (
                f"intervention at inactive site L{layer}/{feat}@{pos} -- "
                "position re-derivation is broken")
            if value != spec.acts[key][1]:
                clamped.add(key)
        return self._logits(spec, clamped), None


SEED_MAGS = {lf: float(10 + 5 * i) for i, lf in enumerate(poc.TEXAS)}


def seed_acts(mags=SEED_MAGS, pos=9, with_decoys=True):
    """TEXAS features at `pos` plus, per layer, two same-magnitude decoy
    features (zero causal weight) so the matched-null sampler has in-band
    candidates."""
    acts = {lf: (pos, mags[lf]) for lf in mags}
    if with_decoys:
        for (layer, feat), (_, val) in list(acts.items()):
            if (layer, feat) in set(poc.TEXAS):
                acts[(layer, feat + 101)] = (5, val)
                acts[(layer, feat + 202)] = (5, val)
    return acts


def t0_args(**kw):
    base = dict(multipliers=[0.0, -2.0], n_controls=5, seed=0)
    base.update(kw)
    return argparse.Namespace(**base)


def t1_args(paraphrases, **kw):
    base = dict(transport_multiplier=-2.0, recruit_frac=0.25,
                min_recruited=3, transport_frac=0.5, paraphrases=paraphrases)
    base.update(kw)
    return argparse.Namespace(**base)


def approx(a, b, tol=1e-5):
    return abs(a - b) <= tol


checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# --------------------------------------------------------------------------

def test_unit_helpers():
    print("== unit: measurement helpers ==")
    logits = torch.tensor([1.0, 5.0, 3.0])
    check("rank_of top-1", poc.rank_of(logits, 1) == 0)
    check("rank_of third", poc.rank_of(logits, 0) == 2)

    acts = torch.zeros(N_LAYERS, SEQ, D)
    acts[3, 2, 100] = 7.0
    acts[3, 8, 100] = 1.5
    ivs = poc.clamp_interventions(acts, [(3, 100)], -2.0)
    check("clamp covers every active position",
          sorted(ivs) == [(3, 2, 100, -14.0), (3, 8, 100, -3.0)])
    check("clamp skips inactive features",
          poc.clamp_interventions(acts, [(5, 100)], 0.0) == [])

    model = MockModel({"p": PromptSpec(seed_acts(), "redundant")})
    base, a = poc.logits_and_acts(model, "p")
    check("wrapper reduces 3-D logits to [vocab]",
          base.shape == (VOCAB,) and float(base[AUSTIN_ID]) == 12.0)
    vac = poc.measure_clamp(model, "p", a, [(9, 9999)], 0.0, AUSTIN_ID, base)
    check("absent feature -> vacuous no-op",
          vac["vacuous"] and vac["drop"] == 0.0 and vac["n_interventions"] == 0)

    # matched null: in-band decoys exist for every TEXAS layer
    import random
    ctrl = poc.matched_random_features(a, random.Random(0))
    check("matched null returns one control per TEXAS feature",
          len(ctrl) == len(poc.TEXAS))
    check("matched null excludes TEXAS itself",
          not set(ctrl) & set(poc.TEXAS))
    by_layer = {l: f for l, f in poc.TEXAS}
    check("matched null is layer-matched",
          all(l in by_layer for l, _ in ctrl))
    mags = {(l, f): float(a[l, :, f].max()) for l, f in ctrl}
    check("matched null is magnitude-matched (band 1)",
          all(0.5 * SEED_MAGS[(l, by_layer[l])] <= m <= 2 * SEED_MAGS[(l, by_layer[l])]
              for (l, _), m in mags.items()))

    # band-widening fallback: lone same-layer candidate far outside band 1
    acts2 = torch.zeros(N_LAYERS, SEQ, D)
    for (l, f), (p, v) in seed_acts(with_decoys=False).items():
        acts2[l, p, f] = v
    acts2[poc.TEXAS[0][0], 5, 1] = SEED_MAGS[poc.TEXAS[0]] * 10  # band 3 only
    ctrl2 = poc.matched_random_features(acts2, random.Random(0))
    check("band widening falls back to the only candidate",
          (poc.TEXAS[0][0], 1) in ctrl2)


def test_t0(rule, expect_s2):
    print(f"== t0 end-to-end: {rule} ground truth ==")
    model = MockModel({poc.SEED_PROMPT: PromptSpec(seed_acts(), rule)})
    poc.stage_t0(model, t0_args())
    r = json.loads((poc.OUT_DIR / "t0.json").read_text())

    check("audit: firing positions match spec",
          all(r["firing_positions"][f"L{l}/{f}"] == [[9, SEED_MAGS[(l, f)]]]
              for l, f in poc.TEXAS))
    check("base rank is top-1", r["austin_base_rank"] == 0)
    for m in ("0.0", "-2.0"):
        b = r["by_multiplier"][m]
        joint, singles = b["joint"], b["singles"]
        check(f"m={m}: joint drop = ground truth",
              approx(joint["drop"], JOINT_EFFECT))
        check(f"m={m}: S1 (rank flip and drop>1)",
              joint["rank_after"] > 0 and joint["drop"] > 1.0)
        check(f"m={m}: joint touches all six sites",
              joint["n_interventions"] == 6 and not joint["vacuous"])
        if rule == "redundant":
            check(f"m={m}: every single clamp fully compensated",
                  all(approx(s["drop"], 0.0) for s in singles.values()))
            check(f"m={m}: S2 ratio < 0.3 (redundant)",
                  b["max_single_over_joint"] < 0.3)
        else:  # dominant
            l0, f0 = poc.TEXAS[0]
            check(f"m={m}: dominant single carries full effect",
                  approx(singles[f"L{l0}/{f0}"]["drop"], JOINT_EFFECT))
            check(f"m={m}: S2 ratio > 0.7 (NOT redundant)",
                  b["max_single_over_joint"] > 0.7)
        check(f"m={m}: controls = 6 decoy clamps each",
              all(approx(c, 6 * DECOY_EFFECT) for c in b["control_drops"]))
        check(f"m={m}: S3 specificity > 3x",
              b["joint_over_control"] > 3)
    assert expect_s2  # documents which S2 outcome this scenario encodes


def test_t1(tmp):
    print("== t1 end-to-end: regime assignment ==")
    full = seed_acts()
    recruited_weak = {lf: (9, 0.3 * SEED_MAGS[lf]) for lf in SEED_MAGS}
    two_only = {lf: (9, SEED_MAGS[lf]) for lf in list(SEED_MAGS)[:2]}
    world = {
        poc.SEED_PROMPT: PromptSpec(full, "redundant"),
        "pA transported": PromptSpec(recruited_weak, "redundant"),
        "pB backup": PromptSpec(full, "backup"),
        "pC non-recruited": PromptSpec(two_only, "redundant"),
        "pF filtered": PromptSpec(full, "redundant", austin_base=8.0),
    }
    pj = tmp / "mock_paraphrases.json"
    pj.write_text(json.dumps({"paraphrases": [
        {"prompt": "pA transported", "structure": "minimal"},
        {"prompt": "pB backup", "structure": "syntactic"},
        {"prompt": "pC non-recruited", "structure": "reordered"},
        {"prompt": "pF filtered", "structure": "minimal"},
    ]}))
    poc.stage_t1(MockModel(world), t1_args(str(pj)))
    r = json.loads((poc.OUT_DIR / "t1.json").read_text())
    by = {rec["prompt"]: rec for rec in r["records"]}

    check("seed joint drop = ground truth",
          approx(r["seed_joint_drop"], JOINT_EFFECT))
    a, b, c, f = (by["pA transported"], by["pB backup"],
                  by["pC non-recruited"], by["pF filtered"])
    check("A: recruited at 0.3x seed (>= recruit-frac 0.25)",
          a["n_recruited"] == 6)
    check("A: normalized drop 1.0 -> A_transported",
          approx(a["drop_normalized"], 1.0) and a["regime"] == "A_transported")
    check("B: recruited but no effect -> B_backup_candidate",
          b["n_recruited"] == 6 and approx(b["drop_normalized"], 0.0)
          and b["regime"] == "B_backup_candidate")
    check("C: 2 of 6 recruited (< min 3) -> C_non_recruited",
          c["n_recruited"] == 2 and c["regime"] == "C_non_recruited")
    check("filtered: Austin not top-1, negative margin, no clamp run",
          f["regime"] == "filtered" and f["austin_rank"] > 0
          and f["margin"] < 0 and "drop" not in f)
    check("survivor margins positive",
          all(by[p]["margin"] > 0 for p in
              ("pA transported", "pB backup", "pC non-recruited")))


def test_paraphrase_file():
    print("== frozen paraphrases.json integrity ==")
    d = json.loads(Path("paraphrases.json").read_text())
    ps = d["paraphrases"]
    check("seed field matches frozen SEED_PROMPT", d["seed"] == poc.SEED_PROMPT)
    check("exactly 16 paraphrases", len(ps) == 16)
    check("no duplicate prompts; none equals the seed",
          len({p["prompt"] for p in ps}) == 16
          and poc.SEED_PROMPT not in {p["prompt"] for p in ps})
    check("all structure tags valid",
          {p["structure"] for p in ps} == {"minimal", "syntactic", "reordered"})
    check("every paraphrase mentions Dallas",
          all("Dallas" in p["prompt"] for p in ps))
    check("no paraphrase leaks the answer or the mediator",
          all("Austin" not in p["prompt"] and "Texas" not in p["prompt"]
              for p in ps))


def main():
    import tempfile
    test_unit_helpers()
    test_t0("redundant", expect_s2="redundant")
    test_t0("dominant", expect_s2="NOT redundant")
    with tempfile.TemporaryDirectory() as tmp:
        test_t1(Path(tmp))
    test_paraphrase_file()
    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit("FAILED: " + "; ".join(failed))


if __name__ == "__main__":
    main()
