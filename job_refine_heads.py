"""Salience-copy head-set refinement loop (MVP) -- FRAMING_NOTES sec 3.7-3.10 + Paper-1 stability.

Hardens the attention claim
    "knocking out attention-to-anchor for head set H reverts the salience flip"
against an adversary of {entity-swap x paraphrase}, in FIXED base-fragment mode, on gemma-2-2b.
'Adopt' grows H by the per-pair early-writer heads (sec 3.10) until necessity is stable
(mean >= DELTA, CV <= TAU) on a held-out split, OR no candidate clears the bar (boundary charted).

Design choices, each load-bearing (see the design discussion):
  - Mode is FIXED (base-fragment), never an adversary axis: sec 6/8 show the copy's *presence*
    is mode-dependent, so varying mode would make counterexamples uninterpretable.
  - The strong adversary axes are entity-swap + (weakly) paraphrase; size (2b->9b) is a
    transport check, not run here. Paper-1 (2510.00845): paraphrase is the *stable* axis.
  - Pass/fail is a stability INTERVAL (mean/CV/bootstrap-CI), never a scalar -- Paper-1 compliant.
  - Adoption is decided on the JOINT necessity of H u {cand} (sec 3.9: per-head necessities are
    non-additive), ranked by individual necessity, gated by a bootstrap CI -- not a single run.

Reuses the HookedTransformer knockout/score primitives of job_scale_mechanism.py and the
per-head-set knockout of job_localize_joint.py. No edits to those files.

  python job_refine_heads.py --selftest                 # verify loop logic on a synthetic oracle, no model
  python job_refine_heads.py --name google/gemma-2-2b   # real run (CPU ok, slow); writes out/refine_heads_2b.json
"""
import argparse
import json
import random
from pathlib import Path

# ---- frozen experimental config (changing these is a different experiment) ----
STEM = "The capital of {r} is the city of"
MIN_EFFECT = 0.5                 # below this there is no flip to revert -> instance filtered
SEED_H = [(18, 5)]               # the sec-3.10 universal reader head on gemma-2-2b
CANDIDATE_LAYERS = [0, 1, 3, 4, 7, 18]   # sec-3.9 early-to-mid search space for writers
TRAIN_PAIRS = [("Australia", "Sydney", "Canberra"), ("Texas", "Houston", "Austin"),
               ("Canada", "Toronto", "Ottawa"), ("Switzerland", "Zurich", "Bern"),
               ("Morocco", "Casablanca", "Rabat")]
HELDOUT_PAIRS = [("Florida", "Miami", "Tallahassee"), ("Washington", "Seattle", "Olympia"),
                 ("Illinois", "Chicago", "Springfield")]
FRAMES = [("famous", "{a} is the most famous city in {r}. "),
          ("associate", "Everyone associates {r} with {a}. "),
          ("thinkof", "When people think of {r}, they think of {a}. ")]


# ----------------------------- pure loop logic -----------------------------
def cv(vals):
    """Coefficient of variation; inf when the mean is ~0 (sign-mixed)."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    if abs(m) < 1e-9:
        return float("inf")
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return var ** 0.5 / abs(m)


def bootstrap_ci(vals, B=1000, seed=0, lo=5, hi=95):
    """Mean + percentile bootstrap CI of the mean. (Paper-1: report variance, not points.)"""
    if not vals:
        return None, None, None
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(vals) / n, means[lo * B // 100], means[hi * B // 100]


def refine(scorer, train, heldout, H0, delta, tau, candidate_layers, adopt_min, max_rounds, B):
    """Grow H until necessity is stable on train, or no candidate clears the bootstrap bar."""
    H = list(H0)
    rounds = []
    for r in range(max_rounds):
        necs = [(i, scorer.necessity(i, H)) for i in train]
        necs = [(i, n) for i, n in necs if n is not None]
        vals = [n for _, n in necs]
        mean, lo, hi = bootstrap_ci(vals, B)
        cex = [i for i, n in necs if n < delta]
        rec = {"round": r, "H": [list(h) for h in H], "n": len(vals),
               "necessity_mean": mean, "necessity_cv": cv(vals), "ci": [lo, hi],
               "counterexamples": [i["label"] for i in cex], "adopted": None, "action": None}
        if not cex:
            rec["action"] = "stop: train stable"
            rounds.append(rec)
            break
        # rank candidate heads by mean individual necessity across the counterexamples
        agg = {}
        for inst in cex:
            for lh, nh in scorer.head_sweep(inst, candidate_layers):
                if lh in H:
                    continue
                agg.setdefault(lh, []).append(nh)
        ranked = sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
        # adopt the first candidate whose JOINT marginal gain clears a bootstrap CI
        for cand, _ in ranked[:3]:
            gains = [scorer.necessity(i, H + [cand]) - scorer.necessity(i, H) for i in cex]
            gm, glo, ghi = bootstrap_ci(gains, B)
            if glo is not None and glo > adopt_min:
                H = H + [cand]
                rec["adopted"] = {"head": list(cand), "gain_mean": gm, "gain_ci": [glo, ghi]}
                break
        rounds.append(rec)
        if rec["adopted"] is None:
            rec["action"] = "stop: no candidate clears bootstrap (boundary charted)"
            break
    # held-out evaluation of the final H
    ho = [(i, scorer.necessity(i, H)) for i in heldout]
    ho = [(i, n) for i, n in ho if n is not None]
    hv = [n for _, n in ho]
    hmean, hlo, hhi = bootstrap_ci(hv, B)
    heldout_rec = {"H": [list(h) for h in H], "n": len(hv), "necessity_mean": hmean,
                   "necessity_cv": cv(hv), "ci": [hlo, hhi],
                   "fail": [i["label"] for i, n in ho if n < delta],
                   "stable": bool(hv) and hmean is not None and hmean >= delta and cv(hv) <= tau}
    return {"final_H": [list(h) for h in H], "rounds": rounds, "heldout": heldout_rec}


# --------------------------- synthetic oracle (selftest) ---------------------------
class FakeScorer:
    """Universal reader (18,5)->0.2 on every instance; global writer (1,0)->0.4; others 0.
    With DELTA=0.5 the loop must adopt (1,0) to make every instance stable -> deterministic."""
    CONTRIB = {(18, 5): 0.2, (1, 0): 0.4}
    CANDS = [(1, 0), (1, 3), (3, 0), (4, 5), (7, 1), (0, 2)]

    def necessity(self, inst, head_set):
        return min(1.0, sum(self.CONTRIB.get(h, 0.0) for h in head_set))

    def head_sweep(self, inst, layers):
        return [(lh, self.CONTRIB.get(lh, 0.0)) for lh in self.CANDS]


# --------------------------- model-backed scorer ---------------------------
class ModelScorer:
    def __init__(self, name):
        import torch
        from transformer_lens import HookedTransformer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[load] HookedTransformer {name} on {self.dev}", flush=True)
        self.model = HookedTransformer.from_pretrained_no_processing(name, dtype=torch.bfloat16, device=self.dev)
        self.model.eval()
        self.tok = self.model.tokenizer
        print(f"[load] done (n_layers={self.model.cfg.n_layers}, n_heads={self.model.cfg.n_heads})", flush=True)

    def _first(self, s):
        return self.tok.encode(s, add_special_tokens=False)[0]

    def _last(self, ids, hooks=None):
        with self.torch.no_grad():
            return self.model.run_with_hooks(ids, fwd_hooks=hooks)[0, -1] if hooks else self.model(ids)[0, -1]

    def _score(self, logits, cid, aid):
        lp = self.torch.log_softmax(logits.float(), -1)
        return float(lp[cid] - lp[aid])

    def _apos(self, ids_list, anchor):
        aset = set(self.model.to_tokens(anchor, prepend_bos=False)[0].tolist())
        return [i for i, t in enumerate(ids_list) if t in aset and i > 0]

    def instances(self, pairs, frames):
        out, dropped = [], []
        for region, anchor, cap in pairs:
            cid, aid = self._first(" " + cap), self._first(" " + anchor)
            n_sc = self._score(self._last(self.model.to_tokens(STEM.format(r=region)).to(self.dev)), cid, aid)
            for flab, ftxt in frames:
                s_ids = self.model.to_tokens(ftxt.format(a=anchor, r=region) + STEM.format(r=region)).to(self.dev)
                s_sc = self._score(self._last(s_ids), cid, aid)
                eff = n_sc - s_sc
                apos = self._apos(s_ids[0].tolist(), anchor)
                lab = f"{region}/{flab}"
                if apos and abs(eff) > MIN_EFFECT:
                    out.append({"label": lab, "s_ids": s_ids, "s_sc": s_sc, "eff": eff,
                                "apos": apos, "cid": cid, "aid": aid})
                else:
                    dropped.append(lab)
        if dropped:
            print(f"[filter] dropped {len(dropped)} (no effect/anchor): {dropped}", flush=True)
        return out

    def _ko_hooks(self, apos, head_set):
        by_layer = {}
        for (l, h) in head_set:
            by_layer.setdefault(l, []).append(h)

        def mk(heads):
            def hook(pattern, hook):                      # [batch, head, q, k]
                for h in heads:
                    pattern[:, h, :, apos] = 0.0
                    pattern[:, h] = pattern[:, h] / pattern[:, h].sum(-1, keepdim=True).clamp_min(1e-9)
                return pattern
            return hook
        return [(f"blocks.{l}.attn.hook_pattern", mk(hs)) for l, hs in by_layer.items()]

    def necessity(self, inst, head_set):
        if not head_set:
            return 0.0
        ko = self._score(self._last(inst["s_ids"], hooks=self._ko_hooks(inst["apos"], head_set)),
                         inst["cid"], inst["aid"])
        return (ko - inst["s_sc"]) / inst["eff"]

    def head_sweep(self, inst, layers):
        return [((l, h), self.necessity(inst, [(l, h)]))
                for l in layers for h in range(self.model.cfg.n_heads)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="HF repo, e.g. google/gemma-2-2b")
    ap.add_argument("--selftest", action="store_true", help="verify loop logic on a synthetic oracle")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.5, help="necessity pass threshold")
    ap.add_argument("--tau", type=float, default=0.5, help="CV stability bar")
    ap.add_argument("--adopt-min", type=float, default=0.05, help="min bootstrapped marginal gain to adopt")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--tag", default="2b")
    a = ap.parse_args()

    if a.selftest:
        train = [{"label": f"I{i}"} for i in range(4)]
        heldout = [{"label": f"H{i}"} for i in range(2)]
        res = refine(FakeScorer(), train, heldout, [(18, 5)], a.delta, a.tau,
                     CANDIDATE_LAYERS, a.adopt_min, a.rounds, 200)
        ok = (res["final_H"] == [[18, 5], [1, 0]] and res["heldout"]["stable"]
              and len(res["rounds"]) == 2)
        print(json.dumps(res, indent=2))
        print("SELFTEST:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    assert a.name, "--name is required (or use --selftest)"
    scorer = ModelScorer(a.name)
    train = scorer.instances(TRAIN_PAIRS, FRAMES)
    heldout = scorer.instances(HELDOUT_PAIRS, FRAMES)
    print(f"[run] train={len(train)} heldout={len(heldout)} seed_H={SEED_H}", flush=True)
    res = refine(scorer, train, heldout, SEED_H, a.delta, a.tau,
                 CANDIDATE_LAYERS, a.adopt_min, a.rounds, a.boot)
    Path("out").mkdir(exist_ok=True)
    out = f"out/refine_heads_{a.tag}.json"
    Path(out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["heldout"], indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
