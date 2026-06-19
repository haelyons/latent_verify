"""NEXT-3b: does the 27b RLHF copy-head OV-gain change MATTER behaviorally? (RESEARCH_QUESTIONS Part 4.)

NEXT-3 (ov_magnitude_characterize.py) showed RLHF rescales the OV write of specific 27b copy heads by a
pure scalar alpha (L17.H4 x1.52, L11 kv-group x1.33, L23.H24 x0.75), same direction. That is a WEIGHT
fact. This control asks the behavioral half: on a copy task where these heads fire, does restoring the
head's BASE OV gain measurably change the copied-token logit?

Causal test (forward passes on -it 27b; needs an 80GB GPU):
  SCALE-ABLATION: hook the head's hook_z and multiply by 1/alpha. Because W_OV_it = alpha * W_OV_base, the
  head's residual write out = z @ W_O scales linearly with z, so z *= 1/alpha makes the head write at its
  BASE magnitude on the -it model (valid whether alpha lives in W_V or W_O). This is the counterfactual
  "what if RLHF had not changed this head's gain". Measure delta logit(copied token).
  KNOCKOUT (calibration): z := 0 -> how much the head contributes to the copy at all. If ~0, the head is
  behaviorally inactive on this probe and the gain change cannot matter (probe did not engage it; that is
  inconclusive, NOT "does not matter").

Substrate: batched random-token INDUCTION prompts (seq = k random tokens repeated; at the second copy the
induction/copy heads attend the earlier occurrence and copy the next token). Readout = logit of the
correct copied token at the induction position. Copy heads are exactly the heads that should fire here.

Per head, mean over prompts: knockout_dlogit (z=0) and scale_dlogit (z*=1/alpha). NEUTRAL decision:
  INACTIVE        if |knockout_dlogit| < K_TOL                 (head does not drive the copy on this probe)
  MATTERS         if active AND |scale_dlogit| >= SCALE_TOL    (restoring base gain moves the copy logit)
  NEGLIGIBLE_GAIN if active AND |scale_dlogit| <  SCALE_TOL    (head matters but its gain change does not)

  python ov_behavioral_scale.py --selftest
  python ov_behavioral_scale.py --name-it google/gemma-2-27b-it --tag 27b   # 80GB GPU
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch

# per-head OV scalar alpha (it/base) from results_27b_ovmag/out/ov_magnitude_27b.json (committed).
# scale-ablation multiplies hook_z by 1/alpha to restore the base-magnitude write on the -it model.
ALPHA = {(17, 4): 1.518, (11, 2): 1.332, (11, 4): 1.333, (11, 7): 1.332, (11, 21): 1.332,
         (23, 24): 0.753, (16, 3): 0.971, (19, 2): 0.970, (19, 5): 0.970, (19, 7): 0.968}
K_TOL = 0.10          # |knockout dlogit| below this -> head inactive on the probe
SCALE_TOL = 0.05      # |scale-ablation dlogit| at/above this -> the gain change has a behavioral effect
N_PROMPTS = 16
K_REP = 12            # induction sequence length (doubled -> seq len 24)


# --------------------------------------------------------------------------- pure decision
def decide(knockout_dlogit, scale_dlogit, k_tol=K_TOL, scale_tol=SCALE_TOL):
    if abs(knockout_dlogit) < k_tol:
        return {"verdict": "INACTIVE", "note": "head does not drive the copy on this probe (knockout ~0)"}
    if abs(scale_dlogit) >= scale_tol:
        return {"verdict": "MATTERS", "note": "restoring base OV gain moves the copy logit"}
    return {"verdict": "NEGLIGIBLE_GAIN", "note": "head drives copy but its gain change barely moves the logit"}


# --------------------------------------------------------------------------- prompt builder
def induction_batch(vocab_lo, vocab_hi, n, k, seed=0):
    """n random-token induction prompts (each = k random ids, repeated). Returns (ids[n,2k], target[n],
    readpos). At readpos = 2k-2 the model should predict target = the k-1'th original token. Pure."""
    rng = random.Random(seed)
    seqs, targets = [], []
    for _ in range(n):
        toks = [rng.randrange(vocab_lo, vocab_hi) for _ in range(k)]
        seqs.append(toks + toks)
        targets.append(toks[k - 1])         # token at position 2k-1; predicted from readpos 2k-2
    return seqs, targets, 2 * k - 2


# --------------------------------------------------------------------------- real run
def _zname(L):
    return f"blocks.{L}.attn.hook_z"


def run(name_it, tag, device):
    from transformer_lens import HookedTransformer
    print(f"[load] {name_it} on {device}", flush=True)
    model = HookedTransformer.from_pretrained_no_processing(name_it, dtype=torch.bfloat16, device=device)
    model.eval()
    nH = model.cfg.n_heads
    d_vocab = model.cfg.d_vocab
    seqs, targets, readpos = induction_batch(1000, min(50000, d_vocab - 1), N_PROMPTS, K_REP)
    ids = torch.tensor(seqs, device=device)
    tgt = torch.tensor(targets, device=device)

    def read_logits(fwd_hooks=None):
        with torch.no_grad():
            lg = model.run_with_hooks(ids, fwd_hooks=fwd_hooks or [])
        return lg[:, readpos, :].float()               # [n, d_vocab]

    clean = read_logits()
    base_tlogit = clean[torch.arange(len(tgt)), tgt]    # [n] clean logit of the copied token
    # induction sanity: how often the copied token is argmax under the clean -it model
    copy_acc = float((clean.argmax(-1) == tgt).float().mean())
    print(f"[probe] induction copy-acc (argmax==target) = {copy_acc:.2f} over {N_PROMPTS} prompts", flush=True)

    rows = {}
    for (L, H) in ALPHA:
        def ko(z, hook, H=H):
            z[:, :, H, :] = 0; return z
        s = 1.0 / ALPHA[(L, H)]
        def sc(z, hook, H=H, s=s):
            z[:, :, H, :] = z[:, :, H, :] * s; return z
        ko_t = read_logits([(_zname(L), ko)])[torch.arange(len(tgt)), tgt]
        sc_t = read_logits([(_zname(L), sc)])[torch.arange(len(tgt)), tgt]
        ko_d = float((ko_t - base_tlogit).mean())
        sc_d = float((sc_t - base_tlogit).mean())
        d = decide(ko_d, sc_d)
        rows[f"{L},{H}"] = {"alpha": ALPHA[(L, H)], "scale_factor": round(s, 4),
                            "knockout_dlogit": round(ko_d, 4), "scale_dlogit": round(sc_d, 4), **d}
        print(f"  L{L}.H{H}: alpha={ALPHA[(L,H)]:.3f} knockout_dlogit={ko_d:+.4f} scale_dlogit={sc_d:+.4f} [{d['verdict']}]", flush=True)

    out = {"name_it": name_it, "tag": tag, "substrate": "random-token induction", "n_prompts": N_PROMPTS,
           "k_rep": K_REP, "readpos": readpos, "induction_copy_acc": round(copy_acc, 3),
           "k_tol": K_TOL, "scale_tol": SCALE_TOL, "alpha_source": "results_27b_ovmag",
           "decision_rule": "INACTIVE if |knockout|<K_TOL; else MATTERS if |scale|>=SCALE_TOL else NEGLIGIBLE_GAIN",
           "measurements": rows}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    Path("out").mkdir(exist_ok=True)
    Path(f"out/ov_behavioral_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[done] wrote out/ov_behavioral_{tag}.json", flush=True)


# --------------------------------------------------------------------------- selftest (model-free)
def selftest():
    # decide branches
    assert decide(0.02, 0.5)["verdict"] == "INACTIVE"                 # knockout ~0 -> inactive (gates first)
    assert decide(2.0, 0.3)["verdict"] == "MATTERS"                   # active + scale moves it
    assert decide(2.0, 0.01)["verdict"] == "NEGLIGIBLE_GAIN"          # active but gain change tiny
    assert decide(-1.5, -0.2)["verdict"] == "MATTERS"                 # sign-agnostic (uses |.|)
    print("[selftest] decide: INACTIVE / MATTERS / NEGLIGIBLE_GAIN fire (knockout gates first)")

    # induction builder: target is the k-1'th token; sequence is the doubled list; readpos = 2k-2
    seqs, tgt, rp = induction_batch(1000, 2000, 4, 5, seed=1)
    assert len(seqs) == 4 and len(seqs[0]) == 10 and rp == 8
    for s, t in zip(seqs, tgt):
        assert s[:5] == s[5:], "sequence must be the k tokens repeated"
        assert t == s[4], "target = k-1'th original token (predicted from readpos 2k-2)"
        # the token AT readpos+1 (the ground-truth next) equals the target
        assert s[rp + 1] == t
    print(f"[selftest] induction batch: seq=doubled, target=tok[k-1], readpos={rp} (predicts pos {rp+1})")

    # scale factor restores base gain: it=alpha*base, so z*=1/alpha -> base magnitude
    for (L, H), a in [((17, 4), 1.518), ((23, 24), 0.753)]:
        assert abs((1.0 / a) * a - 1.0) < 1e-9
    print("[selftest] scale factor 1/alpha inverts the it gain (alpha from results_27b_ovmag)")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--name-it", default="google/gemma-2-27b-it")
    ap.add_argument("--tag", default="27b")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.name_it, a.tag, a.device)
