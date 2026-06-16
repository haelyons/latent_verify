# EXPERIMENT PLAN — the base↔chat transformation as the object of study

> Status board + method record for the next wave of attention-copy experiments.
> Written 2026-06-16. Observation-first: each module is a *measurement* with an
> explicit outcome-space, not a hypothesis to confirm. We minimise intermediate
> claims — the per-item distribution is the finding.

## The spine, in one sentence

`FRAMING_NOTES.md` establishes a copy mechanism (read a referenced prompt token,
write it to the answer slot) that is **clearest in base/fragment** (§3.7) and
**transformed by RLHF + chat format** (§6, §8, §9). The lit review and next-steps
analysis both land on the same point: *the transformation between base and chat is
itself the most interesting and least-understood object here.* So that transformation
is the backbone (ARC2A/B/C); the rest are supporting angles.

The key asymmetry we are chasing (from §8 vs §9):

- **Salience copy** (a salient entity → answer): RLHF appears to **remove** it — the
  reader head L18.H5's attention to the anchor collapses 0.84 → 0.01 on the
  *identical* prompt (§8). But §8 only measured the **attention pattern (QK)**; it
  never tested whether the **copy machinery (OV)** survives.
- **Numeric authority-sycophancy** (an asserted wrong number → answer): RLHF does
  **not** remove it — the latent pull in -it is undiminished, if anything larger
  (+6.84 vs base +4.79, §9). But the -it numeric mechanism was **never localized**
  (§10.2's W-span knockout + per-head map ran on base-2b only).

So we understand the part that *vanishes* better than the part that *survives into
deployment*. The spine fixes that.

---

## Module status board

| ID | Question (mechanism) | Readout | Needs | Script | Artifact | Status |
|---|---|---|---|---|---|---|
| **ARC2A** | Salience copy: did RLHF **delete** it or just **stop the head looking** (mask/gate)? | Transplant base L18.H5 attention into -it; force-attend the anchor; read capital−anchor margin. + OV-projection diagnostic | GPU (base+it) | `job_arc2a_transplant.py` | `out/arc2a_transplant.json` | scripted |
| **ARC2B** | Numeric copy *in -it*: still an attention-copy? through which heads? gated how? | §10.2 W-span knockout + per-head localize, on -it (chat), stratified by -it baseline correctness | GPU (it) | `job_arc2b_numeric_it.py` | `out/arc2b_numeric_it.json` | scripted |
| **ARC2C** | Is disengagement **format** (QA scaffold) or **weights** (RLHF)? | 2×2 {base,it}×{fragment,QA}, one readout: effect, all-heads knockout necessity, L18.H5→anchor | GPU (base+it) | `job_arc2c_format_weights.py` | `out/arc2c_format_weights.json` | scripted |
| SUPP-RED | Is L18.H5 *the* reader or a redundant pool w/ self-repair? | Ablate L18.H5 → re-measure other heads' anchor-attention + necessity; iterative knockout | CPU/GPU (base) | `job_supp_redundancy.py` | `out/supp_redundancy.json` | planned |
| SUPP-ATTR | Is framed L18.H5 its normal attribute-extraction job, captured by a distractor? | L18.H5 QK-source + OV-target on clean "capital of X is" vs framed (Ferrando) | CPU/GPU (base) | `job_supp_attr_extract.py` | `out/supp_attr_extract.json` | planned |
| SUPP-EARLY | Is the early-write attention or MLP enrichment (Geva)? | Zero early-layer MLPs at anchor pos; does L18.H5 still find the anchor / does the flip survive | CPU/GPU (base) | `job_supp_early_write.py` | `out/supp_early_write.json` | planned |
| SUPP-SUPP | Is there a mirror suppressor head, and does susceptibility track copy:suppressor ratio? | Head-level scan for *negative* necessity; regress flip on copy vs suppressor strength | CPU/GPU (base) | `job_supp_suppressor.py` | `out/supp_suppressor.json` | planned |
| SUPP-ARCH | Does the concentrate-vs-diffuse *routing* transport off the Gemma family (Franco)? | §3.7 + §10.2 knockouts on Pythia/GPT-2; per-head concentration metric per cue | GPU | `job_supp_crossarch.py` | `out/supp_crossarch.json` | planned |

Modules are mutually independent: each loads its own model(s), uses its own
stimuli/readout, writes its own artifact. A null or surprise in one informs but never
blocks another. Stage 0 (provision + reproduce §8 base controls) is the only shared
precursor and is instrument-validation, not a finding.

---

## Sample size — from first principles (why n, and how to spend it)

1. **Unit + dominant variance.** Replication unit = a stimulus (fact pair, or `(a,b)`
   product). Three noise sources: bf16 non-determinism (~2% on tail probs —
   negligible for means); **between-item variance — dominant** (`refine_heads`
   necessity CV ≈ 0.32–0.55 over 5 pairs); phrasing variance (currently unmeasured,
   1 wording/cue).
2. **Back out n.** For a per-head necessity, mean ≈ 0.2, CV ≈ 0.4 ⇒ SD ≈ 0.08. To pin
   it to ±0.05 at 95%: n ≈ (1.96·0.08/0.05)² ≈ **10**. Two-sample contrast (does cue A
   route through a head more than cue B, Δ ≈ 0.1): n ≈ 2·(2.8·0.08/0.1)² ≈ **10/group**.
3. **Why n=5 is fine for some claims and not others.** All-heads effect is large and
   low-variance (~1.0 vs ~0.05 control) → n=5 ample. **Per-head attributions are
   underpowered at n=5** (CV ≈ 0.4 demands n ≈ 10–15) — exactly why L18.H5's necessity
   swings 0.20 (free) → 0.046 (forced-choice) on the same pair.
4. **Stratify, don't just inflate.** The mechanism varies along known axes
   (baseline confidence → susceptibility; single- vs multi-token anchors; entity
   frequency). Use n ≈ 12–16 **per stratum**, strata along those axes. Report per-item
   distributions, not just means.
5. **Pre-commit the readout** (the repo already does): MIN_EFFECT = 0.5-nat floor
   (necessity below it is divide-by-~0 → n/a), teacher-forced full-number readout,
   median for skewed necessity, bootstrap CI on concentration metrics.
6. **Paired contrasts are cheaper.** ARC2A (same prompt ± transplant) and ARC2C (same
   readout across cells) are within-item → higher-powered, n ≈ 10 fine.

**Where to spend:** the existing 5 pairs / 60 products are the *pilot* that sets the
SD. Scale per-head + routing measurements (ARC2B, SUPP-RED, SUPP-ARCH) to n ≈ 12–16
per stratum; leave all-heads confirmations at n=5.

---

## How to run

**Instrument.** All new scripts use `transformer_lens.HookedTransformer.from_pretrained_no_processing`
(`dtype=bf16`), matching `job_chat_mechanism.py` / the §8–§10 GPU stack. They run on
CPU too (slow); base-2b modules are CPU-feasible, anything with -it / 9b / cross-arch
wants the A100. **This workstation has no torch** — model runs go to Lambda.

**GPU acquisition** (full detail: `docs/lambda-gpu-access.md`). From the workstation:
load `LAMBDA_KEY_ONE` from `.keys`, check `/instance-types` capacity, `launch`
`gpu_1x_a100_sxm4` (region fallback), poll `/instances/<id>` for `active`+IP, SSH in
with `~/.ssh/lambda_ed25519`. **Terminate when done — billed per hour.**

**On the box:**
```bash
git clone --branch claude/arc2-base-chat-spine https://x-access-token:TOKEN@github.com/haelyons/latent_verify
cd latent_verify && pip install -q transformer_lens transformers torch
python -c "from huggingface_hub import login; login('<HF_KEY_ONE>')"   # Gemma-2 licence
python job_arc2c_format_weights.py     # base+it, both built in
python job_arc2a_transplant.py
python job_arc2b_numeric_it.py
```

**Results-return (important):** `out/` is gitignored. Each run must
`git add -f out/<artifact>.json` then commit+push, or the JSON won't travel back.

---

## Findings log (claim-light; fill in as runs land)

- *(pending first GPU run)*
