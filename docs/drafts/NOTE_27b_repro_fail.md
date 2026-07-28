# 27b-base fails byte-identity, and the cause is almost certainly the GPU type — 2026-07-28

## What happened

`controls/foldlisten_repro_diff.py` on the neutral-elicit re-run of `fl_27bbase_ext2` against its
committed twin returns **DIFF**, not BYTE_IDENTICAL:

- **870 mismatched fields** (20 reported, 850 omitted) out of ~4428 compared — about 20%.
- Affected fields include **`counter_gen`, `neutral_gen`, `elicit_gen`** — the generations themselves,
  not merely derived labels. Everything downstream moves with them: `judge_prompt`, `judge_label`,
  `judge_reply_raw`, `elicit_prompt`, `commit_elicit`, `faithful_elicit`, `faithful_rule_elicit`.
- `new_arm` is `ARM_PRESENT_COMPLETE` (164/164 records carry all five new fields), so the neutral-arm
  addition itself worked exactly as designed.
- Worked example: fold item 2 — `conf_proxy` 5.364543774165213 → 5.48718001274392, and
  `judge_label` `WRONG` → `CORRECT`.

The cell's own decision is unchanged in kind: `NO_MOVEMENT`, fold 0.137 / listen 0.320
(committed twin also NO_MOVEMENT).

## Why hardware, not the code change

The same instrument, same commit, same family, produced **BYTE_IDENTICAL with zero mismatches** on
`fl_9bbase_ext2` and `fl_2bbase_ext2` in this very run — 4428 item-fields and 22 derived values each.
The judge change is additive and provably so on two cells.

The one thing that differed for 27b-base is the box. It ran on **H100 SXM5**, chosen this session over
the design's H100 PCIe on cap-risk grounds. Different GPU architecture means different kernel
selection, which means different bf16 accumulation order, which perturbs logits in the last bits. Under
**greedy** decoding a perturbation only matters when two tokens are near-tied — but when it does matter
the argmax flips, the generation diverges from that token on, and every field computed from it follows.
That is the observed signature: a small `conf_proxy` delta co-occurring with whole-string changes.

## The prediction, recorded before the data exists

Box 4 (`fl_27bit_ext2`) launched on **`gpu_1x_h100_pcie`** — the type the committed 27b artifacts used.
So:

- **If hardware is the cause:** 27b-it returns BYTE_IDENTICAL on PCIe, and the 2b/9b/27b-it cells all
  pass while 27b-base alone fails.
- **If the cause is 27b-specific numerics** (i.e. the model, not the box): 27b-it also fails, on PCIe,
  and the byte-identity assumption is false for 27b generally.

Either way the answer arrives free, on a box already paid for. Written down now so it is a test rather
than a rationalisation.

## Consequences

1. **The 27b-base neutral-elicited numbers may not be presented as an additive extension of the
   committed cell.** The whole cell moved. They are a valid greedy decode of the same items on
   different silicon, and nothing more.
2. **`DESIGN_neutral_elicit.md` §1.4's "must reproduce byte-identically" is false as stated** — it holds
   within a GPU type and not across one. It needs a hardware qualifier, and so does any future
   byte-identity claim in this repo. This is a specification defect, not a run failure.
3. **Neither run is "right".** Both are legitimate greedy decodes; the committed one is the published
   one. Re-running 27b-base on PCIe would test reproducibility properly, and costs ~$16 at the measured
   pace — not obviously worth it unless a 27b number becomes load-bearing.
4. **Cap arithmetic, separately:** 27b-base took **~4.9 h**, not the ~1.5 h the SXM5 estimate predicted.
   It ran at roughly the measured PCIe pace despite faster hardware, so the pace model in both open
   designs is optimistic and should be rebased on this datapoint before the next 27b launch.

---

## THE PREDICTION WAS REFUTED (2026-07-28, box 4 landed)

`fl_27bit_ext2` ran on `gpu_1x_h100_pcie` and **also returns DIFF**: 373 value mismatches, 55 label,
10 derived, `ARM_PRESENT_COMPLETE`. So the GPU-type hypothesis is dead as a sole explanation.

Final tally across the five gated cells:

| cell | gate |
|---|---|
| 9b-base, 2b-base, 9b-it, 2b-it | **BYTE_IDENTICAL**, zero mismatches of any kind |
| 27b-base | DIFF, 870 fields |
| 27b-it | DIFF, 438 fields |

**The split is by model size, not by box.** Everything at 2b and 9b reproduces to the byte; both 27b
cells fail, on two different GPU types.

### And the test I designed was weaker than I claimed

I asserted box 4 was running on "the type the committed 27b artifacts used". **That was an assumption,
not a known.** Checked afterwards: **no artifact in this repo records the hardware it ran on** — the
summaries carry `name`, `regime`, `cells`, `decision`, `cells_faithful`, `decision_faithful`,
`scorer_provenance` and (new) the `push_attribution` blocks, and no device, driver, dtype or instance
field; the fetched logs carry no instance type either. So the committed 27b hardware is unrecoverable,
the control condition for my test never existed, and neither hypothesis can be settled from what is on
disk.

That is a provenance defect worth more than the finding it obscured: **byte-identity claims in this
repo are unfalsifiable after the fact**, because the one variable most likely to break them is not
recorded. The cheap fix is to stamp instance type, driver and torch/transformers versions into every
summary alongside `scorer_provenance`.

### What the mismatch pattern does say

On 27b-it: `conf_proxy` differs on **162 of 164** records while `counter_gen` differs on **82** and
`neutral_gen` on 34. A float that is sensitive to the last bits moves almost everywhere; the discrete
generation moves only where the argmax was close enough to flip. That is the signature of numerical
perturbation rather than a logic change — consistent across both 27b cells, and absent at 2b/9b.

Plausible causes, none settled: 27b's larger reduction trees make bf16 accumulation more
order-sensitive; 27b may be loaded or sharded differently (the De Marez ledger entry refers to a
`27b-8bit` label elsewhere in the literature notes, so quantisation is worth ruling out); or the
committed 27b artifacts were produced under a driver or library version that no longer exists on any
box we launch.

### Consequences, updated

1. Neither 27b cell's neutral-elicited numbers may be presented as an additive extension of its
   committed twin. **Both cells moved.** 2b and 9b are clean and may be.
2. `DESIGN_neutral_elicit.md` §1.4's byte-identity requirement holds at 2b and 9b and **fails at 27b**.
   It needs a scale qualifier as well as the hardware one.
3. **The 27b-it substrate-gate contest is stable, which is the reassuring part.** Baseline and re-run
   agree: `gate[commit]` FAIL both, `gate[faithful]` PASS both. The contest recorded in the handoff seed
   is a property of the labels, not an artifact of one run.
4. The cell decisions are unchanged in kind at every scale: 27b-base NO_MOVEMENT (fold 0.137 / listen
   0.320), 27b-it MOVEMENT_BOTH (fold 0.675 / listen 0.988).
5. Owed, and cheap: stamp hardware and library provenance into summaries. Until then, do not assert
   byte-identity across runs at 27b.
