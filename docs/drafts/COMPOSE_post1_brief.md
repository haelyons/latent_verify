# COMPOSE post1 — composition brief for the final intro + notes pass

Date: 2026-07-30. Session: orientation + six isolated read-only investigations (structure
extraction, drafts/figures inventory, core-unrun verification, two H3 grounding passes, a
framing→territory map). $0 GPU, no result artifact touched, vault untouched, nothing here
drafts prose for the post. Every number below was re-derived by an isolated reader from the
artifact named next to it; anything not reproducible is flagged as such.

Scope note: this brief JOINS the six investigations for the researcher's final composition
pass. It supersedes nothing; where it repeats `JOIN_post1_crossvariant_scale.md` or
`GROUNDING_crossvariant_scale.md` it is citing them, and those stay authority.

---

## §A. Latest strands (what landed since the researcher's last edit, newest first)

- `1f28ea6` (2026-07-29) — every 27b figure repointed to the reproducible decode
  (`results_foldlisten_nelicit_27b/`), two new figures built
  (`figB_listen_strict_allscales.png`, `fig_topk_ankara_9bbase.png` = the empty Fig-3b slot),
  the neutral-ELICITED fourth column lands in both counterfactual figures,
  `RESULTS_FOLDLISTEN.md` Addendum 10 corrects the mask-survivor forensics,
  `docs/drafts/figs/VAULT_SYNC_NOTE.md` maps repo PNG → vault embed.
- `7e867f6` — count correction: 3 of 6 within-variant scale comparisons are null
  (`out/gapclose_foldrate_sig.json`), not 4 of 6.
- `d9d884b` — `docs/drafts/PATCHSET_tranche3.md`, 24 blocks resolving the
  cross-variant/scale brackets. **Zero applied**; all hand-apply, vault never written.
- `1853e27` — `docs/drafts/JOIN_post1_crossvariant_scale.md`: the five-investigation join,
  §D core-unrun ranking, §E quotation rules. Four stale ledger lines got dated addenda.
- **Neutral-elicit run VERIFIED against artifacts** (this session, isolated reader): all 12
  ext2 cell-directions have `n = n_neutral_elicit = 82`
  (`results_foldlisten_nelicit_{2b9b,27b}/out/`). Gate state, adjudicated not waived:
  4/6 cells `BYTE_IDENTICAL` (`out/foldlisten_repro_diff_fl_*.json`); the two 27b cells are
  `DIFF`, triaged by `out/27b_decode_determinism_result.json` — the nelicit re-run is
  byte-identical to pass A (4428 fields, 0 mismatches), the COMMITTED ext2 draw is the
  anomaly. The three headline numbers all reproduce:
  27b-base fold elicit **41/7/34**, listen **16/31/35** faithful
  (`foldlisten_judge_fl_27bbase_ext2_summary.json`, `cells_faithful`);
  9b-base withholding **push-INVERTED** — neutral-elicited abstain 52 vs pushed 38, verdict
  `INVERTED_NEUTRAL_HIGHER` (`foldlisten_judge_fl_9bbase_ext2_summary.json`,
  `push_attribution_faithful.cells.fold`);
  9b-it listen **25/82 move to C unprompted** (`..._9bit_...`,
  `cells_faithful.listen.neutral_elicit = {moved 25, held 55, abstain 2}`).
- **Untracked file present:** `docs/drafts/REGISTRATION_demarez_spans.md` (mtime 2026-07-29
  22:16, `git status ??`). Likely a parallel session's; not read here, not vouched for.

## §B. Where things live

Gold (never write): `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md`
(28 lines) and `..._notes.md` (345 lines). Patch application ledger:
tranche 1 (`PATCHSET_final.md`, 28 blocks) applied at `f403686`;
tranche 2 (`PATCHSET_tranche2.md`) 11 of 25 applied at `598de5e`, **14 HELD — the reasons
live ONLY in the body of commit `598de5e`** (no tranche-2 review file exists;
`RESEARCH_QUESTIONS.md`'s pointer to `REVIEW_patches_v2.md` is wrong — that reviews the
earlier patches_v2 round). Four of the held (C02, D10, D18, D19) have **no recorded hold
reason anywhere**. Tranche 3 (`PATCHSET_tranche3.md`) 0 of 24 applied.
File defect: tranche 3 contains a duplicated body — lines 55–385 (T3-01…T3-14) repeat
byte-identically at 422–752, with a stray `### ` welded onto the preamble sentence at
line 388. Unique blocks are still 24; a `^### T3-` or `^ITEM:` scan double-counts.

Figures: `docs/drafts/figs/` — 12 build scripts, 24 PNGs, 7 caption files,
`CASE_STUDIES_pushback.md` transcripts. Per `VAULT_SYNC_NOTE.md` + md5 check this session:

| vault embed (researcher's swap to do) | repo PNG (current) | why stale |
|---|---|---|
| `figB_synthesis_strict_ext2.png` (intro Fig 1 / notes "Fig 4") | same name, rebuilt | vault md5 `6942c40b…` = the ANOMALOUS 27b draw (fold 39/11/32, listen 20/34/28); repo `50a3f28f…` = reproducible draw (41/7/34, 16/31/35) |
| `IMG_3919.png` (notes Fig 3) | `figB_fold_strict_allscales.png` | same 27b repoint |
| `IMG_3917.png` (notes Fig 1) | `figB_neutral_counterfactual_ext2.png` | gains the neutral-elicited 4th column — resolves the notes-L137 bracket |
| `Pasted image 20260724190541.png` (notes Fig 4-listen) | `figB_neutral_counterfactual_listen_ext2.png` | same 4th column; carries the 25/82 unprompted-listen number |

Also: `figB_fold_ext2.png` / `figB_listen_ext2.png` are stale against their own script
(`make_figB_sankey.py` pointers updated, PNGs deliberately not rebuilt — warning at
`make_figB_sankey.py:12`). New figures with no vault embed yet: `fig_topk_ankara_9bbase.png`
(Fig 3b slot), `figB_listen_strict_allscales.png`. Vault-root `image-1785076502213.png` is
unprovenanced — do not cite off it.

Grounding authorities for drafting: `docs/drafts/CITATIONS_post1_verified.md` (with
`GROUNDING_crossvariant_scale.md` §11 superseding parts of it — see §D flag on Perez),
`EXHIBITS_post1_grounded.md`, `NOVELTY_boundary_post1.md`, `STYLECARD_researcher.md`,
`HOLES_post1_v2.md` (108-marker hole map), JOIN §E quotation rules.

## §C. Framing → territory (MECE)

**The story the intro already tells, and the territory backs (verified this session):**
fold+listen, 82 pairs, 12 cells at 2/9/27b — every panel of Fig 1 reproduces from per-item
records (all 12 elicited cells match `make_figB_sankey.EXPECT`; the base strict-counter
column re-derived independently). -it never abstains at the final answer — 1/492 across six
-it cells, the one exception is item "birthplace of chess" answered "Persia",
`faithful_rule_elicit = bare_alias_miss`, draw-independent — **true in the faithful register
only** (commit register has 5, all named entities, none silences). "Folds significantly
more": exact McNemar per scale, p = 7.1e-15 / 1.2e-14 / 7.5e-11, all DIFFERS
(`out/gapclose_foldrate_sig.json`) — with 13/82 pairs dropped at 27b as unresolved-alias,
which "significantly" should not hide. "Usually assigns higher probability to C": holds at
all six cells on the content margin (54–74 of 82; `family_cave_diagnose` M0), fails at 2b-it
on the first-token readout (28/82). The push moves the distribution while C stays ahead **of
W\*** — 57/82 at 9b-base, 50/82 at 27b-base moved with C still pairwise-ahead — but C is the
vocabulary argmax at that slot on ~0/82 items (top risers are " Yes"/"You"), and W\* is never
the top riser; "remains highest probability" must become a pairwise statement.

**Territory deliberately out (keep out; circuit-post material per the directional
commitments):** the doubt circuit (3/3 scales incl. the 27b discharge), social-source
gradient, downstream-distributed / 2b attribution graph, cave-direction-is-a-handle, the
standing nulls (no installed head-set, no entropy neuron, no confidence gate), RLHF-edits-no
-copy-weights, mid-stack THINK-probe depth, verifier arc, Yang & Jia read.

**Territory with no home yet — CANDIDATES for the notes (completeness, not obligation):**
1. Taxonomy of withholding — one label, three phenomena by scale (2b 76% asserted
   confidence, 9b 53% genuine uncertainty, 27b 94% off-target; 33 of 34 genuine-uncertainty
   withholds are 9b-base — publishable form 33 of 39 across draws)
   (`docs/drafts/TAXONOMY_withholding.md`). The intro's TL;DR "abstains and hedges"
   currently generalises the 9b reading.
2. The neutral-elicited column and its two findings (§A) — including that base
   push-attribution reads `INVERTED_NEUTRAL_HIGHER` at 9b/27b-base, which cuts against notes
   L131's "any change must be attributable to the pushback" as written.
3. The format-artifact result — base-vs-it bare-rank gap is a FORMAT artifact; the triple is
   quotable whole or not at all (`out/fmt_matched_join.json`).
4. Monitor-not-lever at -it, 3/3 scales — the honest content behind intro L25's second half.
5. The 27b two-draw disclosure — mandatory clause whenever a 27b digit is printed.
6. Judge demotion (self-judge failed human validation 0.679 vs commit 0.982, n=56) — the
   persisted, true version of the notes-L76 anecdote.
7. Scale non-separation — 3 of 6 within-variant comparisons null; 9b and 27b never separate;
   the cross-VARIANT gap is the decisive axis, not scale. Bears directly on the
   "« Sycophancy Scaling Laws »" heading (notes L300).

**Post without territory (must change or carry the honest bracket):** see §D per line.
Beyond the intro: notes L74 "greedy… ensuring determinism" (refuted at 27b), L76 judge
anecdote (no persisted run), L200 "67 of 74" (withdrawn, RETRACTIONS R-6; replacement 73/74
register C70/W\*3/N1), L274 "sufficient AND necessary" (copy-KO never necessary; head-SET
retracted under power), L279 "not present in chat models" (routing weights intact — not-used
≠ not-present), and the stale "no top-k run exists" brackets at L295/L311/L342 (false since
the R1 fill).

## §D. The intro, line by line (grounding verdicts)

- **L5 TL;DR** — GROUNDED in the faithful register; sentence must name the register (see
  §C). T3-01 is the fix; **coupled with T3-21** (notes L133) — apply both or neither.
- **L7 protocol** — GROUNDED (82 pairs, provenance re-derived 91→87 KEEP→82). Two wording
  precisions owed: the plant is TEACHER-FORCED into the model's own turn, not "prompted
  with"; base cells are raw `Q:/A:`, -it cells chat turns — format co-varies with variant.
- **L15 obs 1** — GROUNDED; the researcher's own bracket is exactly right ("I don't know."
  is the forced answer 6/82; the reply's hedge is "I'm not sure" 56/82, modal "No, I'm not
  sure. I'm just guessing." ×37). 27b caveat: ~⅓ of the grey is alias-unresolvable, not
  hedging (12/34 fold, 15/35 listen).
- **L17 obs 3 "significantly"** — GROUNDED (§C McNemar). Quote with the 27b UA-drop
  disclosure.
- **L19 "carrying, or 'copying'"** — behavioural carry backed (9b-it 137:27
  pushed:planted; "base runs the other way" holds at 9b/27b, FAILS at 2b-base 41:25 — scale
  qualifier owed); "copying" as MECHANISM is contradicted by hardened territory (copy-KO
  never necessary at base). The researcher's "[does it? …repeated several times]" bracket is
  right: this is duplication pair A12 (§E).
- **L21 De Marez paragraph** — "[this needs a major revision]" confirmed. What survives:
  the pairwise-margin phenomenon (§C numbers). What must change: "remains highest
  probability" → pairwise-vs-W\* only; the readout named (content margin, not first-token);
  and the whole paragraph reads at the reply-to-challenge slot while the sankey's verdicts
  are at the forced-final slot, where **no distribution or residual read exists at any cell**
  (OWED B2) — the honest disclosure sentence for "its going in the lab notes".
- **L23** — the researcher's three brackets are all confirmed. The 17/23 is verbatim-correct
  BUT the paragraph misdescribes De Marez: in their data BOTH channels (flip rate AND
  margin) favour IT — their flip rate does not flatter base; what "runs the other way" is
  THIS post's spoken-answer readout with its abstain outcome
  (`GROUNDING_crossvariant_scale.md` §11). SycEval "on different math-based examples" is
  wrong in both directions (the ~3× is pooled math+medical; maths ≈9:1, medical ≈1.06:1) —
  T3-02 is the READY fix. SYCON: Gemma is their NAMED exception (narrowest base-vs-it gap) —
  usable as support, not just contrast. Zhou carries a stronger unused quote ("In base
  models, we see a preference for weakeners but the trend reverses among RLHF models") —
  the closest published neighbour to the grey-band deletion. Unresolved ledger conflict:
  `CITATIONS_post1_verified.md:29-31` says cite Perez 2212.09251 as inverse-scaling;
  GROUNDING §11 says that is backwards (Perez: sycophancy flat in RL steps incl. 0). Do not
  cite Perez either way until reconciled.
- **L25** — MISMATCH as written, and the researcher's "which run is it?" has a sharp answer:
  no run. Verified this session from `results_fold_vs_listen{,_2b}/out/`: base fold∩listen
  top-5 overlap is **4/5** (9b and 2b; no 27b run exists) — but **-it overlap is 5/5 at both
  scales**, i.e. MORE shared, directionally contradicting "at -chat, this mechanism is
  distributed". The "distributed" impression comes from a different, unmatched, -it-only
  instrument (phase 3a/3b), and what THAT shows is *no single lever* (read-side greedy
  selects nothing; write resample-ablation flips 0/37 at 9b; `MONITOR_AGAIN` at 3/3 scales),
  not distributed heads. All four fold-vs-listen cells are `MOVE_UNMATCHED` (causal gate
  failed) — the base result is correlational only. T3-03 holds the honest replacement and is
  the one NEEDS-RESEARCHER-DECISION block.

## §E. MECE debt inside the two docs (structure extraction, 2026-07-30)

Notes: 85 top-level prose brackets across 59 lines; intro: 11. Six relegated sections
(L77, L132, L195, L199, L205, L269 — tag spelling inconsistent).

- **Duplication intro↔notes, 15 pairs.** Highest-value: the intro's three numbered
  observations (L15–17) vs notes L306–313 are two prose readings of the SAME image
  (`figB_synthesis_strict_ext2.png` is intro Fig 1 AND notes "Fig 4"); De Marez appears at
  intro L21/L23 and notes L181; the mechanism claim at intro L25 and notes
  L200-202/L274/L279; the alias-miss correction bracket is written twice (intro L5, notes
  L133 — T3-01/T3-21 resolve as a pair).
- **Prose restates figure.** Intro L9 recites the legend the figure draws; notes L139, L149,
  L181, L192, L207–215, L248/L252/L261 (L261 restates L248 nearly verbatim; L252 begins
  mid-sentence — the "lost head clause" decision), L271–276, L286–290-vs-L293/295, L306–313.
- **Figure numbering** (researcher-only decision, unchanged): "Figure 4" used twice in
  notes; Fig 3b now has a BUILT figure awaiting the renumber decision; Figure 5 (L267) and
  "Figure N[big matrix]" (L248) unresolved; the intro/notes share one asset under two
  numbers.
- **L319 vs L321** — two drafts of the same two sentences, adjacent; survivorship is the
  researcher's (T3-05's replacement is written to move with whichever survives).

## §F. Core results still unrun (all five verified STILL UNRUN this session)

(a) **B2 forced-final distribution/residual read** — no instrument reads the slot the
verdicts are decided on, any cell (nelicit populates the slot behaviourally only).
Registration owed #2. (b) **Listen distributional column** — withdrawal stands
(`out/cleangate_same_box_result.json`: diagnose `NOT_NEUTRAL`); no replacement run.
Registration owed #1. (c) **Base mechanism arm + 27b `cave_fold_vs_listen`** — `assert
is_chat` ×4 (`foldlisten_phase2.py:155`, `3a.py:317`, `3b.py:734`, `3c_riders.py:325`); no
27b artifact in either arm. The only path to an intro-L25-strength sentence. Registration
owed #3/#4. (d) **Hand-labels for the headline cells** — 9b VF22, 9b-it ext2 (the post's
central cell), all base ext2, all listen, T3n: absent. What exists: spot-checks elsewhere +
9b-it fold-finals n=56 (1.0 agreement) which covers none of these. Registration owed #6.
(e) **-it top-k with a regime-aware key** — K4 ×14 multi-line, no chat path; the existing
-it top-k numbers carry the leading-space key confound (base share ~0.97 vs -it ~0.08–0.12).

## §G. Researcher-only gates (seven, across two seeds — collected here)

1. T3-03: whether the intro carries the mechanism contrast at all, and in what form (§D L25).
2. T3-01 + T3-21: apply together or not at all.
3. T3-16 only if T3-09 lands.
4. L319 vs L321 survivorship.
5. Figure renumbering (now interacts with the new Ankara figure in the 3b slot).
6. The lost head clause near L250/L252.
7. The L60 speaker tag (STYLECARD §A5 has no `Model (Elicited):` label).
Plus the four vault image swaps (§B table) — researcher's own, per VAULT_SYNC_NOTE.

## §H. Open-questions candidates for the post's close

From territory the post genuinely leaves open (each with its receipt): what the distribution
does at the forced-final slot (B2); whether the margin moves under a correct push (listen
column withdrawn); whether base and -it run the same revision machinery (base mechanism arm
unrun — the L25 question, stated as open); belief vs compliance beyond 9b (probe 9b-specific:
PROBE_INVALID at 2b, INSUFFICIENT_LABELS at 27b); why genuine uncertainty lives only at 9b
(described, not explained); whether W\*-plausibility holds in -it (needs the regime-aware
key); which training stage deletes the grey band (externally blocked — no staged Gemma
checkpoints; OLMo 2 / Tülu 3 would answer but break scope). Disclosure-grade, not a
question: the headline cells carry no human-agreement statistic.

## §I. Honest flags (things a reader cannot audit, or defects found)

- The -it strict-counter cells of Fig 1 are the only figure numbers NOT re-derivable from
  any saved label — classified live at build time, frozen only as `COUNTER_EXPECT`
  constants in `make_figB_sankey.py`.
- Tranche-2 held-block reasons exist only in a commit body (§B); four blocks have no
  recorded reason at all.
- `PATCHSET_tranche3.md` duplicated body (§B) — cosmetic, but any block census must dedup.
- `docs/drafts/REGISTRATION_demarez_spans.md` untracked, unread, unvouched.
- Perez citation: two ledgers disagree (§D L23); unresolved.
- The vault's live Fig 1 embed is the anomalous 27b draw (md5-confirmed, §B).
