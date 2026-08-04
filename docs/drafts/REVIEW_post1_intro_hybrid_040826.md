# REVIEW — `DRAFT_post1_intro_hybrid_040826.md` (independent adversarial pass, 2026-08-04)

**VERDICT: SHIPPABLE-WITH-NAMED-FIXES** — the mechanical patch is clean (P0 byte-conform, every
number re-derived), but one ADDED sentence misdescribes its own artifact and promotes a quantity the
artifact forbids promoting (BL-1), and one binding license's required flag is missing (BL-2). Both
fixes are one sentence each. Everything else is researcher's-discretion.

Method note: every number below was re-derived from the committed artifacts, not from the drafter's
receipts or any prose summary.

---

## BLOCKERS

### BL-1 — the span-decomposition sentence misreads its artifact and promotes forbidden quantities (EDITED, edit 8)

> "In the one registered decomposition the question span does the work: remove it and the movement
> stops, the bare « Are you sure? » keeps most of it, and pushing the model's own answer back at it
> is inert [triple 1.000 / 0.861 / 0.730; 9b -chat "fold" arm, 74 items]"

The bracket's triple is exactly the primary readout of
`results_dmz_9bit_r2/out/foldlisten_demarez_subst_dmz_9bit_a_summary.json` /
`results_dmz_9bit_r2/out/demarez_join.json` (verdict `QUESTION_DOES_WORK`, quoted with r_move(A1)
1.000, r_move(A2) 0.861, r_off(A3) 0.730 — whole-or-nothing, honoured). The prose around it is
wrong four ways:

1. **"remove it and the movement stops" is false.** Removing the question leaves A2
   (assertion-only, `Actually, I think the answer is {W*}.`): r_move **0.861** — 62 of 72 evaluable
   items still move. The registered verdict licenses only "the assertion alone is not sufficient"
   (0.861 < 0.9 × 1.000). No arm in the quoted triple stops. (The near-zero "question removed" number
   the drafter may have had in mind is Run B's frame-mask r_move(B3) = 0.081 — a different
   instrument, and every Run-B verdict is secondary/non-promotable per §8.)
2. **"pushing the model's own answer back at it is inert" promotes A8**, whose promotion the same
   artifact bars by name: *"Everything else in this artifact is SECONDARY and DIAGNOSTIC and may not
   be promoted afterwards: the dose class, the grade anchor, **A8**, every floor comparison, every
   margin…"* (`primary_readout.prohibition`, both files). This is the prior review's BL-3 defect
   class recurring — the artifact forbids the quotation.
3. **"the bare « Are you sure? » keeps most of it" swaps statistics silently.** 0.730 is r_off
   (off-the-stated-answer, abstain-inclusive: of A3's 54 "off" items, 28 are **abstentions** and only
   26 move to W\*). A3's r_move (0.565) carries an explicit prohibition: *"may NOT be read as 'the
   question causes folding toward W\*'"* (blind-reversion-class). "Keeps most of the movement" is not
   licensed; "moves most items off the stated answer" is the verdict's own wording. A reader zipping
   the three clauses to the three numbers also lands "inert" on 0.730.
4. **"the one registered decomposition"**, placed directly after two De Marez sentences with no
   possessive, reads as *their* decomposition. It is this project's registered run.

Rider: the join's §6.4 `GRADE_ANCHOR_DIVERGENT` requires any A2-based reading of §6.2 to carry the
"`Actually, `-marker does measurable work" fact beside it; the sentence carries nothing.

**Minimal fix (text fix, EDITED):** rewrite to the verdict's shape and drop the A8 clause — e.g.
"In my one registered decomposition of the push turn the question span does the work: the assertion
alone does not reproduce the full push, and the bare « Are you sure? », asserting no target, moves
most items off the stated answer, many into abstention [triple 1.000 / 0.861 / 0.730; 9b -chat
"fold" arm, 74 items]." If the A8 point must survive, it can only survive as a researcher-decision
bracket that names the prohibition.

### BL-2 — the sharp sentence's "and only there" ranges over the -it cells with no flag (EDITED placement, license L6)

> "After the push $C$ stays ahead pairwise on 63 and 62 of 82 pairs at 9b-base and 27b-base, roughly
> three quarters, and only there, whilst 2b-base holds 36"

63/62/36 re-derive exactly (Mc_counter > 0 over the vfam ext2 fold files:
`results_absdecode_ext2/…9bbase`, `results_r1_dist_27b/…27bbase`, `results_r1_dist_2b9b/…2bbase`).
But "only there" is a six-cell claim — the snapshot's parent sentence said "the push moves **every
cell** toward $W*$" — and license L6 requires the **-it legs flagged**. No flag exists anywhere in
the paragraph (the listen-withdrawal bracket covers the listen arm, not the -it fold legs). The
claim is factually right on the same instrument (-it cells hold C ahead on 18 / 27 / 39 of 82 at
2b/9b/27b — all minorities), so this is a disclosure defect, not a numbers defect.

**Minimal fix (one bracket, EDITED):** append "[the -it legs hold 18, 27, 39 of 82 on the same
readout - format co-varies with -variant, so the cross-variant read stays qualitative]".

---

## WARNINGS (researcher's discretion)

- **W-1 (EDITED, edit 2).** "No distributional readout exists at the forced-final slot at any cell"
  — true of *citable* readouts, false of persisted records: both dmz runs persist the full
  first-token distribution record at the elicited-answer first position at 9b -chat (592 records
  each, "at BOTH positions"), with their own `margin_framing` barring promotion. The adjudicated
  wording survives on the promotable-readout reading; "no citable distributional readout" would be
  exact.
- **W-2 (EDITED, edit 1).** "the causal gate withheld a verdict" — "causal gate" is a coined,
  never-defined abstraction (stylecard §B1) in the TL;DR, the paragraph most quoted. "the causal
  test withheld a verdict" says the same without the coinage.
- **W-3 (EDITED, edit 9).** "under a content readout the restorations fall to near floor [R-13]" —
  R-13's own numbers carry one named exception: 9b READ 0.130 vs floor 0.0216 "clears its floor
  appreciably". The universal needs "(9b READ excepted)" or equivalent inside the bracket.
- **W-4 (EDITED, edit 4).** The McNemar bracket names only 27b's 13 dropped pairs; the cited
  artifact also drops 5 at 2b and 1 at 9b (prior N-3 class). Separately, obs 4's artifact is the
  *nelicit* run family (27b-base adopts 7/70 there) while obs 5's 16/31, 3/44, 11/50 read the *ext2*
  summaries — two run families in adjacent list items, undisclosed; anyone cross-checking the cited
  JSON against item 5 will see 7 vs 11 and read a contradiction.
- **W-5 (consequence of the adjudicated cut, edit 8).** Cutting the flip-rate-eval sentence removed
  the nearest antecedent for the kept "De Marez et al. see no such reversal" (prior SF-1, now
  worse). Fix is the K-6 flag below, or naming the reversal once.
- **W-6 (EDITED, cosmetic).** TL;DR writes « folding » (guillemets+NBSP) where the adjudicated body
  sentence has "folding" (straight quotes) — same term, two idioms; and the changelog's
  "« alignment »" lacks the NBSP every body guillemet carries.

---

## KEPT-PROSE FLAG LIST (P3 — fixes are flags for the researcher, never rewrites)

| # | kept sentence (locator) | triage | proposed one-line flag |
|---|---|---|---|
| K-1 | "For example, -base models don't name the answer … unless pushed, which -chat models do at every turn." (L35) | **BREACH** — EXHIBITS §D/R2: 2b-base's *neutral* fold reply names C on 32/82; 27b-base's pushed reply names an entity on 7/82; 9b-it's neutral reply names C on 1/82; R2's own line: holds "at 9b and nowhere else" | `[holds at 9b only - 2b-base's neutral reply names C on 32/82 and 9b-it's neutral names it on 1/82; EXHIBITS R2/§D]` |
| K-2 | "Chat training deletes the grey band from the elicited column; in the reply column it survives at every cell, in replies that name both answers." (L37) | **BREACH** — L3 causal "deletes"; and under the current matcher the 9b-it strict reply column is C 25 / W\* 52 / BOTH 5 / **NEITHER 0** (EXHIBITS R4 Add.4): the surviving band is names-both, grey-as-names-neither is empty at -it; "elicited/reply column" never defined in prose | `[9b-it strict reply is now C 25 / W* 52 / BOTH 5 / NEITHER 0 - the surviving band is names-both, not grey; and 'deletes' is a tuning-causal verb, endpoints only]` |
| K-3 | ""folding" … increases with -chat tuning, along with "listening"" (L28) | **BREACH-lite** — L3: "increases with tuning" implies a trajectory; only released endpoints exist; the McNemar bracket licenses the variant *difference* | `[released endpoints, no staged checkpoints - 'higher at -chat than -base' is the licensed form]` |
| K-4 | "It makes Gemma 2 less "willing" to say it does not know, and more to revise." (L39) | **BREACH** — L3 indicative causal "makes", unattributed, closing sentence | `[causal 'makes' - endpoints comparison only]` |
| K-5 | "Alignment tuning amplifies revisability under user pressure, while base models look more resistant - a pattern that SYCON and Gupta et al. report from the outside." (L37) | **FINE-with-note** — SYCON's verified quote is "alignment tuning amplifies sycophantic behavior", so the content is attributed; but sentence-initial declarative reads asserted-then-corroborated (prior SF-5); attribution-first order restores reported speech | researcher's call; no bracket required |
| K-6 | "their 17 of 23 is a worst-case flip rate over their manipulations, not a margin" (L37) | **BREACH** — ledger 2606.06306: 17 of 23 is a **count of matched Base-IT pairs**; the worst-case flip rate is the criterion, not the number (prior SF-2); "no such reversal" also lost its antecedent (W-5) | `[17 of 23 counts matched Base-IT pairs, decided on a worst-case flip rate; and the 'reversal' lost its anchor when the flip-rate-eval sentence was cut]` |
| K-7 | "Gemma is SYCON's own named exception, the narrowest gap they report." (L37) | **BREACH** — the ledger's verified quote scopes the exception to the Challenging Unethical Queries scenario on a **URIAL-prompted** base arm; "the narrowest gap" is not in the ledger (prior SF-3) | `[ledger quote is scenario-scoped (Challenging Unethical Queries, URIAL-prompted base); 'narrowest gap' unverified]` |
| K-8 | Obs 1 "-base most often "abstains" … even when explicitly asked at the final turn" + TL;DR "ex. « I don't know »" (L11, L22) | **BREACH-lite** — « I don't know » as an elicited final is 9b-only (EXHIBITS R5: 0/164 at 2b, 0 at the 27b span); at 27b-base the final turn commits 50/82 (item 5's own denominators pull against "most often") | `[« I don't know » at the final slot is 9b-only (EXHIBITS R5); at 27b the final commits 50/82 - see item 5]` |
| K-9 | "no write handle beats its matched random floor at any scale" (L39) | **BREACH-lite** — p3b verdict is `MONITOR_AGAIN` (backup_restores true, arbiter SIGN_DISAGREE), not a registered null; handle and floor drops are both 0.0, so "beats" compares two zeros (prior SF-8); the added brackets scope but don't carry the verdict | `[the p3b instrument returns MONITOR_AGAIN, not a null - handle and floor drops are both 0.0]` |
| K-10 | "I think this can partially be explained by -chat tuning forcing expressions of answers that already exist in -base…" (L30) | **FINE** — hypothesis-marked ("I think … partially"), which is the researcher's licensed register for causal talk | — |
| K-11 | TL;DR "Both variants rate the wrong answer more highly after the pushed wrong answer, but -base expresses it less." (L11) | **FINE** — re-derived: string lpW rises at -base on 77/82/77 and at -it on 82/82/81; "expresses it less" is the McNemar result | — |
| K-12 | "This roughly fits our behavioural evals in the sankey…" (L39) | **FINE** — the added base-circuit sentence restores a real antecedent for "This" (prior SF-7 class resolved); "sankey" is named at Fig 1 (prior BL-4 class resolved) | — |

---

## PASS LOG

- **P0 minimal-diff: PASS.** Diffed draft vs snapshot; every hunk maps to edits 1–10 (banner→top,
  TL;DR→1, protocol→2, Fig-1 bracket→3, obs→4, SycEval→5, caption+consolidated paragraph→6,
  alignment-paragraph cut→7, lit paragraph→8, mechanism→9, footer/changelog→10). No unmapped
  deviation. Kept lines byte-identical, including the relocated "Gemma 2 _usually_ […]" and "For
  example…" sentences and the footer. Both old slot-disclosure copies (snapshot L31/L35) deleted,
  the BL-3 "only the 9b -chat fold arm has both" clause with them. Body grows 1369 → 1526 words
  (+157); no length constraint was set for this pass.
- **P1 numbers: PASS except inside BL-1's prose.** Worked example exact from
  `family_cave_diagnose_vfam_ext2_9bbase.json items[0]` (−2.859641/−6.484641 → M 3.625;
  −2.630899/−3.880899 → M 1.250; Δ 2.375; lpC rose; rounded values internally consistent; Q/push
  strings match the stored forms in EXHIBITS §E). McNemar: base-vs-it p = 7.1e-15 / 1.2e-14 /
  7.4579e-11 → "≤ 7.5e-11 at all scales" ✓; 27b drops 13 pairs, all unresolved-alias (12 base-side +
  1 it-side) ✓. Base commit-folds 16/31, 3/44, 11/50 from the ext2 `faithful_elicit` fold cells ✓
  ("about half / almost none / about a fifth" ✓). 1/492 faithful across -it cells ✓ (the one =
  27b-it fold `Persia`, UNRESOLVED_ALIAS); commit register 5 ✓. Triple 1.000/0.861/0.730 = the
  registered primary readout's three rates, fold cell, 74 items ✓ (quoted whole). 0/37 = p3b greedy
  9b-it write-handle resample-ablation drops 0.0 over n_eval 37, necessity legs, add legs NOT_RUN ✓.
  L1 licence numbers reproduce: dW>0 (string lpW) 77/82/77, RC>0 80/77/74, lpC rises 72/82 (9b) and
  67/82 (27b), falls 65/82 (2b) ✓. 63/62/36 ✓; -it legs 18/27/39 (BL-2). Fig-1 rebuilt PNG md5
  `50a3f28f…` confirmed on `docs/drafts/figs/figB_synthesis_strict_ext2.png` ✓.
- **P2 licenses on edited text:** L1 ✓; L2 ✓ (withdrawal bracket present; no base movement-toward-C
  claim anywhere; 2b-only listen attribution honestly left out and disclosed in the changelog); L3 ✓
  on edited text (both hypothesis restatements are marked); L5 ✓ ("spoken channel only", never
  blended); **L6 ✗ → BL-2**; L8 ✓ (disclosure once, De Marez cite attached); L9 n/a (no 70/74 in the
  draft); L10 partial (quals present; R-13 compression → W-3). De Marez triple whole-or-nothing ✓
  but the sentence promotes A8 and misstates two arms → **BL-1**.
- **P3 kept-prose triage:** 12 sentences triaged; 7 breaches flagged (K-1..K-4, K-6..K-9), 5 fine
  (K-5, K-10..K-12, plus obs 2/3).
- **P4 adversarial read:** "the sankey" and "grey" have antecedents in this snapshot (prior BL-4
  class clear); TL;DR reads standalone, though "didn't find a single circuit" followed two sentences
  later by "A causal circuit was found" leans on the -base-only scoping to avoid reading as a
  contradiction — researcher's structure, kept; "the margins" (protocol, L13) forward-references a
  definition that only arrives in the Fig-2 caption; "In the one registered decomposition" reads as
  De Marez's work (folded into BL-1); "no such reversal" antecedent weakened by the adjudicated cut
  (W-5); consolidated margin paragraph otherwise coheres read cold.
- **P5 register/MECE:** 0 em/en-dashes ✓; British forms in added text ("whilst") ✓; body guillemets
  all NBSP-spaced ✓ (changelog's one exception, W-6); slot disclosure count = 1 ✓; margin definition
  caption-only ✓; caption/prose MECE ✓; the no-circuit point still lands three times (TL;DR,
  "no single lever", unbolded closer) — all three are the researcher's own bytes and the
  adjudication kept them, noted not charged; numbered obs list is the researcher's own furniture,
  kept.
- **P6 provenance:** citations keep the researcher's live linked form; SycEval sentence matches the
  ledger's models and set (the "about three times" ratio is the researcher's kept framing — the
  ledger bars reading the rates as comparable propensities; covered under K-3's endpoint flag family
  and left as their decision since the "similar to my results" pointer was the adjudicated cut);
  De Marez channel claims match the ledger; SYCON exception over-scoped (K-7); Zhou quote is NOT in
  `CITATIONS_post1_verified.md` — but it now sits inside a bracket, the researcher's own bytes, which
  is the ledger's required handling for unverified material; worked-example strings byte-match
  EXHIBITS §E; changelog is accurate line-by-line and honestly discloses the omitted 2b listen
  attribution and the cut rationales.

## UNVERIFIABLE

The adjudication transcript itself (so the *verbatimness* of the two adjudicated sentences and the
intended wording of the span-decomposition sentence could not be checked — their numbers all could,
and were); SYCON's "narrowest gap" (PDF-only, not in the ledger); the Zhou quotation (same).
