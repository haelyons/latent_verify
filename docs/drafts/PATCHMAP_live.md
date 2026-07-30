# PATCHMAP_live — collision map for a new POST1 patch tranche

Read-only pass, 2026-07-30. Nothing was written to `/home/hal/Documents/`. This file is the only
artifact.

## Live gold state, measured this pass

| document | md5 | `wc -l` | split lines | trailing NL |
|---|---|---|---|---|
| `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md` | `83a55a14a8079403fa6be41c309c7f3b` | 28 | 29 | no |
| `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_notes.md` | `71c3b3c52236520189f0944232c4118a` | 345 | 346 | no |

Every "L*n*" below is a **split-line** number, i.e. what `Read` shows. All 24 tranche-3 line numbers
are stated against these md5s and all 24 verify.

**Load-bearing: the researcher edited BOTH gold files after `598de5e`.** The committed post-apply
snapshots are `docs/drafts/DARWIN_post1_user_intro_snapshot_280726.md` (`1bf7f06f…`, 29 `wc -l`) and
`..._notes_snapshot_280726.md` (`bad35792…`, 345). Live intro is `83a55a14…` at 28 `wc -l`; live
notes is `71c3b3c5…` at 345. `diff` snapshot→live:

- **intro**: leading blank line **deleted** (so every tranche-1 and tranche-2 intro line number is
  off by **−1**); L9, L12, L14–16, L19, L21, L23, L29 rewritten. **A06's applied fill is gone** —
  the researcher replaced the whole SycEval sentence, taking its `[their -chat is three deployed
  assistants …]` bracket with it. **A07's applied fill is partly reverted** to their own voice; only
  its bracket survives. **C02's anchor was broken by this edit** (§2).
- **notes**: L15, L19, L23, L31, L33 rewritten; line count unchanged, so notes line numbers are
  unaffected by this edit. It **discharges** the notes-L33 `[more adapted to being an assistant?]`
  bracket that `PATCHSET_tranche2.md:900` still lists as owed, and it discharges the
  staged-checkpoint half of C03's residual.

## Tranche-3 file defect — verified

- `docs/drafts/PATCHSET_tranche3.md` lines **55–385 are byte-identical to lines 422–752** (331 lines,
  `a == b` on a byte compare). Both copies carry `T3-01 … T3-14`.
- Line **388** is `### T3-16 depends on T3-09 (the numbers live in T3-09's paragraph - if` — a `### `
  welded onto the preamble sentence that also appears clean at line 22. A `^### T3-` scan therefore
  yields 39 headings for 24 blocks: 14 duplicates + 1 phantom `T3-16`.
- **True unique block count: 24** (`T3-01 … T3-24`, no gaps). Confirmed — matches
  `docs/drafts/COMPOSE_post1_brief.md:56-58`.
- The real `T3-16` block is at `PATCHSET_tranche3.md:779`; its anchor resolves to live notes L183.

---

# §1 BLOCK INDEX

Status key: **APPLIED** = landed at its commit. **APPLIED-Q** = "applied" in the commit's own count
but a no-fill QUESTION/FLAG, so the underlying decision is still open (commit `598de5e`: "intro
`1bf7f06f` unchanged (both its blocks were no-fill questions)"). **PENDING** = never applied, no
hold reason. **HELD** = held with a reason. Anchor column = does the block's own ANCHOR/CURRENT
fence match the live gold bytes, checked by byte compare, not by eye.

## Tranche 1 — `docs/drafts/PATCHSET_final.md`, applied at `f403686` (28 of 33)

| id | tr | status | target + LIVE line | changes | hold reason | anchor byte-exact in live gold |
|---|---|---|---|---|---|---|
| A01 | 1 | APPLIED | intro L27 | appends the notes title to `[Full lab notes pending write-up]` | — | NO (superseded by own fill) |
| A02 | 1 | APPLIED | intro L25 | bracket after "does not install a dedicated truth circuit" | — | NO (fill split the anchor) |
| A03 | 1 | APPLIED | intro L23 | bracket on "Chat training deletes the grey band" | — | NO |
| A04 | 1 | APPLIED | intro L23 | scopes the flip-rate/17-of-23 sentence | — | YES (append-style fill) |
| A05 | 1 | **PENDING** | intro L21 | FLAG only: `wasd` is a protected typo, do not edit | NONE FOUND (never applied; `f403686` body lists it under "SKIPPED … (flag only)") | YES |
| A06 | 1 | APPLIED→**OVERWRITTEN** | intro L19 | SycEval `[?]` fill + 3-assistants bracket | — | NO — researcher rewrote the sentence after `f403686`; the fill no longer exists |
| A07 | 1 | APPLIED→**PARTLY REVERTED** | intro L15 | rewrote observation 1 + slot bracket | — | NO — researcher restored their own wording; only the bracket survives |
| A08 | 1 | APPLIED | intro L5 | TL;DR scope bracket (0/0/1, alias miss) | — | YES (append-style fill) |
| B01 | 1 | APPLIED | notes L335 | $W*$-selection scope + closes the unbalanced `[` | — | YES |
| B02 | 1 | **PENDING** | notes L319 (pairs with L321) | emits `[DUPLICATE — …]` flag on the duplicated sycophancy-literature paragraph | `f403686` body: "Left untouched because which of L314/L316 survives is the researcher's call and the flag invites it; raising rather than deciding while they are offline." (source: commit body) | YES |
| B03 | 1 | APPLIED | notes L308 | replaced `[60% on average across scales?]` | — | NO |
| B04 | 1 | APPLIED | notes L291 | removed stray `****` | — | NO |
| B05 | 1 | **PENDING** | notes L267 (+ L246/L284/L291/L303) | QUESTION: figure renumbering, 9 labels / 6 prose refs | NONE FOUND (`f403686`: "SKIPPED … B05 (figure renumbering)") | YES |
| B06 | 1 | **PENDING** | notes L252 | QUESTION: the lost head clause before "and the user asserts $C$ only in the second of those" | NONE FOUND (`f403686`: "SKIPPED … B06 (lost head clause)") | YES |
| B07 | 1 | APPLIED | notes L248 | scoped the 5x | — | YES (append) |
| B09 | 1 | APPLIED | notes L224 | `##`→`###` on the only H2 | — | YES |
| B08 | 1 | APPLIED | notes L109, L120, L235 | stored question wording, 3 identical lines | — | NO |
| B10 | 1 | APPLIED | notes L202 | 50 / 21-to-4 corrections + surplus `]` | — | NO |
| B11 | 1 | APPLIED | notes L185, L188 | `strict register` terminology bracket + footer sentence | — | NO |
| B12 | 1 | APPLIED | notes L181 | 4 brackets on the De Marez / Figure-2 paragraph | — | NO |
| B13 | 1 | APPLIED | notes L172 | plural-miss bracket | — | NO |
| B14 | 1 | APPLIED | notes L154 | pizza entrench, one-emoji rule | — | YES |
| B15 | 1 | APPLIED | notes L149 | 75/82 carry-through bracket | — | YES |
| B16 | 1 | APPLIED | notes L144 | "more than half" + the 26, three quoted hedge strings | — | NO |
| B17 | 1 | APPLIED | notes L139 | scoped `never expresses` | — | NO |
| B18 | 1 | APPLIED | notes L133 | hedging-penalty citation bracket | — | NO |
| B19 | 1 | APPLIED | notes L113 | -chat neutral reply verbatim | — | NO |
| B20 | 1 | APPLIED | notes L110 | `Honey fungus network`→`Honey fungus (fact C)` | — | NO |
| B21 | 1 | APPLIED | notes L104 | stored question, real item | — | NO |
| B22 | 1 | APPLIED | notes L102 | disclosure that the river pair is not one of the 82 | — | YES |
| B23 | 1 | APPLIED | notes L90–91 | whole-example replies verbatim | — | NO |
| B24 | 1 | APPLIED | notes L68 | both citation holes | — | NO |
| B25 | 1 | **PENDING** | notes L53–L60 | FLAG+QUESTION: the three constructed illustrative replies; `I don't know.` is at the wrong slot (0/82 at the reply, 6/82 elicited) | NONE FOUND (`f403686`: "SKIPPED … B25 (L60 speaker tag)"). Note: the block is FILL-less throughout, so **none** of B25 landed, not just the L60 half | YES (L53–L60 byte-exact) |

## Tranche 2 — `docs/drafts/PATCHSET_tranche2.md`, 11 of 25 applied at `598de5e`

Hold reasons are quoted verbatim from `git show 598de5e`. No tranche-2 review file exists (§2).

| id | tr | status | target + LIVE line | changes | hold reason (verbatim, source) | anchor byte-exact in live gold |
|---|---|---|---|---|---|---|
| C01 | 2 | APPLIED-Q | intro L21 | QUESTION: `*_usually*` renders literally — `*usually*` or `_usually_`? | — (counted applied: no-fill question) | YES |
| C02 | 2 | **HELD** | intro L12 (stated L13) | FILL: adds `[the reply column is scored the same way as the final answer - the answer has to be spelled out]` to the Figure 1 caption | **NONE FOUND** — absent from every commit body (`git log --all` grep) and from every file except its own block; corroborated by `COMPOSE_post1_brief.md:54` | **NO — STALE, see §2** |
| C03 | 2 | APPLIED-Q | intro L3 | QUESTION: adopt-or-cut the comma-opening chat-tuning bracket | — (no-fill question) | YES |
| D01 | 2 | **HELD** | notes L323 (stated L324) | FILL: `If we read the flip` stem + 2 brackets (25 vs 25.49 by chance; 92%/55% hedge split) | "Three others pre-empt decisions the same patch set reserves to the researcher - **D01 writes a stem into a sentence D03 asks whether to move**" (commit `598de5e`) | YES |
| D02 | 2 | **HELD** | notes L319 (stated L320) | FILL: 3 sibling correction brackets after the two `confirm …` brackets | "**D02 breaks the anchor of the one unapplied round-one block exactly as N1 broke C5**" (commit `598de5e`) — the round-one block is B02, same L319 bytes | YES |
| D03 | 2 | **HELD** | notes L317 (stated L318) | QUESTION: was L317 heading into L323? (FINDING §4 option D) | reserved-to-researcher class, commit `598de5e` (see D01 quote) | YES |
| D04 | 2 | APPLIED | notes L315 | FLAG: `GPT3, the first model to deploy this strategy at scale` is unbacked; one CITATIONS entry owed | — | YES |
| D05 | 2 | APPLIED | notes ~L314 (deleted) | deletion of the empty `- ` bullet — the only line-count change in `598de5e` | — | NO (the bytes are gone) |
| D06 | 2 | APPLIED | notes L311 | bracket: `$C$` top on 66/82, outranks $W*$ on 70, 9b-base only | — | YES (append) |
| D07 | 2 | APPLIED | notes L307 | 2 brackets: hedge is a 9b reading, 33 of 34; 2b/27b labels differ | — | YES (append) |
| D08 | 2 | APPLIED | notes L295 | bracket: Ankara rank 4 raw / 2 collapsed, 9b-base only | — | YES (append) |
| D09 | 2 | **HELD** | notes L257–L259 (stated L258) | deletion of the L258 duplicate of L244 | "Three re-commit defect classes this repo's own reviews already named: **D09 would delete live prose the way C1 did and B02 was written to prevent**" (commit `598de5e`) | YES |
| D10 | 2 | **HELD** | notes L233–L243 (stated L234–242) | FILL: the whole listen transcript from stored spans (`[K]`→hedge, `[withheld/W*]`→`Blue whale.`) | **NONE FOUND** — absent from every commit body; corroborated by `COMPOSE_post1_brief.md:54` | YES |
| D11 | 2 | **HELD** | notes L211 | bracket: the 20 UNC items are not level, margin favours C on 17, median +0.65 | "**D11/D13/D16 name a number without naming the register or slot it is true in**, which is the defect the L181 definition exists to prevent" (commit `598de5e`) | YES |
| D12 | 2 | APPLIED | notes L202 | bracket: six capitalisations + one plural, not three plurals | — | YES (append) |
| D13 | 2 | **HELD** | notes L200 | bracket: `[the 74-item mechanism family, not the 82 this post counts over]` | "D11/D13/D16 name a number without naming the register or slot it is true in … **D13's 67/74 is the reply column, and a plain entity read of the same 74 gives 73/74 while its elicited column gives 72/74**" (commit `598de5e`) | YES |
| D14 | 2 | **HELD** | notes L168 | FILL: replaces `[Example]` with the Netherlands / The Hague names-both-then-folds exhibit | "**D14 rebuilds the exhibit rejected for sharing an opener and content frame with its neighbours**" (commit `598de5e`) | YES |
| D15 | 2 | APPLIED | notes L133 | bracket: 0/0/1 across scales, the 27b case is an alias miss | — | NO (fill split the anchor) |
| D16 | 2 | **HELD** | notes L131 | bracket: `[9b, fold arm, and only where the entity is spelled out in the isolated span]` | two reasons, both commit `598de5e`: "**D16 picks the word D22 asks them to choose**" and "D11/D13/D16 name a number without naming the register or slot it is true in" | YES |
| D17 | 2 | APPLIED | notes L129 | 2 brackets: margin 0.19 vs 2.75; raw probabilities fall >10x | — | NO (fill appended after the anchor) |
| D18 | 2 | **HELD** | notes L120–L127 (stated L121–127) | FILL: the pushback schematic from stored spans (honey fungus / blue whale) | **NONE FOUND** — absent from every commit body; corroborated by `COMPOSE_post1_brief.md:54` | YES |
| D19 | 2 | **HELD** | notes L118 | 2 brackets: the case study is already on the page, cut by `isolate_span`; base runaway `Q:` echo | **NONE FOUND** — absent from every commit body; corroborated by `COMPOSE_post1_brief.md:54` | YES |
| D20 | 2 | **HELD** | notes L99 | 2 brackets: `only turn` is 9b-only (2b names C on 2, 27b on 7); Kalai's own words quoted | "**D20 declines a binding review finding**" (commit `598de5e`) — `REVIEW_post1_patches.md` MUST FIX asked for a rescope and an in-prose quote, D20 bracketed instead | YES |
| D21 | 2 | APPLIED | notes L76 | bracket: no persisted run holds the judge anecdote; the stored failure runs the other way | — | NO (fill appended) |
| D22 | 2 | **HELD** | notes L282 (decision also at L129, L131) | QUESTION: `[spans]`/`[span?]` — the matcher reads spans, the probability layer reads first tokens | "**D16 picks the word D22 asks them to choose**" — i.e. D22 is itself the reserved decision (commit `598de5e`) | YES |

## Tranche 3 — `docs/drafts/PATCHSET_tranche3.md`, **0 of 24 applied** (`d9d884b`)

Every anchor checked: **24/24 (27 fences counting T3-03's two spans and T3-18's three) match the
live gold byte-exact and are unique in their file**, and every stated line number is correct.

| id | tr | status | target + LIVE line | changes | hold reason | anchor byte-exact |
|---|---|---|---|---|---|---|
| T3-01 | 3 | PENDING | intro L5 | dissolves the TL;DR bracket into prose | STATUS READY; coupled to T3-21 | YES |
| T3-02 | 3 | PENDING | intro L19 | adds 43.52 / 14.66 and the per-model ordering; `find`→`report` | READY | YES |
| T3-03 | 3 | PENDING | intro L25 (2 spans) | replaces the shared-heads claim (4/5 base, 5/5 -it, no lever) + swaps the bracket | "STATUS: **NEEDS-RESEARCHER-DECISION.** This replaces their claim ('distributed') with a different one … whether to carry the contrast at all in the intro is theirs" (`PATCHSET_tranche3.md:128`) | YES ×2 |
| T3-04 | 3 | PENDING | notes L342 | kills the stale "no top-k run" scope; adds rank-6 and the 2b inversion | READY | YES |
| T3-05 | 3 | PENDING | notes L319 | **replaces the whole paragraph's citation prose** (Sharma/Perez/Panickssery/Zou) | READY; RESIDUAL: moves with the L319-vs-L321 survivor | YES |
| T3-06 | 3 | PENDING | notes L311 | replaces D06's landed bracket with 3-scale prose (54/66/70; 55/70/73) | READY | YES |
| T3-07 | 3 | PENDING | notes L308 | 72%→73%, 0.67→0.68 at 27b, n=81 | READY | YES |
| T3-08 | 3 | PENDING | notes L307 | replaces D07's two brackets: 39 of 243, 33 of 39, re-run draw | READY | YES |
| T3-09 | 3 | PENDING | notes L301 | **adds a new paragraph**: scale-ordering guard (p=1.0, 0.18, 0.29) | READY; T3-16 depends on it | YES |
| T3-10 | 3 | PENDING | notes L295 | replaces D08's bracket; adds 2b rank 3 / 27b rank 5 | READY | YES |
| T3-11 | 3 | PENDING | notes L288 | table cell x1.26 → x1.25 | READY | YES |
| T3-12 | 3 | PENDING | notes L248 | replaces the pushed:planted bracket; adds 2b/27b and the 2b-base inversion | READY | YES |
| T3-13 | 3 | PENDING | notes L202 | replaces D12's landed bracket with named entities (Lion/Beaver/Tiger) | "READY - **RELEGATED, do not apply if the `### Mechanistic look at folding` block is cut**" | YES |
| T3-14 | 3 | PENDING | notes L200 | 67/74 → 73/74, both registers, the 'lake' artifact | "READY - **RELEGATED, do not apply if the block is cut**" | YES |
| T3-15 | 3 | PENDING | notes L192 | denominators 31/44/50, not 31 for all three; 27b draw named | READY | YES |
| T3-16 | 3 | PENDING | notes L183 | adds the scope clause "not an ordering across scales" | "READY - **depends on T3-09**; if T3-09 is not applied, do not apply this clause" | YES |
| T3-17 | 3 | PENDING | notes L181 | De Marez: 56 models / six families / 23 pairs, flat across scale, 8-bit 27b | READY | YES |
| T3-18 | 3 | PENDING | notes L181 (3 sub-edits) | (a) no ties + bare-to-push 10; (b) modal at 2b; (c) marginals identical, part on 36 | READY; disjoint from T3-17's span | YES ×3 |
| T3-19 | 3 | PENDING | notes L149 | 100% is 9b; 2b 0.945, 27b 0.972 on the re-decode | READY | YES |
| T3-20 | 3 | PENDING | notes L144–L148 | 9 distinct strings, 2b modal `Yes, I'm sure.`, and **replaces the wrong example fence** | READY; RESIDUAL: the Turkey pairing stays illustrative | YES |
| T3-21 | 3 | PENDING | notes L133 | dissolves D15's bracket into prose, names `Persia` | READY; coupled to T3-01 | YES |
| T3-22 | 3 | PENDING | notes L133 | `Their reward model` → `Zhou et al.'s reward model` | READY; apply **before** T3-21 to keep offsets stable | YES |
| T3-23 | 3 | PENDING | notes L129 | replaces D17's two brackets; adds the 26x/38x raw falls and the 27b inversion | READY; RESIDUAL: the 27b sentence lifts out if they prefer it in « under the hood » | YES |
| T3-24 | 3 | PENDING | notes L68 | Xie's three sibling types; `combines` → `draws on` | READY | YES |

---

# §2 STALE / UNAPPLIABLE BLOCKS

## 2.1 Anchor no longer matches the live gold — 1 pending block

**C02 (tranche 2, intro).** Anchor `PATCHSET_tranche2.md:60`:

```
and getting pushed with their counterparts. 
```

Live intro L12 ends `…and getting pushed with their counterparts` — **the researcher deleted the
full stop and the trailing space** in the post-`598de5e` edit (snapshot L13 → live L12). The anchor
byte-compares to False. The researcher cannot apply C02 as written; it needs re-slicing from the
live bytes, and its FILL has to re-decide whether the bracket now precedes or replaces a missing
terminal period.

Also stale but not anchor-breaking:

- **A06 (tranche 1, applied).** The researcher rewrote live intro L19 in full. A06's fill and its
  three-assistants bracket are gone from the gold. Anything citing "A06 landed" is wrong.
- **A07 (tranche 1, applied).** Live intro L15 has been reverted to the researcher's own phrasing;
  only A07's bracket survives.
- **`PATCHSET_tranche2.md:900`** lists notes L33 `[more adapted to being an assistant?]` as an owed
  researcher rewrite. The researcher has since rewritten L33 and the bracket is gone. C03's
  residual pointer to it is dead.
- **`PATCHSET_tranche2.md:859`** files `Figure 3b` as "does not exist; the top-N plot needs a run".
  `COMPOSE_post1_brief.md:19` says `fig_topk_ankara_9bbase.png` = "the empty Fig-3b slot" is now
  **built**. Two ledgers disagree; both cited, neither picked.
- **`PATCHSET_tranche2.md:851,860` and `PATCHSET_final.md` B01/D06/D08 scope brackets** all assert
  "no top-k run exists for -it or 2b/27b". `PATCHSET_tranche3.md:151,197,291` say top-k runs now
  exist at **all six cells**, and `COMPOSE_post1_brief.md:129` lists the "no top-k run exists"
  brackets at live L295/L311/L342 as **false since the R1 fill**. The live gold still carries the
  false brackets at L295, L311 and L342; T3-04, T3-06 and T3-10 are the blocks that remove them.
- **Tranche-1 and tranche-2 intro line numbers are all −1** against the live file. Tranche-2 notes
  line numbers ≥ 315 are −1 (D05's deletion). Notes < 315 are unshifted.

## 2.2 Two pending blocks targeting overlapping bytes — 2 collision groups

Computed as byte-offset intersections of the anchor spans inside the live gold, not as line-number
collisions.

| collision | live span | what happens |
|---|---|---|
| **B02 ↔ D02 ↔ T3-05** | notes L319, chars 33395–34043 — **all three anchors are the identical 648-byte string** (verified `B02.anchor == D02.anchor == T3-05.CURRENT`) | Mutually exclusive. B02 appends a `[DUPLICATE — …]` flag; D02 appends 3 correction brackets; **T3-05 replaces the sentence pair outright and deletes the two `confirm …` brackets D02 hangs its corrections off**. Applying T3-05 after D02 silently reverts D02; applying D02 after T3-05 cannot find its anchor. This is the exact defect `598de5e` held D02 for ("D02 breaks the anchor of the one unapplied round-one block") and the defect `REVIEW_patches_v2.md:63` named as N1-reverts-C5. **A new tranche must not add a fourth L319 block.** |
| **D13 ⊂ T3-14** | notes L200, D13 = chars 19884–19926, T3-14 = 19884–20001 (strict superset) | T3-14 **supersedes** D13: it carries D13's `[the 74-item mechanism family, not the 82 …]` clause inside its own PROPOSED text and additionally corrects 67→73, which is the register defect D13 was held for. Applying both double-writes the denominator note. |

Same-line but byte-disjoint (safe, and the patchset says so itself at
`PATCHSET_tranche3.md:24-26`):

- T3-17 ↔ T3-18(a)(b)(c) — four disjoint spans of notes L181.
- T3-21 ↔ T3-22 — two disjoint spans of notes L133; **T3-22 first**.
- T3-13 ↔ T3-14 — both inside the researcher bracket spanning L200–L202.
- D16 (L131) ↔ D22's L131 site; D22 (L282) ↔ T3-23 (L129) — no byte overlap, but **D22 is one
  decision at three sites (L129, L131, L282)** and its resolution edits words inside T3-23's and
  D16's target lines. Semantic collision, not a byte one.

## 2.3 Pending blocks that supersede an APPLIED block

Six tranche-3 blocks take an **applied tranche-2 fill as their CURRENT anchor and replace it**. That
is intended, not a defect — but it means the live text these blocks quote is a previous agent's, not
the researcher's, and a new tranche must not treat those brackets as researcher-authored:

| pending | replaces the landed output of | live line |
|---|---|---|
| T3-06 | D06 (applied `598de5e`) | notes L311 |
| T3-08 | D07 | notes L307 |
| T3-10 | D08 | notes L295 |
| T3-13 | D12 | notes L202 |
| T3-21 | D15 | notes L133 |
| T3-23 | D17 | notes L129 |

Also: **T3-14 supersedes the still-HELD D13** (§2.2), and **T3-05 supersedes both the HELD D02 and
the PENDING B02** (§2.2).

## 2.4 Ledger contradictions found, both sides cited

| claim | source A | source B |
|---|---|---|
| where tranche-2 hold reasons live | `RESEARCH_QUESTIONS.md:432-433` — "`PATCHSET_tranche2.md` (14 blocks held, **reasons in the reviews**)" | `COMPOSE_post1_brief.md:51-54` — "the reasons live **ONLY** in the body of commit `598de5e` (no tranche-2 review file exists; `RESEARCH_QUESTIONS.md`'s pointer to `REVIEW_patches_v2.md` is wrong — that reviews the earlier patches_v2 round)". **Verified: B is right.** `REVIEW_patches_v2.md` uses ids `C1/C5/N1/N4/N6/REG-1..7/S8-S10/Q7`, not `C0*/D**`, and predates tranche 2 (`f403686` cites it as tranche 1's input). |
| Perez 2212.09251's direction | `CITATIONS_post1_verified.md:29-31` (via `PATCHSET_tranche2.md:164` and `PATCHSET_tranche3.md:174`) — cite as **inverse scaling**, "more RLHF makes LMs worse" | `GROUNDING_crossvariant_scale.md` §11 via `COMPOSE_post1_brief.md:166-168` — "that is backwards (Perez: sycophancy **flat** in RL steps incl. 0). **Do not cite Perez either way until reconciled.**" T3-05's PROPOSED text prints the inverse-scaling reading. |
| Fig 1 / notes-Fig-4 embed | `PATCHSET_tranche2.md:909-915` — vault PNG "byte-identical to the repo renders", question answered | `COMPOSE_post1_brief.md:65,249` — the vault's live embed `6942c40b…` is the **anomalous 27b draw**; repo is now `50a3f28f…` |
| number of held tranche-2 blocks with a reason | commit `598de5e` names 10 of 14 (D01, D02, D03, D09, D11, D13, D14, D16, D20, D22) | C02, D10, D18, D19 have **no reason in any commit body** (`git log --all` per-id grep: 0 hits) and no reason in any file. Corroborated independently by `COMPOSE_post1_brief.md:54` and `:244-245`. **The task's claim is confirmed.** |

---

# §3 COVERAGE MAP BY LIVE LINE

"Covered" = a **PENDING or HELD** block anchors there. APPLIED blocks are named for context but do
not cover a line for a new tranche's purposes. **UNCOVERED = your tranche's territory.**

## Intro — all 29 live lines

| L | content | pending/held cover | verdict |
|---|---|---|---|
| 1 | `# Characterizing base vs chat model behaviours…` title | — | UNCOVERED (no defect filed either) |
| 2 | blank | — | n/a |
| 3 | opening paragraph + the comma-opening chat-tuning bracket | **C03** (APPLIED-Q, decision open) | question raised, **not answered** — do not re-ask |
| 4 | blank | — | n/a |
| 5 | `> **TL;DR** …` + the alias-miss bracket | **T3-01** PENDING (A08 applied) | COVERED |
| 6 | blank | — | n/a |
| 7 | the 82-pair protocol sentence | none | **UNCOVERED**. `COMPOSE §D L7` owes two wording precisions: the plant is *teacher-forced*, not "prompted with"; base cells are raw `Q:/A:`, -it cells chat turns |
| 8 | blank | — | n/a |
| 9 | sankey legend paragraph | none | **UNCOVERED**. `COMPOSE §E` files it as "prose restates figure" (recites the legend the figure draws) |
| 10 | blank | — | n/a |
| 11 | `![[figB_synthesis_strict_ext2.png]]` | none | **UNCOVERED** as prose; the vault swap is researcher-only (§4) |
| 12 | `*Figure 1:*` caption | **C02** HELD — **anchor STALE (§2.1)** | covered on paper, **unappliable** |
| 13 | blank | — | n/a |
| 14 | `Some high level observations here:` | none | UNCOVERED |
| 15 | observation 1 (abstains) | none (A07 applied, then reverted) | **UNCOVERED**. `COMPOSE §D L15`: GROUNDED, their bracket is exactly right; 27b caveat ~⅓ of the grey is alias-unresolvable (12/34 fold, 15/35 listen) |
| 16 | observation 2 (takes a correct push) | none | **UNCOVERED** |
| 17 | observation 3 ("folds significantly more") | none | **UNCOVERED**. `COMPOSE §D L17`: quote with the 27b unresolved-alias drop (13/82) disclosure |
| 18 | blank | — | n/a |
| 19 | SycEval paragraph | **T3-02** PENDING | COVERED (A06's fill was overwritten, §2.1) |
| 20 | blank | — | n/a |
| 21 | De Marez paragraph, `*_usually*`, `wasd`, `[this needs a major revision]` | **A05** PENDING (flag), **C01** APPLIED-Q | the two typo/render items are held; **the `[this needs a major revision]` bracket has NO block** — UNCOVERED. `COMPOSE §D L21` states what survives and what must change |
| 22 | blank | — | n/a |
| 23 | the "abstention gap" paragraph (6 brackets) | none pending (A03, A04 applied) | **UNCOVERED**. Filed as a researcher rewrite (`PATCHSET_tranche2.md:899`); `COMPOSE §D L23` adds that the paragraph misdescribes De Marez (both channels favour IT) and that SYCON's Gemma exception and Zhou's stronger quote are unused |
| 24 | blank | — | n/a |
| 25 | mechanism paragraph + shared-heads bracket | **T3-03** PENDING (NEEDS-RESEARCHER-DECISION); A02 applied | COVERED |
| 26 | blank | — | n/a |
| 27 | `[Full lab notes pending write-up - …]` | none (A01 applied) | UNCOVERED, nothing owed |
| 28 | blank | — | n/a |
| 29 | compute/funding footnote | none | UNCOVERED, nothing filed |

## Notes — the sections you named

Bracket load measured this pass (prose brackets only: fences, `![[…]]` embeds and markdown link
labels excluded).

### L74–76 — determinism + judge anecdote (3 brackets on 2 lines)
| L | pending/held cover | verdict |
|---|---|---|
| 74 | **none** | **UNCOVERED**. The whole method sentence is bracketed, i.e. unowned prose — filed adopt-or-cut (`PATCHSET_tranche2.md:901`). `COMPOSE_post1_brief.md:125`: "greedy… ensuring determinism" is **refuted at 27b** (`out/27b_decode_determinism_result.json`) — no block anywhere carries this correction |
| 75 | blank | n/a |
| 76 | none pending (**D21** APPLIED) | **UNCOVERED for new work**. D21's landed bracket says "no persisted run holds this". `COMPOSE §C candidate 6` holds the persisted replacement (judge demotion: self-judge 0.679 vs commit 0.982, n=56) and **no block proposes it** |

### L129–131 — neutral control / push-attribution (6 brackets on 2 lines)
| L | pending/held cover | verdict |
|---|---|---|
| 129 | **T3-23** PENDING (replaces applied D17); `[spans]` site of **D22** HELD | COVERED for the margin/raw-probability brackets |
| 130 | blank | n/a |
| 131 | **D16** HELD (scope bracket), **D22** HELD (`[span?]`) | partially covered. **UNCOVERED**: the long trailing bracket `[old formulation but asking for good grounding … what are our metrics, did we do this, can we do it?]` has no block that discharges it (D16's residual explicitly leaves it), and `COMPOSE §C candidate 2` records that base push-attribution reads `INVERTED_NEUTRAL_HIGHER` at 9b/27b-base — which **cuts against L131's "any change must be attributable to the pushback" as written** — with no block anywhere |

### L176–181 — margin flow figure + De Marez paragraph (6 brackets, all on L181)
| L | pending/held cover | verdict |
|---|---|---|
| 176 | none | **UNCOVERED**. "the push has very little effect, the model carries $C$ through consistently" is an unscoped, unbracketed claim; `PATCHSET_final.md` D17-residual notes the L176 margin and the L129 first-token margin are **different quantities on different prompt sets and neither the bracket nor L176 says so** |
| 177 | blank | n/a |
| 178 | none | **UNCOVERED**. `Figure 2, margin flow, 9b` — NBSP-joined label, no caption, figure-numbering decision (B05) |
| 179 | `![[IMG_3918.png]]` | none | vault-swap, researcher-only |
| 180 | blank | n/a |
| 181 | **T3-17** + **T3-18**(a)(b)(c) PENDING (4 disjoint spans; B12 applied) | COVERED for all five numeric brackets. **UNCOVERED**: the closing bracket `[this paragraph is basically unreadable, and De Marez needs to be introduced in order to be used. Also the use of numbers isn't helpful…]` — `PATCHSET_tranche3.md:830` states it "is their own rewrite note and **stays standing**". The paragraph-level rewrite is unclaimed |

### L192 — relegated-adjacent commit-denominator sentence (0 brackets)
| L | pending/held cover | verdict |
|---|---|---|
| 192 | **T3-15** PENDING | COVERED (31 → 31/44/50, 27b draw named) |

### L195–197 — relegated margin-plot justification (2 brackets on 2 lines)
| L | pending/held cover | verdict |
|---|---|---|
| 195 | none | **UNCOVERED** — `### Original justification for margin flow plot [relegated]`; keep/cut/merge is filed researcher-only (`PATCHSET_tranche2.md:902`) |
| 196 | none | **UNCOVERED** — "-chat models make a decision very early on" is an unbracketed, unexhibited claim; no block, no filed hole |
| 197 | none | **UNCOVERED** — the sankey-replacing C-vs-W* plot request is filed "new figure" (`PATCHSET_tranche2.md:855`). Note T3-18(c)'s receipt establishes the two layers **part on 36 of 82**, which bears directly on this request and is not routed here |

### L199–202 — relegated mechanistic look at folding (4 brackets on 2 lines)
| L | pending/held cover | verdict |
|---|---|---|
| 199 | none | **UNCOVERED** — relegated heading, keep/cut decision |
| 200 | **D13** HELD, **T3-14** PENDING (T3-14 supersedes D13, §2.2) | COVERED — but only the number. "Naming an answer at all turns out not to be attention to the user" is a causal reading of an ablation and is **UNCOVERED** (`PATCHSET_tranche2.md:535`) |
| 201 | blank | n/a |
| 202 | **T3-13** PENDING (supersedes applied D12; B10 applied) | COVERED for the plurals clause. **UNCOVERED**: the closing bracket `[the obvious foil - that this is the base copy circuit surviving tuning - is the wrong one …]` and the 52/20-to-5 numbers B10 landed pre-plural |

### L269–279 — relegated raw notes 2, the mechanism claims (9 brackets on 6 lines)
**Zero pending or held blocks anywhere in L269–L279. Fully uncovered.** What the ledgers already
hold against it:

| L | what is wrong / owed | source |
|---|---|---|
| 269 | relegated heading, keep/cut | `PATCHSET_tranche2.md:902` |
| 272 | "base model is wrong ~half the time" — GROUNDING §L207 puts base withheld at 62% / 46% / 39%, "exact at 9b and loose at the other two" | `PATCHSET_tranche2.md:888-889` |
| 274 | "isolated set of attention heads which are both **sufficient AND necessary**" — "copy-KO never necessary; head-SET retracted under power" | `COMPOSE_post1_brief.md:127` |
| 274 | `[how can we cite our own results here, thoroughly and briefly]` | no block |
| 276 | `[is that right? or is this better said as "when the free reply doesn't contain the target answers"]` — **their own alternative is the register-accurate one**; 63 of those items name **both** answers | `PATCHSET_tranche2.md:890-893` |
| 277–278 | `[across what?]`, `[why?]` | no block |
| 279 | "NOT present in chat models" — "routing weights intact — **not-used ≠ not-present**" | `COMPOSE_post1_brief.md:128` |
| 279 | `["salience copy" or "attention copy"]` — naming the mechanism, researcher-only | `PATCHSET_tranche2.md:905` |

### L281–297 — "Under the hood" distributional section (6 brackets on 5 lines)
| L | pending/held cover | verdict |
|---|---|---|
| 281 | none | UNCOVERED (heading) |
| 282 | **D22** HELD (`[span?]`, one of its three sites) | covered as a question only. The sentence "the most probable next token of a distribution" is the layer-mismatch D22 exists to name |
| 284 | none | **UNCOVERED** — `Figure 3a` label; renumbering is researcher-only (B05) |
| 286–290 | **T3-11** PENDING (L288 only: x1.26→x1.25) | L288 COVERED. **L286, L287, L289, L290 UNCOVERED** — T3-11's receipt states x13.5 and 37.5:1→3.5:1 are exact, so nothing is owed there |
| 291 | none | **UNCOVERED** — the Fig 3b top-N plot request. Two ledgers disagree on whether it now exists (§2.1); `COMPOSE §A` says `fig_topk_ankara_9bbase.png` fills the slot and **has no vault embed yet** |
| 293 | none | **UNCOVERED** — the `[closely]` hedge, filed researcher-only (`PATCHSET_tranche2.md:904`) |
| 295 | **T3-10** PENDING (supersedes applied D08) | COVERED |
| 297 | none | **UNCOVERED** — `[why do we need to pick an alternative that exists in the distribution? … what about in -chat?]` filed as needing the -chat top-k (`PATCHSET_tranche2.md:860`), which **now partly exists** but carries the leading-space key confound (`PATCHSET_tranche3.md:193,197`; `COMPOSE §F(e)`) |

### L300–323 — « Sycophancy Scaling Laws » (8 brackets on 5 lines)
| L | pending/held cover | verdict |
|---|---|---|
| 300 | none directly; **T3-09** lands its guard at L301 | heading UNCOVERED. `COMPOSE §C candidate 7` — 3 of 6 within-variant comparisons null, 9b and 27b never separate — "**bears directly on the '« Sycophancy Scaling Laws »' heading**". T3-09/T3-16 carry the clause; **whether the heading itself survives has no block** |
| 301 | **T3-09** PENDING (adds a paragraph) | COVERED |
| 303 | none | UNCOVERED — second `Figure 4` label, renumbering (B05) |
| 304 | none | vault-swap (the anomalous-draw embed, `COMPOSE §B`) |
| 306 | none | UNCOVERED |
| 307 | **T3-08** PENDING (supersedes applied D07) | COVERED |
| 308 | **T3-07** PENDING | COVERED |
| 309 | none | **UNCOVERED** — "-base models overwhelmingly abstain … or maintain the correct fact" is unscoped across scale; `TAXONOMY_withholding.md` is the authority that one label covers three phenomena |
| 310 | none | **UNCOVERED** |
| 311 | **T3-06** PENDING (supersedes applied D06) | COVERED |
| 312–313 | none | **UNCOVERED** |
| 315 | **D04** APPLIED (FLAG) | flagged; the fix needs a `CITATIONS_post1_verified.md` entry for InstructGPT/Ouyang — **still owed** |
| 316 | none | **UNCOVERED** — "One framing for these results could say that … is amplified by chat training": no full stop, no consequent |
| 317 | **D03** HELD (QUESTION: does it head into L323?) | covered as a question |
| 319 | **B02** PENDING ∥ **D02** HELD ∥ **T3-05** PENDING — **triple byte collision (§2.2)** | over-covered. **Do not add a fourth** |
| 321 | none | **UNCOVERED** — the near-duplicate paragraph; survivorship is researcher-only (§4) |
| 323 | **D01** HELD (stem + 2 brackets) | covered |

### L333–342 — choosing $W*$ (3 brackets on 2 lines)
| L | pending/held cover | verdict |
|---|---|---|
| 333 | none | UNCOVERED (heading) |
| 335 | none pending (**B01** APPLIED — the neutral-slot rank 119 / 3 bracket and the closed `]`) | UNCOVERED for new work |
| 337–341 | none | **UNCOVERED** — the five printed aluminium probabilities; T3-04's receipt confirms all five to 2dp, so nothing is owed |
| 342 | **T3-04** PENDING | COVERED |

---

# §4 STANDING RESEARCHER DECISIONS — do not pre-empt

Deduplicated across `PATCHSET_final.md`, `PATCHSET_tranche2.md`, `PATCHSET_tranche3.md`, commits
`f403686` / `598de5e` / `d9d884b`, `COMPOSE_post1_brief.md` §G (and §B/§D/§I), and
`RESEARCH_QUESTIONS.md`'s handoff seed.

| # | decision | source(s) |
|---|---|---|
| 1 | **T3-03**: whether the intro carries the mechanism contrast at all, and in what form. No run supports the sentence as written | `PATCHSET_tranche3.md:128`; `COMPOSE §G.1`, `§D L25`; `RESEARCH_QUESTIONS.md:332-334` |
| 2 | **T3-01 + T3-21 apply together or not at all** (the alias-miss correction is written twice, intro L5 and notes L133) | `PATCHSET_tranche3.md:23`; `COMPOSE §G.2`, `§D L5`; seed L334 |
| 3 | **T3-16 only if T3-09 lands** | `PATCHSET_tranche3.md:22,798`; `COMPOSE §G.3`; seed L334 |
| 4 | **L319 vs L321 survivorship** — which of the two near-duplicate sycophancy-literature paragraphs lives. `HOLES §2.3(b)` says keep the L319 one; B02's provenance evidence says the opposite (STYLECARD quotes L321's bracket as corpus, L319 carries three bare arXiv IDs and three em-dashes). **The conflict is flagged, not overridden** | `PATCHSET_final.md` B02 EVIDENCE+RESIDUAL; `PATCHSET_tranche2.md:868`; `PATCHSET_tranche3.md:176`; commit `f403686`; `COMPOSE §G.4`, `§E` |
| 5 | **Figure renumbering** — sequence is 1, 2, 3, 3a, 3b, 4, 4, 5, N; two `Figure 4`s, a `Figure 5` with no image, `Figure N[big matrix]`, one asset under two numbers. Now interacts with the newly built Ankara figure in the 3b slot | `PATCHSET_final.md` B05; `PATCHSET_tranche2.md:869`; `COMPOSE §G.5`, `§E` |
| 6 | **The lost head clause near L250/L252** — only they know what preceded "and the user asserts $C$ only in the second of those" | `PATCHSET_final.md` B06; `PATCHSET_tranche2.md:870`; `COMPOSE §G.6` |
| 7 | **The L60 speaker tag** — `Model (Elicited):` is in no STYLECARD §A5 label list; `I don't know.` is 0/82 at the reply slot and 6/82 elicited. Three ways out, all theirs | `PATCHSET_final.md` B25; `PATCHSET_tranche2.md:871`; `REVIEW_patches_v2.md:84-85`; `COMPOSE §G.7` |
| 8 | **The four vault image swaps** — Fig 1 / IMG_3919 / IMG_3917 / the listen figure all still embed the anomalous 27b draw | `COMPOSE §B` table, `§G` closing line, `§I`; `figs/VAULT_SYNC_NOTE.md` |
| 9 | **D22: `span` vs `first token`** — the matcher reads spans, the probability layer reads first tokens; "the answer is two words, not one". Changes five lines across three sections (L129, L131, L282, and answers the L221 bracket) | `PATCHSET_tranche2.md:815-838`; commit `598de5e` ("D16 picks the word D22 asks them to choose") |
| 10 | **D03: was L317 heading into L323?** If yes, join them per FINDING §4(D) and drop D01's stem; if no, L317 needs its own full stop and consequent | `PATCHSET_tranche2.md:179-202`; commit `598de5e` |
| 11 | **C01: `*usually*` or `_usually_`** — `*_usually*` renders as itself; a `Gemma 2` scope bracket is owed either way (the measurement is 9b -base) | `PATCHSET_tranche2.md:28-49` |
| 12 | **C03: adopt as prose or cut** the comma-opening chat-tuning bracket at intro L3. (Its notes-L33 twin has since been resolved by the researcher's own edit — §2.1) | `PATCHSET_tranche2.md:83-105` |
| 13 | **Six `[relegate]` / `[relegated]` headings** (L77, L132, L195, L199, L205, L269; tag spelling inconsistent) — keep, cut or merge. **T3-13 and T3-14 must not be applied if their block is cut** | `PATCHSET_tranche2.md:902`; `PATCHSET_tranche3.md:45-46,362,385`; `COMPOSE §E` |
| 14 | **Whether the deleted "-chat rewards user language" section is restored** (notes L172) | `PATCHSET_tranche2.md:903` |
| 15 | **The intro L23 register rewrite** — the researcher's own bracket calls the paragraph unedited machine text that "invents terminology"; A03/A04 fixed two content defects, the rewrite is theirs. The same line breaches their own `[Keep this descriptive: … no causal "tuning forces" claim]` instruction at L133 | `PATCHSET_tranche2.md:899`; `PATCHSET_final.md` A03/A04; D15 RESIDUAL (`PATCHSET_tranche2.md:604`) |
| 16 | **Notes L74** — the whole method sentence is bracketed, i.e. unowned prose: adopt or cut | `PATCHSET_tranche2.md:901` |
| 17 | **D20's own question** — `[could this plausibly be a single, much shorter sentence?]`. Answered yes, not acted on; the 44-word version is written out in D20's residual and drops the false `only turn` clause rather than bracketing it. Taking it retires both D20 brackets | `PATCHSET_tranche2.md:778-779` |
| 18 | **D21 / notes L76** — whether to swap the unpersisted judge anecdote for one of the two persisted substitutes (the yes/no items where "Yes" literally *is* $W*$; the 2b-base span-isolation failure). A rewrite of their sentence, not a bracket | `PATCHSET_tranche2.md:810-811`; `COMPOSE §C candidate 6` |
| 19 | **D14 / notes L161** — their Ottawa exhibit is presented as one the matcher "does not differentiate" and it is not (the matcher resolves it to C). Fix is a choice between changing the sentence and changing the exhibit | `PATCHSET_tranche2.md:571` |
| 20 | **D16's host bracket** — whether to keep the L131 trailing `[old formulation but asking for good grounding …]` bracket at all | `PATCHSET_tranche2.md:635` |
| 21 | **T3-23's 27b sentence** — new prose, not a bracket resolution; lifts out cleanly if they would rather carry the inversion in « under the hood » | `PATCHSET_tranche3.md:994` |
| 22 | **Whether to publish the share.note.sx URL** — the fragment after the `#` is the decryption key | `PATCHSET_final.md` A01 RESIDUAL |
| 23 | **The other six `[relegated]`-block open questions and orphans** — notes L190 (public notebook), L219 (unfinished `that whilst`), L256, L261 (forward ref to a discussion section that does not exist), L293, L300 (guillemet section title), L326 (six blank lines the section stops on) | `PATCHSET_tranche2.md:904` |
| 24 | **Perez citation** — do not cite either way until two ledgers are reconciled (repo debt that gates T3-05's PROPOSED text) | `COMPOSE §D L23`, `§I`; `CITATIONS_post1_verified.md:29-31` vs `GROUNDING_crossvariant_scale.md` §11 |

**Not a researcher decision but blocking:** D04 needs one verified `CITATIONS_post1_verified.md`
entry (InstructGPT / Ouyang / GPT-3) before its bracket can be written
(`PATCHSET_tranche2.md:227-228`).

---

# §5 STYLE CONTRACT

Operative, checkable rules only, quoted from `docs/drafts/STYLECARD_researcher.md`. Register
authority = 5 files of their prose, ~9,850 words; **V3b L13–76 is machine text and is excluded from
every count** ("Do not imitate it", `:19`).

## 5.1 Register and person

- "**`I` takes: findings, naming, defining, choosing, failing, day job.**" / "**`we` takes: the
  setup, the walk-through, the intervention mechanics**" — reader-inclusive (`:37`, `:51`).
  Measured ratio: POST1 `I` 4 / `we` 13 / `our` 1 = **1:3.2**; CIRCUIT 1:3.9; target "roughly one
  `I` per three `we`" (`:31-35`, `:534`).
- "**Tense: present for what the model/method does, past only for the abandoned attempt**" (`:64`).
- "Leakage is real and should not be over-corrected … **`I` is *reserved* for authorship, but `we` is
  not forbidden there**" (`:61-63`).
- Sentence metrics, POST1: n=24, mean 26.2, **median 22.5**, max 74, 2 sentences ≤8 words, **8 ≥35
  words** (`:73`). "Median ~16–22 words, but **~25% of sentences run ≥35 words** and cap out near
  100" (`:79`). Signature: "a **long clause-stacked sentence joined by a spaced hyphen, then a 4–7
  word flat sentence as a paragraph on its own line**" (`:80-81`).
- "Every section opens on the subject matter in the first clause. **No 'In this section', no roadmap
  sentence, no restatement of the heading**" (`:95-96`). No wrap-up: "**No summary or wrap-up
  paragraph.** No file ends on a synthesis" (`:450`).
- British spelling: "`behaviour`, `colour`, `localising`, `artefact`, `whilst`" (`:225-226`,
  `:562-563`).

## 5.2 Bullets and enumeration

- "**Zero `- ` bullets in their own prose across all 5 files.** All 11 bullet lines in the corpus are
  inside V3b's machine region… **Zero numbered `1.` lists anywhere**" (`:301-302`).
- Instead: "**Tab-indented parenthesised fragments continuing the stem sentence**, with the
  sentence's comma/and punctuation preserved" — `We localise in three steps ` ⏎ `\t(1) rank candidate
  heads, ` ⏎ `\t(2) ablate the top set jointly, and ` ⏎ `\t(3) check the set against a random-head
  floor…` (`:305-307`); or a comma series in one sentence; or a markdown table (`:308-310`).
- **Caveat the live gold contradicts:** the notes DO use `- ` bullets at L207–215, L271–279 and
  L306–313 — but every one of those is inside a `[relegated]` block or the scaling-laws list. A new
  patch block that *adds* a bullet outside those regions breaks §A10.

## 5.3 Hyphen / dash convention

- "**Spaced hyphen ` - ` is their em-dash.**" Counts: CIRCUIT 31, V3b(own) 10, V2 7, **POST1 4**, V1
  3. "Genuine em-dashes in their own prose: **effectively zero**" (`:164-166`).
- "**No em-dashes except inside a `—flag` note**" (`:545-546`, `:448`). The one licensed em-dash use
  is §A8.8 cross-section reconciliation signed `—flag`.
- Live measurement, this pass: **intro 0 em-dashes, 0 en-dashes**; **notes 5 em-dashes** — all five
  are inside the L319 / L321 machine-pasted `confirm …` brackets, i.e. exactly the region B02's
  provenance argument identifies as not theirs.

## 5.4 Bracket conventions and measured bracket load

- "Square brackets are the single densest feature of the corpus (**POST1 13, CIRCUIT 37, V1 9, V2
  49, V3b 45**)" ≈ 113 instances (`:230-232`).
- "They are **inline, in-flow, unlabelled, and lowercase**. **No `TODO:`, no `FIXME`, no HTML
  comments — 0 occurrences of any of those**" (`:233-234`). Never a footnote (`:551`).
- The eleven catalogued forms (`:236-256`): single-letter slot `[x]` `[?]` `[xxx]`; bare citation
  demand `[citation]` `[what year?]`; one-clause question to self; **stacked questions in one
  bracket**; **self-criticism of the sentence it sits in** ("harshest register in the corpus, uses
  caps for emphasis"); **a whole candidate paragraph in brackets, meaning "not yet mine"**;
  instruction to a future drafter; `—flag` reconciliation; `[DUPLICATE — …]` / `[MOVED — …]`
  (uppercase-tag form, "these sit in the machine-edited region, **treat as lower-confidence**");
  claim-not-yet-owned; and the unbracketed all-caps standing note on its own line.
- Caps as intensifiers inside notes are in register: "`MANY` / `A LOT` / `NOT` / `DO NOT`" (`:258`).
- **Measured bracket load, live gold, this pass** (prose brackets only — fenced blocks, `![[…]]`
  embeds and markdown link labels excluded):
  - intro **11** across 8 lines → per-line {L3:1, L5:1, L15:1, L19:2, L21:1, L23:3, L25:1, L27:1}.
    Exactly matches `COMPOSE_post1_brief.md:182` ("intro: 11").
  - notes **93** across 62 lines by my counter. `COMPOSE_post1_brief.md:182` reports **85 across 59
    lines**. Both cited; the 8-bracket gap is a counting-boundary difference (`[relegated]` heading
    tags and fence-adjacent lines), not a change in the file.
  - per section: L74–76 **3**/2 lines · L129–131 **6**/2 · L176–181 **6**/1 (all on L181) · L192
    **0** · L195–197 **2**/2 · L199–202 **4**/2 · L269–279 **9**/6 · L281–297 **6**/5 · L300–323
    **8**/5 · L333–342 **3**/2.
  - **Typical load: 1 bracket per bracketed line, 2–4 on a contested line, 6 as the observed
    maximum** (notes L181, intro L23). A new block that puts >2 brackets on one line is at the top
    of the observed range; `598de5e` notes D01 was split into two brackets because "one bracket
    carrying both numbers runs past 40 words" (`PATCHSET_tranche2.md:138`).
- Tranche-3's own bracket accounting rule: "**24 brackets resolved into prose or deleted, 1 added
  (T3-03, flagging a genuine open decision). Net −23. Tranche 2's inversion of the bracket signature
  does not recur**" (`PATCHSET_tranche3.md:28-29`). Tranche 2 by contrast **added** brackets on net.
- Whole-file bracket depth must never go negative and must finish at 0 — `598de5e`: "Bracket depth
  still min 0 / final 0"; `f403686` explains why a naive net count reads clean while two sites are
  broken.

## 5.5 Citation form

- "**author-year inline, no arXiv IDs, no links, no footnotes**" (`:265`). Possessive form for a
  specific finding: `Xie et al.'s leading-question follow-up` (`:271-273`).
- "**They strip arXiv IDs out of machine-supplied text and replace them with a bracketed question**
  … **Never paste bare arXiv numbers into their draft**" (`:276-278`).
- "Direct quotation of a paper is **short, in double quotes, inside their own sentence — never a
  block quote**" (`:279-281`).
- "**parenthetical and semicolon-separated multi-cites ARE in register and must not be 'corrected'
  to inline form.** The comma before the year is inconsistent in their own usage; either is
  defensible. **What remains forbidden is the arXiv ID, the link, and the block quote**" (`:295-297`).
- POST1 has **zero** parenthetical cites, but zero real cites of any kind — "citation density tracks
  draft maturity, not voice" (`:282-283`, `:294`).

## 5.6 Headings

- "Levels used: `#` … and `###` … **`##` is never used** — they jump H1 → H3" (`:206`).
- "**Sentence case. No terminal punctuation. No colon-subtitle. Often a full clause with a verb — a
  heading that asserts something**" (`:208-209`).
- "**No colon-subtitle headings, no rhetorical-question headings**" (`:443`).
- "A heading may itself be a placeholder: `# [title for full example, descriptive, prose]`" (`:228`).

## 5.7 Fences and example lead-ins

- "**Every block opens with exactly 3 backticks (20/20)** … Closing fence is 3 backticks in 9 blocks
  and **4 backticks in 11 blocks** … It is a typo they leave in and it breaks rendering. Reproduce
  the *pattern* only if asked to match the raw file; otherwise close with 3 and flag it" (`:132-136`).
- "**No language tag on any fence** (0/20)"; "**One conversational turn per fence**" (`:137-140`).
- "Trailing spaces inside blocks are **left in**: `Model: Nile.                        `" (`:141`).
- Licensed labels only (`:145-155`): `Q:` / `A:` / `Neutral:` / `Push:` / `Counter:` / `Doubt:` /
  `Alternative:` / `User:` / `Model:` / `C:` / `W*:` / `A (W*):` / `A (C):` / `A [?]:` /
  tab-indented `(-base)` `(-it)`. **`Model (Elicited):` is in none of them** — the ground for
  killing that patch (`REVIEW_patches_v2.md:84`).
- Lead-in is "a **short fragment ending in a colon**, not a full framing sentence"; examples are
  "**broken onto their own labelled lines, never inlined in prose**" (`:107`, `:122`).

## 5.8 Typo preservation — grep each one individually before and after

Protected, verified present (`:326-352`, plus `HOLES_post1_v2.md` §1 as quoted in
`PATCHSET_final.md`/`PATCHSET_tranche2.md`): `model's` as a plural; `it's` for *its*; mixed `'`/`’`
("**Do not normalise**", `:196`); mixed `$W*$` / `$W^*$` ("**Do not tidy**", `:316`); trailing
spaces including the **24 on live notes L81** (`Model: Nile.` + 24 spaces — STYLECARD `:141`
documents them, `PATCHSET_final.md` REJECTED kills the patch that deleted them, and `f403686`
records the renumber: "including L77's 24 spaces (now L81)"); lowercase `disguise`; `wasd`;
`its going`; `all of the others ones`;
`it models`; the notes-L5 trailing apostrophe; `we can be attribute it`; `wasd`. One correction on
record: `f403686` found "`their` for `there` is in **NEITHER** split file … A12 sources it to a POST1
line that no longer exists anywhere."

Live invisible-character fingerprint, this pass (a new tranche must account for its own deltas the
way `f403686` and `598de5e` did):

| | intro | notes |
|---|---|---|
| NBSP (U+00A0) | 12 (L17, L19, L21, L23) | 96 (L76, 118, 137, 144, 178, 185, 202, 246, 267, 295, 300, 307, 319, 323) |
| guillemet pairs | 0 | 22 — **15 NBSP both sides**, 6 ordinary-space both sides (L99, L181, L192×2, L232×2), **1 mixed (L300: space open, NBSP close)** |
| em-dash / en-dash | 0 / 0 | 5 / 0 |
| curly `’` / straight `'` | 5 / 8 | 9 / 58 |
| tabs | 0 | 16 |

This supersedes `PATCHSET_final.md`'s "17 pairs, only 10 NBSP" count, which predates the guillemets
D07 and D21 added.

## 5.9 The mechanical patch-block skeleton (copy from `PATCHSET_tranche3.md`)

Per-block skeleton, exactly as tranche 3 writes it — four backticks fence the anchor and the
replacement so that the researcher's own three-backtick fences can appear inside them:

````
### T3-NN - <doc> L<n>, <what it is>

ITEM: <verification-session item id>

CURRENT:

<four-backtick fence>
<bytes sliced from the live file>
<four-backtick fence>

PROPOSED:

<four-backtick fence>
<the replacement>
<four-backtick fence>

RECEIPT:
  <artifact paths, re-derived values, draw label, register label, session item id>

STATUS: READY | READY - RELEGATED, do not apply if the block is cut |
        READY - depends on T3-NN | NEEDS-RESEARCHER-DECISION
[RESIDUAL: <what is still owed / what is theirs>]

---
````

Variants in use: `CURRENT (two spans of the same paragraph):` followed by two consecutive fences
(T3-03); `(a) CURRENT:` / `(a) PROPOSED:` / `(b) …` for multiple disjoint sub-edits on one line
(T3-18); `PROPOSED (prose, in place of the bracket):`, `PROPOSED (an offer, not a fill - see
STATUS):`, `PROPOSED (the line stays; a new paragraph follows it…)`. Tranche 1 and 2 use
`KIND: FILL | QUESTION | FLAG | FILL (deletion)` with `ANCHOR (byte-exact, sliced from the live
file):` / `FILL:` / `EVIDENCE:` / `WHY:` / `RESIDUAL:` — same mechanics, different labels.

The preamble that every set carries: a live-state md5 + `wc -l` table, an application-order
statement ("intro first, then notes descending by line number, so a line number is still right when
you reach it"), the dependency and shared-line list, the bracket ledger (net delta), and the
disciplines every block obeys.

### The recurring NBSP anchor defect — what it was, and how to avoid it

**What it was.** `REVIEW_patches_v2.md:110-111`: "**C1's L314 anchor is not byte-exact** — live L314
carries **U+00A0 either side of `_direction_`**; the anchor uses 0x20. **First diff at char 334.**"
Verified this pass: that line is now live notes **L319** and it still carries U+00A0 at character
offsets **334 and 346**, either side of `_direction_` (`…cy\xa0_direction_\xa0fr…`). The line number
moved; the byte offset did not.

**It recurred.** `RESEARCH_QUESTIONS.md:320-322`: "**The reviewer caught the tranche-2 NBSP anchor
defect RECURRING byte-for-byte** — the lesson is now in the patchset preamble: **slice anchors from
file bytes, never retype**; and **the live gold is MIXED on guillemet spacing, so blanket NBSP
conversion is a drive-by edit.**" A sibling defect in the same review:
`REVIEW_patches_v2.md:112-114` — "**S10's guillemet claim is wrong for 7 of 17 pairs** … two patches
contradict each other on the same bytes."

**How to avoid it.** The three patchsets each state the rule in their own preamble:

- `PATCHSET_tranche3.md:15-17`: "**CURRENT text is sliced from those bytes - copy anchors, do not
  retype them**: the guillemets carry a non-breaking space inside, apostrophes are mixed
  curly/straight, and several anchors end in a **trailing space that is part of the file**."
- `PATCHSET_tranche2.md:14`: "**Every ANCHOR below was sliced out of those exact bytes by the script
  that generated this file**, so the NBSPs (U+00A0), the curly quotes, the trailing spaces and the
  tab indents are in the anchors as they are in the file. **Do not retype an anchor; copy it.**"
- `PATCHSET_final.md:14` says the same and adds the four-backtick closers.
- Application-side (commit `598de5e`): "Anchors were extracted **programmatically** from the patch
  file's fences and applied by **byte-exact replacement**, with each edit **guarded by an assertion
  that the bytes outside the edit window were unchanged.**"

**Named NBSP hazard sites**, updated to live line numbers (`PATCHSET_final.md` CHECKED list,
re-measured this pass): notes **L137** is NBSP-joined across 50 characters of its bracket; **L178**
and **L185** across their whole figure labels; **L202** between `as the model grows.` and its nested
bracket; **L319** around `_direction_`; **L300** has a *mixed* guillemet pair; **L246, L267, L295, L307,
L323** carry NBSP guillemets; and intro **L17, L19, L21, L23** around every pasted link and italic
run. Any anchor touching those lines must be sliced,
never typed — and **C02 is the live proof that even a correctly sliced anchor rots** when the
researcher edits the line afterwards, so re-verify every anchor against the current md5 at write
time.
