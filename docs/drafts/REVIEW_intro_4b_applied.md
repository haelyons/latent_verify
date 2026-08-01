# REVIEW — `DRAFT_post1_intro_tranche4b_applied.md`

Independent adversarial read, 2026-08-01. Read-only pass; nothing outside this file was written, and
nothing under `/home/hal/Documents/` was touched (gold md5 re-checked after the pass:
`83a55a14a8079403fa6be41c309c7f3b`, unchanged).

**Locations** are given as `draft L<n>` (line in `DRAFT_post1_intro_tranche4b_applied.md`) with the
marker-stripped intro line in brackets, e.g. `draft L38 [applied L23]`.

---

## What is not wrong — stated first so the findings below are not read as a wholesale rejection

The mechanical application is **correct and I verified it independently**. I re-sliced all twelve
anchors out of `PATCHSET_tranche4b_intro_register.md`, `PATCHSET_tranche3.md` and
`PATCHSET_tranche4_intro.md` (never retyped), applied them to the gold in the patchset's stated
descending order, and the result is **byte-identical** to the draft's marker-stripped intro. Every
anchor was unique (`count() == 1`). The DELTAS table reproduces exactly: 1132 → 1253 words, 20 → 15
`[`, 12 → 12 NBSP, 29 → 27 split lines, 7054 → 7594 bytes, 0/0 em/en-dash. `REDISTRIBUTE`, `0.875`,
`0.751`, "distributed", "teacher-forc", "monotonic" and Perez are all absent from the applied prose
(R-12, PATCHMAP §4 item 24 respected). `wasd`, `its going` and `*_usually*` all survive. The
`>>`-stripping contract in the header is true.

The apparatus-swamp defect from `b609418` is genuinely fixed: 137 file lines for a 31-line intro,
down from 399.

Everything below is about the *text*, not the machinery.

---

## BLOCKER

### BL-1 — draft L17 [applied L5], the TL;DR's second sentence

**What is wrong.** "Under the push the two variants' distributions move much the same way: the
pushed wrong answer gains probability at -base too, it just doesn't get said."

The headline clause is a **cross-variant equivalence claim**, and `T4b-I07`'s receipt cites only
`-base` numbers for it (`INVENTORY_distributional.md:442-445`, 57/82 at 9b-base and 50/82 at
27b-base). It never cites an `-it` number. Worse, the two `-it` measurements that do exist point the
other way:

- `INVENTORY §2.5`, the format-matched `RC_effect` table: the base→`-it` residual on the canonical
  key is **+4.58 / +2.93 / +2.04 nats at 2b / 9b / 27b** — the `-it` margin moves *further* under the
  push at every scale, not "much the same way". §2.5 also carries a caveat this sentence would need:
  "`canonical` is a **different key at the two arms**".
- `INVENTORY §3.1b`: the margin crosses C→W\* on **48 of 82 at 9b-it against 15 of 82 at 9b-base**
  (and 48 vs 41 at 2b, 36 vs 14 at 27b).
- `INVENTORY §3.1` caveat (i) names `Mc_*` among the fields whose `-it` rows are "readable *within* a
  cell across slots but **not** comparable to the base rows".
- `GAPS_C_claims.md:134` files the neutral→pushed movement comparison as **PARTIAL — 9b only, fold
  only**.

Separately, "gains probability" is the loose phrasing the rest of this pass exists to remove. The
same quantity is scoped carefully 14 lines later ("over the answer strings, not the first token",
`T4b-I04a(ii)`) and left unscoped here, in the line most readers will quote. `INVENTORY §3.2`
explicitly marks ❌ *"The correct answer remains the highest-probability token"* as false.

**Why it matters.** It is the TL;DR. A cross-variant claim with no cross-variant receipt, on a
readout whose own inventory says the arms are not comparable, in the one line that gets screenshotted.

**Minimal fix.** Cut the sentence (−26 words). If the point must survive, it can only be the base
half: "At -base the pushed wrong answer gains ground in the margin, it just doesn't get said" — and
even that needs the readout named.

### BL-2 — draft L38 [applied L23], "with no 27b run in the base arm"

**What is wrong.** The clause is a scope qualifier and it points at the wrong arm. There is **no 27b
run of `cave_fold_vs_listen` in either arm**:

- `SNAPSHOT_circuit_groundtruth.md:317` — `find . -name "cave_fold_vs_listen*"` → **2 JSONs only (9b,
  2b)** plus the instrument.
- `GROUNDING_crossvariant_scale.md §12` — "**No 27b run of `cave_fold_vs_listen`** in either arm".
- `GAPS_C_claims.md:541` — "2b and 9b (891-pool); **no 27b**".
- I confirmed it directly: `results_fold_vs_listen/out/cave_fold_vs_listen.json` carries
  `models.base` = `google/gemma-2-9b` and `models.it` = `google/gemma-2-9b-it` only.

As written, naming the gap *in the base arm* implies the `-chat` arm has 27b coverage. The `-chat`
half of the same sentence ("whilst at -chat fold and listen share all five") then carries **no scale
scope at all**, so the reader takes it as a three-scale result. It is 2b and 9b.

**Why it matters.** This is the intro's only mechanism claim, and its scope qualifier is
affirmatively misleading. It is exactly the defect class the RETRACTIONS register was built for.

**Minimal fix.** "- a correlational read, at 2b and 9b only -" (same length; drops the false
implicature and scopes both halves at once).

### BL-3 — draft L32 [applied L19], "only the 9b -chat "fold" arm has both"

**What is wrong.** Two things, in the same sentence.

1. **Readout contradiction inside one paragraph.** Sentence 2 of this paragraph defines the margin as
   "over the answer strings, **not the first token**". The exception `T4b-I04b` names is a
   **first-token** instrument. I read the artifact:
   `out/foldlisten_demarez_subst_dmz_9bit_a_summary.json` carries `distributions.counter_first` and
   `.elicit_first`, whose only margin fields are `margin_first_space` / `margin_first_bare`, and the
   file's own `margin_framing` reads: *"every margin is a FIRST-TOKEN, Rule-S-class reading. No
   number in this artifact may be called 'the probability of C' or 'the model's belief'."* So the
   paragraph says "these margins exist at only one cell at both slots" while pointing at a cell where
   a *different* margin exists at both slots. The span/content margin still exists at the elicited
   slot **nowhere** (`OWED.md` B2 stands).
2. **The artifact forbids the quotation.** The same file's `primary_readout.prohibition`: *"THE
   PRIMARY READOUT is exactly this verdict … Everything else in this artifact is SECONDARY and
   DIAGNOSTIC and **may not be promoted afterwards**: the dose class, the grade anchor, A8, every
   floor comparison, **every margin** and every dissociation column."* The sentence promotes exactly
   that.

Minor rider on the same clause: that run measures the **frozen 74**, not the intro's 82
(`n_margin_defined` 74 per turn arm), which the paragraph does not say.

**Why it matters.** A disclosure sentence written to fix a scope defect introduces a worse one, and
it lands on the wrong side of the standing D22 span-vs-first-token decision (PATCHMAP §4 item 9)
inside the paragraph that just took the span side.

**Minimal fix.** Delete "- only the 9b -chat "fold" arm has both" (−9 words). The disclosure stands
without it and is then simply true.

### BL-4 — draft L32 and L38 [applied L19, L23], "the sankey" has no antecedent

**What is wrong.** `T4-I02` deletes gold L9, which was the only place the figure is called a sankey
("The results are presented in the below sankey"). The word survives twice as a **definite
reference**:

- applied L19: "This is not shown in the sankey" / "the final answer the sankey scores"
- applied L23: "our behavioural evals in the sankey"

Nothing in the applied text, the embed, or the Figure 1 caption ever calls the figure a sankey. A
reader hits "the sankey" cold.

**Why it matters.** It is a broken reference visible on a first read, and the draft's OPEN section
flags only the *grey-band* consequence of the L9 deletion — it does not name this one. That is the
same shape of miss as the two defects that killed the previous version.

**Minimal fix.** Either `C02` re-sliced so the caption names the figure ("a sankey of answer flows…"),
or `T4-I02` not taken. There is no honest zero-cost repair inside the current text.

### BL-5 — draft L17 and L38 [applied L5, L23], the mechanism point lands three times, and the
counts the pass removed come straight back

**What is wrong.** The draft's own DUPLICATION section reports the pairs, and its report is accurate
as far as it goes. Two things it does not say, and they are the reason this is a BLOCKER and not a
noted trade:

**(a) The triple, verbatim.**

| # | line | text |
|---|---|---|
| 1 | applied L5 | "I found no single circuit carrying it" |
| 2 | applied L23 | "yet no single lever moves the behaviour" |
| 3 | applied L23 | "**Chat training does not appear to install a dedicated truth circuit.**" |

Plus "correlational at the head level" (L5) against "a correlational read" (L23) against "[the base
and -chat head rankings come from unmatched instruments, so the contrast is qualitative]" (L23) —
**three** landings of the same caveat; and "the causal search returns nulls at every scale" (L5)
against "no write handle beats its matched random floor at any scale" (L23).

**Which survives.** #3 — it is the researcher's own bytes, it is the bolded sentence, and the trade
note is right that it survives every option. #1 goes with TL;DR sentence 3; #2 goes with `T3-03`.

**(b) `T3-03` reverses this pass's own fault 1.** The patchset's "Where the numbers went" table
routes `4/5` and `5/5` head overlap **out of prose** ("now in prose: 'no single circuit carrying
it'"). Applying `T3-03` puts them back — "four of their five", "all five", "0 of 37" — three raw
counts in prose, in a pass whose stated first fault is "Raw statistics in prose" and which
deliberately removed `12 of 34`, `15 of 35`, `13`, `57 and 50 of 82`, `43.52%/14.66%` and
`13 manipulations`. `T3-03` predates tranche 4b and was never register-audited by it.

**Why it matters.** The researcher's live bracket on the same paragraph already says *"this claim
seems like its been repeated several times in different forms?"* (applied L17). Shipping them a draft
that lands it three more times answers their complaint with more of it.

**Minimal fix.** Do not apply `T3-03`; take the trade note's option 3 cut instead. See LENGTH below —
it is also the cheapest cut available.

**Also note:** `PATCHMAP_live.md` §5.4 is **"Bracket conventions and measured bracket load"**, not a
duplication ledger. The draft cites it as one at L112, and `T4b-I04a`'s receipt does the same
(`PATCHSET_tranche4b_intro_register.md:326-327`). There is **no duplication ledger anywhere in
`PATCHMAP_live.md`** — I grepped it. The citation resolves to the wrong section in both places.

---

## SHOULD-FIX

### SF-1 — draft L35 [applied L21], "De Marez et al. see no such reversal" — orphaned antecedent
The gold's preceding sentence was "Read the same pressure off a two-option margin, as De Marez et al.
do, and it runs the other way". `T4b-I05` replaced it, so the *reversal* is never named. The nearest
candidate ("will score -base as steadier than -chat") is two sentences back and is not called a
reversal. **Fix:** "De Marez et al. do not see that inversion" — or name it once earlier.

### SF-2 — draft L35 [applied L21], "their 17 of 23 is a worst-case flip rate"
Category error. `CITATIONS_post1_verified.md:176-178`: "In 17 of 23 Base-IT pairs, IT is more
robust." 17 of 23 is a **count of matched model pairs**; the *criterion* behind each pair's verdict is
the worst-case flip rate. As written the appositive says a count is a rate. **Fix:** "their 17 of 23
is decided on a worst-case flip rate over their manipulations, not a margin".

### SF-3 — draft L35 [applied L21], SYCON — UNVERIFIED source, and a dropped scope
"Gemma is SYCON's own named exception, the narrowest gap they report." Two problems:
`GROUNDING_crossvariant_scale.md:598-599` records **SYCON is UNFETCHED** (PDF-only, no HTML/ar5iv
render) so "its three quoted facts in §11 are unverified" — this sentence is one of them, stated flat
in prose. And `§11` records that SYCON's base arm is **URIAL-prompted base, not raw base**, which the
paragraph drops while using SYCON as its outside base-vs-tuned witness. "Gap" is also undefined, in
the paragraph the researcher rewrote *because* it invented "abstention gap". **Fix:** cut the
sentence (−11 words), or carry "on their URIAL-prompted base arm" and mark the source unfetched.

### SF-4 — draft L30 [applied L17], the SycEval ratio is stated at a scope the ledger bars
`CITATIONS_post1_verified.md:164-168` closes the claim-list: the post **may not claim** "that the two
rates are comparable propensities, or that the asymmetry is uniform". `T3-02b`'s own receipt concedes
"the rates are not a ratio and 43.52/14.66 is not 'three times'". The applied line now reads "revise
toward truth **about three times as often** over their combined math and medical set, **an ordering
that holds for each model**" — the added scope clause and the added per-model clause make an
unlicensed ratio read as fully sourced, and "holds for each model" is an **aggregate** statement
(`:126-127` "No model has regressive above progressive **in aggregate**") that the researcher's very
next sentence contradicts ("on medical advice this reverses for Claude-Sonnet"). I also could not
verify the receipt's assertion that "about three times as often" is "SycEval's own framing" —
**UNVERIFIED**; the ledger records 43.52/14.66 verbatim and nothing about a ratio. **Fix:** "an
ordering that holds for each model in aggregate".

### SF-5 — draft L35 [applied L21], the attribution is now inverted
Gold: "sits next to a broader pattern that SYCON and Gupta et al. report from the outside:
alignment tuning amplifies…" — attribution first, claim second. Applied: "Alignment tuning amplifies
revisability under user pressure, while base models look more resistant - a pattern that SYCON and
Gupta et al. report from the outside." Sentence-initial declarative reads as **asserted**; the
trailing appositive reads as corroboration. `T4b-I05`'s RESIDUAL claims the clause is "attributed …
not asserted" — after the inversion that is no longer true. **Fix:** restore the gold's order, or
open "SYCON and Gupta et al. report from the outside that alignment tuning amplifies…".

### SF-6 — draft L17 [applied L5], a new causal clause, added by this pass
"What chat tuning changes is the policy of answering" attributes a difference to tuning. The draft's
OPEN section correctly flags the researcher's **two** carried causal clauses against their own
notes-L133 instruction ("Keep this descriptive: no causal 'tuning forces' claim") — and does not flag
the **third**, which this pass wrote. There are no staged checkpoints and format co-varies with
variant (`INVENTORY §4.1`), so it is unlicensed on the same two grounds. **Fix:** add it to OPEN, or
"The variants differ in the policy of answering".

### SF-7 — draft L38 [applied L23], "This roughly fits our behavioural evals" now points at a null
`b609418`'s commit body names the dangling "This" as one of the two defects that killed the previous
pass. With `T3-03` applied the antecedent is no longer *missing* — it is **wrong**. "This" now
resolves to a clause-stack ending in "no write handle beats its matched random floor at any scale (at
9b, write-ablating the top heads flips 0 of 37)", and a causal null does not "roughly fit" behavioural
evals. The defect recurred in a second form and is unreported. **Fix:** whichever L25 option is taken,
the sentence needs to open on the sankey directly.

### SF-8 — draft L38 [applied L23], the null is stated harder than the instrument decided
I read the three phase-3b artifacts. `write_drops.wf_to_l` / `wl_to_f` = 0.0/0.0 (9b-it, 2b-it) and
0.0/−0.027 (27b-it), `cross_write.both_at_floor` true at 3/3 — so "no write handle beats its matched
random floor at any scale" is literally true. But the registered verdict at all three is
**`MONITOR_AGAIN`**, not the registered null category `DISTRIBUTED_NULL`, and the reasons include
`backup_restores: true` and `arbiter: "SIGN_DISAGREE"`. The instrument **declined to decide**. Two
riders the prose does not carry: the matched random floor's own drop is also **0.0** (floors are 1.0
against a baseline of 1.0), so this is a comparison of two zeros rather than a contested one; and
`H_read_fold`/`H_read_listen` are **empty lists** with `read_weak_fold: true` at 3/3. The
correlational half of the sentence got its "the instrument issued no verdict" caveat; the causal half
did not. **Fix:** the bracket already exists — extend it, e.g. "…and the causal search returns nulls
at every scale, on an instrument that returns MONITOR_AGAIN rather than a null".

### SF-9 — draft L25 and L27 [applied L13, L15], the 27b alias caveat lands twice, two lines apart
> L13 (`T4b-I03a`): "At 27b -base about a third of those are unresolved aliases, not hedges."
> L15 (`T4b-I03b`): "though at 27b the test drops a small share as unresolved aliases"

Different denominators (12/34 and 15/35 of the elicited NEITHER band, against 13/82 excluded pairs in
`out/gapclose_foldrate_sig.json` — I confirmed both), but to a reader they are one caveat written
twice inside three lines, and the second immediately follows the first observation that carries it.
`T4b-I03b`'s own receipt reasons about not "writing it a third time" at L16 while not noticing it is
already the second. **This pair is not in the draft's DUPLICATION section.** **Which survives:** L15's
— it is attached to the significance claim it qualifies. L13's is the more expendable (−13 words).

### SF-10 — draft L38 [applied L23], bare arm labels in new text, against this pass's own fault 4
`T3-03`'s new prose carries `fold` and `listen` **bare four times** ("fold and listen share four of
their five…", "at -chat fold and listen share all five"). `T4b-I04b`'s receipt states the rule and its
reason: T4-I04b's bare "9b -chat folding" "promoted their coined label into plain description (fault
4)", fixed by quoting. The caption introduces them as `"fold"` / `"listen"`. Applying `T3-03`
reintroduces the fault the same pass fixed elsewhere. **Fix:** quote them, or drop `T3-03`.

### SF-11 — three undefined terms and one word used in three senses
- **"channels"** (applied L21, "both their channels favour the tuned model") — never defined. It
  means De Marez's flip-rate channel vs their margin channel. Fault 3 of this pass is "Undefined
  terms"; the fix introduced a new one.
- **"the grey band"** (applied L21) — the draft's OPEN correctly flags that `T4-I02` deleted its only
  operational definition. Confirmed; it is a real hole and `C02` cannot close it (stale anchor).
- **"the elicited column" / "the reply column"** (applied L21) — never stated in prose. They are
  readable off the figure's own x-labels (`make_figB_matrix.py:263`: "planted / counter reply /
  elicited"), so this is recoverable, but only by a reader who studies the figure.
- **"cell"** is used in three incompatible senses: the caption says a cell is a model × experiment
  type (12); applied L19's "at every cell" / "the only two cells where it does" means a model ×
  variant (6); applied L21's "at every cell" means the six `-it` panels. With L9 gone the caption is
  the only definition, and the prose contradicts it twice.
**Fix:** gloss "channels" in four words, and pick one word for the 6-way partition ("at every model",
"at all six").

### SF-12 — draft L86, the DELTAS narrative under-attributes the growth
"the growth is entirely the `T3-03` decision and nothing else drifted" is true of the *arithmetic*
against the patchset ledger's own +52 baseline, and false as a description of the text. Measured per
block against the bytes it replaces: `T4b-I07` **+58**, `T3-03(a)+(b)` **+46**, `T4b-I01` **+24**,
`T4b-I04a(ii)` **+23**, `T4b-I04b` **+21**, `T4b-I03b` **+16**, `T4b-I03a` **+13**, `T4b-I04a(i)`
**+11**, `T3-02b` **+10**, against `T4b-I05` **−51** and `T4-I02` **−50**. The single largest grower
is the TL;DR, not `T3-03` — `f6dc860`'s own commit body said so ("The TL;DR is 58 of that"). **Fix:**
print the per-block column; the reader is being asked to approve a growth decision on a framing that
hides which block costs most.

### SF-13 — draft L3, "the finished intro"
"strip them and what remains is the finished intro, byte for byte" — three of the twelve applied
blocks are not finished: `T4b-I05` and `T4b-I07` are **NEEDS-RESEARCHER-DECISION**, and `T3-03` is
`PATCHMAP §4` **item 1** ("whether the intro carries the mechanism contrast at all, and in what form
… No run supports the sentence as written"). `T4b-I05` is additionally §4 **item 15**. The OPEN
section says so; the header does not, and the header is what a skimming reader reads. **Fix:** "what
remains is the intro with this set applied — three of the twelve are offers, see OPEN".

---

## LENGTH

**It grew: 1132 → 1253 words, +121 (+10.7%).** The brief's constraint ("the intro stays short") is
not met, and the TL;DR alone went **54 → 112 words**, more than doubling the one paragraph the
researcher reads first.

Per-block deltas measured on the exact bytes (`split()`), against the version each block replaces —
these reconcile to +121 exactly:

| block | delta |
|---|---|
| `T4b-I07` (TL;DR) | **+58** |
| `T3-03` (a)+(b) | **+46** |
| `T4b-I01` | +24 |
| `T4b-I04a(ii)` | +23 |
| `T4b-I04b` | +21 |
| `T4b-I03b` | +16 |
| `T4b-I03a` | +13 |
| `T4b-I04a(i)` | +11 |
| `T3-02b` | +10 |
| `T4b-I05` | −51 |
| `T4-I02` | −50 |

**Cheapest honest cuts, ranked by words-saved per unit of harm.** Every one of these also closes a
finding above; none requires inventing prose.

1. **Drop `T3-03`, take the trade note's option 3 cut — −69** (−46 + −23). Closes BL-2, BL-5, SF-8,
   SF-10 and half of SF-7 in one move. Cost: the dangling "This" returns, with the repair `f6dc860`
   already priced (open on the sankey directly, ~0 words) — which BL-4 requires anyway.
2. **Drop TL;DR sentence 2 — −26.** Closes BL-1. Nothing depends on it; `T4b-I07`'s own STATUS says
   sentences 2 and 3 are independent and either can be dropped.
3. **Drop `T4b-I04a(i)`'s gloss or `T4b-I03a` — −11 / −13.** `T4b-I03a` is the weaker of the two 27b
   alias sentences (SF-9). The flip-rate gloss earns its 11 words; the alias sentence does not earn
   its 13 alongside L15.
4. **Delete "- only the 9b -chat "fold" arm has both" — −9.** Closes BL-3 without losing the
   disclosure.
5. **Drop the SYCON exception sentence — −11.** Closes SF-3; it is the one sentence in the paragraph
   sourced from an unfetched PDF.

Taking 1, 2 and 4 lands the intro at **+17 against gold** and removes four of the five BLOCKERs.
Taking 1–5 lands it at **−16**, i.e. shorter than the gold, which is what "stays short" would mean.

---

## REGISTER

Against `STYLECARD_researcher.md`, sentences that read wrong and why:

- **applied L5, "It never abstains at the final answer, at every scale - the one 27b exception is an
  alias miss, not a silence."** The gold held the exception in a bracket, where a caveat at arm's
  length is their idiom (§A8). Promoted into prose, "never … at every scale" collides with "the one
  27b exception" **inside the same sentence**. In brackets it read as a note; in prose it reads as
  self-contradiction.
- **applied L5, "What chat tuning changes is the policy of answering."** "the policy of answering" is
  a named abstraction for their own concept — §B1 says they coin exactly two things, both from an
  operation, and say so in the sentence that names them.
- **applied L5 as a whole.** Five sentences, 112 words, one bracket, a colon-joined synthesis and a
  30-word closer. §A2's signature is long-then-**snap**; this paragraph never snaps. The gold's TL;DR
  did ("It never abstains.").
- **applied L7, "has one of the pair items already in its own turn, as though it had said it".**
  Three subordinate clauses before the main verb chain, and "as though it had said it" is an
  explanatory simile (§B2: zero metaphors/analogies in the corpus). Their own construction for the
  same operation — quoted in `T4b-I01`'s own receipt, POST1 L21 — is flatter and shorter: "we make
  the model predict the next tokens from a set transcript where it has already output the correct
  answer $C$". The receipt chose a softer paraphrase than the one it cites as the model.
- **applied L7, "raw Q:/A: at -base, chat turns at -chat".** §A4/§A5 are explicit that examples go on
  their own labelled lines, never inline in prose. This inlines a label form. `T4-I01`'s residual
  worried about backticks and closed on the unfenced form; the placement question was never asked.
- **applied L21, "De Marez et al. see no such reversal - both their channels favour the tuned model,
  and their 17 of 23 is a worst-case flip rate over their manipulations, not a margin - because their
  readout has no "abstain" outcome."** 44 words carrying three separate corrections, with the
  load-bearing "because" arriving after a 22-word parenthetical. Their long sentences stack clauses;
  they do not suspend the main clause's causal verb behind a correction.
- **applied L23, "fold and listen" bare ×4** — SF-10.
- **"abstain" is handled four ways in one document**: bare in the TL;DR twice (theirs), quoted at
  applied L13 (theirs), quoted at applied L21 (`T4b-I05`'s new text). Mostly inherited, but the pass
  added a fourth instance without reconciling.

Nothing in the applied prose trips §B3–§B7: no hype adjectives, no "Moreover"/"Notably"/"Importantly",
no hedge words, no straw-objection heading, no wrap-up paragraph. British spelling holds. That part is
clean.

---

## ANNOTATION DISCIPLINE

Four `>>` markers for a 31-line intro is light, and each is one line. Two problems:

- **NIT-A. The markers render as blockquotes.** `>> T4b-I07 …` immediately precedes the TL;DR's `>`
  line. In any markdown renderer those two lines merge into a single blockquote with a nested part —
  so on the researcher's screen the marker appears to be *inside* their TL;DR. The one marker whose
  separation matters most is the one that fails. **Fix:** a blank line between marker and TL;DR, or a
  non-`>` marker prefix.
- **NIT-B. `SUPERSEDED ORIGINALS` (draft L116-136, 21 lines) reproduces three gold lines verbatim** in
  a derived file, when the identical bytes are the CURRENT fence of each block in the patchsets and
  the gold is one file away. It is the largest single block of apparatus and the one with least claim
  on the reader.

Nothing else belongs in an appendix; the intro fence itself is clean prose.

---

## NIT

- **N-1.** Draft L80, "`[` , prose only … 12". `PATCHMAP §5.4` and `COMPOSE_post1_brief.md:182` both
  record the live intro at **11** prose brackets. The difference is `[` **characters** vs bracket
  **notes** — applied L21's nested "[… [the abstention gap sits] …]" is one note and two characters.
  Internally consistent, but it will read as a contradiction against the repo's own count.
- **N-2.** Applied L23, "flips 0 of 37" — 37 appears in an intro whose stated design is 82, with no
  explanation. `n_family` is 74 and `n_eval` 37 in all three phase-3b artifacts. The same 0-of-37
  holds at 2b-it, so naming 9b is honest but arbitrary.
- **N-3.** Applied L15, "at 27b the test drops a small share" — the test also drops 5 pairs at 2b
  (`gapclose_foldrate_sig.json`, 2b-base vs 2b-it, `n_excluded_pairs` 5). Naming only 27b is defensible
  (13 vs 5) but reads as if 27b is the only cell that loses items.
- **N-4.** Applied L21, "in the reply column it survives at every cell, in replies that name both
  answers" — asserts universally what `TAXONOMY_withholding.md:101-103` supports for a 63-item class
  (62 of them `-it`, across **both** slots) against 61 strict-register `-it` reply-column NEITHER
  labels. Two different populations, near-equal by count. It was the researcher's own bracket, so the
  approximation is theirs; folding it into prose promotes it from note to assertion.
- **N-5.** Applied L19, "adding another one to this page wasd vetoed by Fable" — the researcher's own
  bytes and correctly untouched, but it is a private-workflow reference standing in publishable prose.
  Worth a line in OPEN so it is a decision rather than an oversight.
- **N-6.** Applied L21, "the narrowest gap they report" reintroduces a bare "gap" into the one
  paragraph the researcher rewrote because it "invents terminology like 'abstention gap'". Cosmetic,
  but it is the exact word.

---

## VERDICT

**Shippable with named fixes — but not as it stands, and the fixes are not cosmetic.** The machinery
is sound: I re-derived the applied text byte-for-byte from the gold and the twelve blocks, every
anchor was unique, the deltas reconcile, and the retraction constraints (R-12's three strings, the
"distributed" ban, the Perez hold) are all honoured. The apparatus-swamp defect from `b609418` is
genuinely fixed. What is not fixed is the *text*: five findings I would not let past — an unsupported
cross-variant claim in the TL;DR that two repo measurements point against (BL-1), a mechanism-scope
qualifier that names the wrong arm and implies coverage that does not exist (BL-2), a slot disclosure
whose exception is a first-token readout the same paragraph disowns and whose source artifact
explicitly forbids the quotation (BL-3), "the sankey" left as a definite reference to a word the pass
deleted (BL-4), and the no-circuit claim landing three times while `T3-03` puts back the very counts
this register pass removed (BL-5). Three of those five (BL-2, BL-4, BL-5) trace to decisions the draft
*reports* but does not treat as defects, and two of them (BL-1, BL-3) come from blocks whose receipts
argue the opposite of what the artifacts say — which is the same failure mode as the previous
version: careful mechanical checking, no adversarial read of the prose. **The single highest-value
action is dropping `T3-03` and taking the trade note's option 3**: it is −69 words, closes BL-2, BL-5,
SF-8 and SF-10 outright, and returns the intro to within +17 of the gold once BL-1 and BL-3 are also
cut. Do that, name BL-1/BL-3/BL-4 and SF-1 through SF-6 in OPEN as researcher decisions rather than
applied text, and this goes to the researcher as an honest reading aid. A rebuild is not warranted —
the byte-level foundation is correct and re-deriving it would risk what is currently right.
