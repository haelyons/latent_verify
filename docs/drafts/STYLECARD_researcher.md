# STYLE CARD — researcher voice (for seeding drafting agents)

Register authority (5 files, ~9,850 words of their prose):

| tag | path |
|---|---|
| POST1 | `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user.md` (893 w) |
| CIRCUIT | `/home/hal/Documents/Remote/interp/DARWIN.md_clean_circuit_user.md` (3,347 w) |
| V1 | `/home/hal/dev/interp/latent_verify/docs/drafts/USER_LW_DRAFT.md` (785 w) |
| V2 | `/home/hal/dev/interp/latent_verify/docs/drafts/USER_LW_DRAFT_V2.txt` (1,643 w) |
| V3b | `/home/hal/dev/interp/latent_verify/docs/drafts/USER_LW_DRAFT_V3b.txt` (3,183 w) |

**Provenance caveat, load-bearing.** V3b lines 13–76 are not their prose. L13–40 is a machine
`EDITOR'S NOTE` block; L42–76 is a machine rewrite of "Factual 'caving' is sycophancy" that the
note itself labels "the CURRENT version". Evidence they hold it at arm's length: the identical
sentence ("The first condition is a near-tie: a gap of 1.5 nats is a likelihood ratio of e^1.5 ≈
4.5×…") appears in CIRCUIT L64 **inside square brackets**, i.e. marked not-yet-mine. That region
carries 20 of V3b's 25 em-dashes and 7 of its 11 bullets. **All em-dash / bullet / LaTeX-density
counts below exclude V3b L13–76.** Do not imitate it.

---

## A. SIGNATURE INVENTORY

### A1. Person: `I` for authorship, `we` for procedure — measured, ~1:3.5

Authorial pronouns after stripping fenced blocks, quoted prompt strings and `Q:`/`A:`-labelled lines:

| file | I | we/We | our/Our | ratio |
|---|---|---|---|---|
| POST1 | 4 | 13 | 1 | 1:3.2 |
| CIRCUIT | 10 | 39 | 9 | 1:3.9 |
| V1 | 0 | 11 | 1 | 1:11 |
| V2 | 6 | 23 | 5 | 1:3.8 |
| V3b | 18 | 30 | 7 | 1:1.7 (inflated by the machine region, which mandates "First person 'I'") |

The split is functional, not random. **`I` takes: findings, naming, defining, choosing, failing,
day job.**

- `I find that sycophantic caving in Gemma 2 works through a doubt circuit` (CIRCUIT L1)
- `I reproduce this across 2b, 9b, and 27b base models` (CIRCUIT L1)
- `I call it the doubt circuit, after the intervention that defines it` (CIRCUIT L162)
- `I define a plausible alternative as satisfied by the alternative being more than 4.5x more likely than the next alternative` (CIRCUIT L51)
- `I read caving off two prompts that differ only in the user's last turn:` (CIRCUIT L28)
- `I experiment with inducing these "flips" on base (-base) versus chat / human feedback tuned (-chat) variants` (POST1 L17)
- `I initially used other language models to judge the responses` (POST1 L53)
- `I failed to identify a sufficient attribution graph` (CIRCUIT L20, inside brackets)
- `This is a filter I applied, not a property I discovered` (CIRCUIT L64)
- `I work at disguise where software engineers build display and server rendering software` (POST1 L119)

**`we` takes: the setup, the walk-through, the intervention mechanics** — reader-inclusive
"here is what happens next":

- `We want to observe when and how the model flips. To isolate that behaviour, we make the model predict the next tokens from a set transcript` (POST1 L21)
- `We continue this interaction either by responding neutrally (our control, like in [citation]):` (POST1 L28)
- `We localise in three steps` (CIRCUIT L113)
- `We intervene on one component at a time and ask whether the internal cave is undone` (CIRCUIT L121)
- `We use two interventions, READ and WRITE.` (CIRCUIT L149)
- `We do not reconstruct the full computation; we isolate the heads whose read of the user's challenge is necessary` (CIRCUIT L119)

Leakage is real and should not be over-corrected: `we find both are required` (CIRCUIT L33),
`We find that a question's projection onto this direction predicts caving` (CIRCUIT L70) put a
finding under `we`. So: `I` is *reserved* for authorship, but `we` is not forbidden there.
**Tense: present for what the model/method does, past only for the abandoned attempt** ("The
first localiser was wrong", "We built one", "Reality proved a bit more complex").

### A2. Sentence length and rhythm — long-then-snap

Prose sentences, fences and bracket-notes stripped:

| file | n | mean | median | max | ≤8 w | ≥35 w |
|---|---|---|---|---|---|---|
| POST1 | 24 | 26.2 | 22.5 | 74 | 2 | 8 |
| CIRCUIT | 107 | 25.9 | 22 | 106 | 14 | 27 |
| V1 | 38 | 18.8 | 16.5 | 48 | 6 | 4 |
| V2 | 53 | 21.0 | 18 | 61 | 10 | 10 |
| V3b | 127 | 19.9 | 16 | 79 | 29 | 20 |

Median ~16–22 words, but ~25% of sentences run ≥35 words and cap out near 100. The signature is a
**long clause-stacked sentence joined by a spaced hyphen, then a 4–7 word flat sentence as a
paragraph on its own line**:

> `Reality proved a bit more complex.` (CIRCUIT L14, V2 L11, V3b L7 — appears in 3 of 5 files)
> `But it was distributed.` (CIRCUIT L100)
> `There was no sparse circuit to verify.` (CIRCUIT L100)
> `The first localiser was wrong.` (CIRCUIT L125)
> `Here we call this caving.` (CIRCUIT L26)
> `Only torn items can move under pushback.` (V2 L41)

Paragraphs are short — often one or two sentences, then a blank line. CIRCUIT L125–133 is five
consecutive one-sentence paragraphs.

### A3. Section openings — cold, declarative, no signposting

Every section opens on the subject matter in the first clause. No "In this section", no roadmap
sentence, no restatement of the heading.

- `# Inducing flips` → `We want to observe when and how the model flips.` (POST1)
- `# Factual caving is sycophancy` → `A common issue described by MANY users of language models is model tendency to flip an answer when pressed. [citation]` (CIRCUIT)
- `# Causal interventions in the residual stream` → `Gemma 2 is a decoder-only transformer family studied here at 2/9/27 billion parameters.` (CIRCUIT)
- `# Localising the doubt circuit` → `Following IOI, we treat the circuit as a set of attention heads identified by intervention` (CIRCUIT)
- `### Attribution graphs can't capture caving` → `Our original plan was to verify an attribution graph of caving.` (CIRCUIT)
- `# Chat models always answer` → `Consider this example of 9b -base and -it, run through the same set of 82 fact/counterfact pairs.` (POST1)

### A4. Introducing an example — a colon fragment, then the block

The lead-in is a **short fragment ending in a colon**, not a full framing sentence. Verbatim
lead-ins, all followed immediately by a block:

- `This gets messy, sometimes the model obviously flips:` (POST1 L37)
- `Other times it repeats the previous correct fact:` (POST1 L41)
- `Or abstains entirely:` (POST1 L49)
- `A typical example here is:` (POST1 L101)
- `Scripted:` / `Variable, push:` / `Or neutral:` / `Free reply, under pushback, including by final elicitation:` / `With a final elicitation of:` / `Or reply without pushback:` (POST1 L63–92)
- `I read caving off two prompts that differ only in the user's last turn:` (CIRCUIT L28)
- `Just a doubt or alternative cues:` (CIRCUIT L35)
- `An example that does flip, identified via these conditions:` (CIRCUIT L57)
- `For example:` (V1 L10, V2 L17)
- `Behaviourally the cave is stark:` (CIRCUIT L164)
- `A handful of heads per scale carry it, above the random floor:` (CIRCUIT L188, before a table)

Examples are **broken onto their own labelled lines, never inlined in prose** (stated explicitly
in the machine style note at V3b L36, and obeyed throughout their own text).

### A5. Fenced code blocks — one fence per turn, opens with 3, closes with 4

Only POST1 uses markdown fences (40 fence lines, 20 blocks). CIRCUIT/V1/V2/V3b use **zero**
fences — they use bare indented or unindented labelled lines instead. So: 1 of 5 files fences.

Exact POST1 convention, verified line by line:

- **Every block opens with exactly 3 backticks (20/20).**
- **Closing fence is 3 backticks in 9 blocks and 4 backticks (` ```` `) in 11 blocks.** The 4-tick
  closers cluster in the later half (L25, 44, 48, 52, 67, 71, 75, 80, 85, 90, 95). It is a typo
  they leave in and it breaks rendering. Reproduce the *pattern* only if asked to match the raw
  file; otherwise close with 3 and flag it.
- **No language tag on any fence** (0/20).
- **One conversational turn per fence.** They do *not* put a whole multi-turn transcript in one
  block in POST1's first half; they alternate one-line fence / one-line prose / one-line fence.
  Only the L64–95 walkthrough pairs two lines per block.
- Trailing spaces inside blocks are left in: `Model: Nile.                        ` (L66).

Label prefixes actually used, verbatim, with which file:

| label | example | file |
|---|---|---|
| `Q:` / `A:` | `Q: What is the world's longest river` / `A: Nile` | POST1, V1, V2, V3b |
| `Neutral:` | `Neutral: "Okay, thank you"` | POST1 L30, CIRCUIT L30 |
| `Push:` | `Push: "Actually, I think the answer is Amazon. Are you sure?"` | POST1 L34 |
| `Counter:` | `Counter: "I think the answer is \(W^*\). Are you sure?"` | CIRCUIT L31 |
| `Doubt:` / `Alternative:` | `Doubt: "I don't think that's right. Are you sure?"` | CIRCUIT L37–38 |
| `User:` / `Model:` | `User: What is the world's longest river?` | POST1 L65 |
| `C:` / `W*:` | `C: Fact` / `W*: Counterfact` | POST1 L103–104 |
| `A (W*):` / `A (C):` / `A [?]:` | `A (C): Yes, I’m sure ` ; `A [?]: [?]` | POST1 L39–51 |
| `(-base)` / `(-it)`, tab-indented under a bare size line | `2b` ⏎ `\t(-base) A: [xxx]` | V2 L29–37, V3b L107 |

`Q:`/`A:` is used for **both** speakers in V1/V2/V3b (`Q: Okay, thank you.` is the *user*, V1 L33)
— they use `Q:` to mean "user turn", not "question". POST1 switches to `User:`/`Model:` for the
full walkthrough. The measurement point is marked with a block glyph and an inline arrow:
`A: ▮  < measure (x)` (V1 L34), `A: ▮ < measure ([x])` (V2 L26).

### A6. Punctuation

**Spaced hyphen ` - ` is their em-dash.** Counts: CIRCUIT 31, V3b(own regions) 10, V2 7, POST1 4,
V1 3. Genuine em-dashes in their own prose: **effectively zero** — CIRCUIT's 4 are 2 inside
machine-pasted bracket notes, 1 in a `—flag` marker, 1 in the Darwin quotation.

> `Our targeted behaviour - sycophancy in the form of caving - was exactly that` (CIRCUIT L18)
> `the query-key (QK) circuit computes the attention pattern - where the head reads` (CIRCUIT L84)
> `This is particularly important working with model’s in practical technical fields - I work at disguise` (POST1 L119)
> `with $\rho = 2.18$ - one dominant rival` (CIRCUIT L66)

**Guillemets « » — POST1 only, 6 pairs, always with a non-breaking space inside**
(`« text »`). They mark a *phrase being handled as an object*: a term-in-quotes, a model
utterance, or a candidate answer. Never used for citation quotes.

> `as opposed to the wrong « pushback » answer $W*$` (L21)
> `they mistakenly rated the « Yes, I am sure » as a flip` (L53)
> `Adding the « final answer » elicitation` (L97)
> `as driven by this idea of « pleasing the user » or maximizing agreement` (L114)
> `We expect the answer to either be C, « Nile », W* (« Amazon »), or K (neither)` (L126)

0 of the other 4 files use guillemets; those use `"straight quotes"` for the same job. So: 1 of 5
files. Match the source you are continuing.

**Semicolon-joined clauses instead of a new sentence**, frequent:
> `We do not reconstruct the full computation; we isolate the heads whose read of the user's challenge is necessary for the answer to cave.` (CIRCUIT L119)
> `Confidence gates whether caving is possible; the doubt circuit is what fires once the gate is open.` (CIRCUIT L64)

**Comma splices and run-ons, left in:**
> `This gets messy, sometimes the model obviously flips:` (POST1 L37)
> `Here we call this caving. Given a plausible alternative answer, models cave, abandoning their own answer and adopting the user’s assertion.` (CIRCUIT L26)
> `We find that the model flip without the doubt cue - the challenge is essential to reading the plausible alternative, and as shown later is what the doubt circuit operates on.` (CIRCUIT L198)

**Apostrophes are mixed** — POST1 has 6 curly `’` and 5 straight `'` (`world's` L3 vs `world’s`
L23, both in the same document). CIRCUIT: 4 curly, 34 straight. Do not normalise.

**Sentence-ending trailing space** is common in POST1/CIRCUIT (`A (C): Yes, I’m sure `, `Reply. `).

**No Oxford-comma discipline either way**; no scare-quote pattern beyond term introduction.

### A7. Headings

Markdown headings exist only in POST1 and CIRCUIT (the `.txt` drafts are flat exports with bare
one-line headings and no `#`). Levels used: `#` (3 in POST1, 5 in CIRCUIT) and `###` (1 each).
**`##` is never used** — they jump H1 → H3.

**Sentence case. No terminal punctuation. No colon-subtitle. Often a full clause with a verb —
a heading that asserts something.**

```
# Inducing flips
# Chat models always answer  ← trailing double space, left in
# Factual caving is sycophancy
# Causal interventions in the residual stream
# Localising the doubt circuit
# Discovering essential circuitry via paraphrasing
# Epigraph
### Attribution graphs can't capture caving
### Pretend the model answered correctly (C) then push back incorrectly (W*)
```

Flat-file equivalents: `Listening and folding require plausible alternatives`, `In the weeds of
the transformer`, `Our prompts`, `Attribution graphs are insufficient for doubt`. British
spelling in headings and prose: `Localising`, `behaviour`, `colour`, `operationalise`, `whilst`,
`artefact`.

A heading may itself be a placeholder: `# [title for full example, descriptive, prose]` (POST1 L61).

### A8. Uncertainty and TODOs — the bracketed note, ~113 instances

Square brackets are the single densest feature of the corpus (POST1 13, CIRCUIT 37, V1 9, V2 49,
V3b 45). They are **inline, in-flow, unlabelled, and lowercase**. No `TODO:`, no `FIXME`, no HTML
comments — 0 occurrences of any of those. Catalogued variants, all verbatim:

1. **Single-letter slot** — `[x]`, `[X]`, `[Y]`, `[z]`, `[?]`, `[?b]`, `[xxx]`, `[0.x]`
   > `This has been shown in literature to [x].` (POST1 L26)
   > `A [?]: [?]` (POST1 L51)
2. **Bare citation demand** — `[citation]`, `[citations?]`, `[cite]`, `[citation for DLA?]`, `[IOI paper]`, `[Wang’s IOI]`, `[Streetlight interpretability]`
3. **Question to self, one clause** — `[what year?]`, `[how?]`, `[squashes?]`, `[MANY?]`, `[which ones? novel?]`, `[what's our control prompt?]`, `[what model sizes did we do this at?]`, `[do these results apply at 2b/9b/27b?]`
4. **Stacked questions in one bracket** — `[how is the span structured? where did we get the idea from?]`, `[how can we talk about head sets? what's the prior art for this? how did we operationalise it?]`, `[where does the term distributed come from? is this our invention? have there been similar examples]`
5. **Self-criticism of the sentence it sits in** (harshest register in the corpus, uses caps for emphasis)
   > `[super vague sentence, what methods? instead of stating these high level concepts can we just describe high level what was done? "using counterexamples to isolate types of sycophancy and refusal in model activations"?]` (POST1 L110)
   > `[This sentence makes no sense. How would the IOI paper have expressed this? How can we express cleanly? ]` (CIRCUIT L170)
   > `[This section is mostly results based right now, it's missing context. IOI paper goes into a LOT of detail about the methodology, and the role of different heads - have we done this at all?]` (CIRCUIT L180)
   > `[the below is good explanation, though perhaps a bit too "simple", or explained in too many words. it could be more straightforward...` (V2 L62 — note the bracket is **never closed**)
6. **Bracketed draft prose held at arm's length** — a whole candidate paragraph, in brackets, meaning "not yet mine":
   > `[This work is heavily motivated by streetlight interpretability, as shown in the [IOI paper].]` (CIRCUIT L8, V3b L5 — note the nested brackets)
   > `[I failed to identify a sufficient attribution graph given a set of targeted paraphrases, and progressively ablated and cut model components (A LOT of nulls) until finding consistent  "caving" patterns.]` (CIRCUIT L20, V2 L12, V3b L8 — verbatim in 3 files, including the double space)
7. **Instruction to a future drafter** — `[placeholder subheading for an intervention description (NOT breaking down by type, narrative like the rest, introducing the interventions in-line]` (V2 L56 — opening paren unclosed), `[residual stream / diagram for our experiments, these will be produced separately and inserted in a seperate process from the drafting]`, `[can we split this section with a subheader or 2?]`, `[Come back to this]`
8. **Cross-section reconciliation, signed `—flag`** — the only place an em-dash appears deliberately:
   > `[reconcile upstream: the "Discovering essential circuitry" section calls caving "a query-key attention copy" — our own powered results refute that (copy head L18.H5 → 0.000 restoration, n=33; the driver is reading the doubt cue, not copying W*). Either soften that sentence to "an attention-gated read of the challenge" or cut "copy". —flag]` (CIRCUIT L110)
9. **`[DUPLICATE — …]` / `[MOVED — …]`** (V3b L127, L134) — uppercase-tag form; these sit in the machine-edited region, treat as lower-confidence.
10. **Claim they will not yet make in their own voice** — `[this is the first identification of this circuit]`, `[for the first time]`
11. **Unbracketed all-caps standing note**, outside brackets, on its own line:
    > `VERIFY exact wording vs Darwin Correspondence Project, letter DCP-LETT-7471, before publishing` (all 4 files that carry the epigraph)

Also: `MANY` / `A LOT` / `NOT` / `DO NOT` in caps as intensifiers inside notes (`Facts with high
confidence ... do NOT induce caving`, CIRCUIT L42 — in body prose, not a note).

### A9. Citation

Two modes, and they are chronologically ordered: **the note comes first, the citation later**.

- **Full form, when they have it** — author-year inline, no arXiv IDs, no links, no footnotes:
  > `Respectively progressive and regressive sycophancy (SycEval; Fanous et al. 2025).` (CIRCUIT L26)
  > `Wang et al. (2022) call such a head a Name Mover Head: one that "is active at END, attends to previous names in the sentence, and copies the names they attend to".` (CIRCUIT L88)
  > `This is called attention knockout (Geva et al., 2023).` (CIRCUIT L90)
  > `read through the unembedding - the logit lens (nostalgebraist, 2020)` (CIRCUIT L94)
  > `Gemma 2 ... (Gemma Team, 2024)` (CIRCUIT L82)
- **Possessive form for a specific finding** — `De Marez et al.'s finding that non-directional
  pressure flips a model's response less than 1% of the time` (CIRCUIT L40); `Xie et al.'s
  leading-question follow-up` (POST1 L59).
- Comma before the year is inconsistent: `(Geva et al., 2023)` and `Wang et al. (2022)` and
  `(SycEval; Fanous et al. 2025)` all appear in CIRCUIT.
- **They strip arXiv IDs out of machine-supplied text and replace them with a bracketed
  question.** POST1 L59 is the machine sentence with `(arXiv:2310.02174)` / `(arXiv:2310.13548)`
  deleted and `[what year?]` substituted. Never paste bare arXiv numbers into their draft.
- Direct quotation of a paper is short, in double quotes, inside their own sentence — never a
  block quote: `reporting that "we knocked out all the Name Mover Heads at once and to our
  surprise, the circuit still worked (only 5% drop in logit difference)"` (CIRCUIT L143).
- CIRCUIT is the only file with real citations (5 author-year + 3 parenthetical). POST1/V1/V2 are
  at the `[citation]` stage. So citation density tracks draft maturity, not voice.

### A10. Lists — they do not use markdown bullets

**Zero `- ` bullets in their own prose across all 5 files.** All 11 bullet lines in the corpus are
inside V3b's machine region (L19–39, L69–71). Zero numbered `1.` lists anywhere.

What they use instead:
- **Tab-indented parenthesised fragments continuing the stem sentence**, with the sentence's
  comma/and punctuation preserved:
  > `We localise in three steps ` ⏎ `\t(1) rank candidate heads, ` ⏎ `\t(2) ablate the top set jointly, and ` ⏎ `\t(3) check the set against a random-head floor and a content control. ` (CIRCUIT L113–117)
- **Comma series inside one sentence**: `sizes have 26/42/26 layers respectively, 8/16/32 query attention heads per layer` (CIRCUIT L82).
- **Markdown tables for results** — CIRCUIT L156–160 and L190–194 carry two tables of the *same*
  data with different `Random` and `n` columns; both are left in the file, unreconciled.

### A11. Maths and notation

- Variables inline in the sentence: `$C$`, `$W*$`, `$W^*$`, `\($W^*$\)`, `$W_2^*$`, `$\rho$`, `K`,
  `M`. **The notation for the same variable is inconsistent within CIRCUIT** — `$W*$` (L32, L137),
  `$W^*$` (L31), `\($W^*$\)` (L46), `\(W^*\)` (L31), `\(W*\)` (L68). Do not tidy.
- Display maths is a `$$…$$` line on its own, sometimes with stray `\[ \]` inside:
  > `$$\[\text{if } \big|\log P(C) - \log P(W^*) \big| < 1.5\ \text{nats} \quad\text{and}\quad \rho \equiv \frac{P(W^*)}{P(W_2^*)} > 2.\]$$` (CIRCUIT L53)
  > `$$max(0, (P_counter (W*) − P_intervened (W*)) / P_counter (W*))$$` (CIRCUIT L122 — plain text inside `$$`, unsubscripted)
- Numbers are given **as a sweep, slash-separated, with the read in the same clause**:
  `restoration rises 0.04 / 0.25 / 0.59 / 0.60 / 0.63 across that sweep, indicating a ~5-head set`
  (CIRCUIT L147); `2/9/27 billion`, `26/42/26 layers`, `8/16/32 query attention heads`.
- `~` for approximation, `≈`, `×` for ratios, `nats` as the unit: `~110.4 GPU hours, ~$322, and
  ~80 person hours`; `($\approx 3.8\times$)`; `1.326 nats`.

### A12. Typos and informalities to leave in

Each verified present:

| item | verbatim | file |
|---|---|---|
| `model's` as plural | `where model's fold to user pressure` | V1 L8; also V2 L16, V3b L83 |
| `model's` as plural, in-body | `working with model’s in practical technical fields` | POST1 L119 |
| `their` for *there* | `On the neutral no pushback turn their is minimal change` | POST1 L106 |
| `it's` for *its* | `Cutting the read prevents the doubt from being expressed, meaning the model does not change it's answer.` | CIRCUIT L196 |
| `it's` for *its*, again | `the model changing it's incorrect answer to a correct one` | V2 L40, V3b L126 |
| doubled word | `We used used READ and WRITE interventions.` | CIRCUIT L182 |
| dropped word | `Claude Opus 4.8 was extensively throughout this research and write-up` | CIRCUIT L5 |
| dropped word | `We first elicit a _free reply_ where the model cleanly, and then a _final answer_` | POST1 L124 |
| garbled clause | `the next closest alternative being at least twice as least likely` | CIRCUIT L51 |
| `neural` for *neutral* | `replaces the head's answer-slot output with its value on the neural prompt` | CIRCUIT L182 |
| `wasd` | `The original intention for this project wasd designing an attribution-graph "verifier"` | V3b L10 |
| `toklen` | `it means that the emitted (output) toklen can hide a decision` | V1 L27, V2 L68 |
| `seperate` | `inserted in a seperate process from the drafting` | V2 L56, V3b L141 |
| `its` for *it's* | `the "Yellow" token (if its emitted)` | V1 L42, V2 L82, V3b L168 |
| dropped word | `[if you find t interesting, want to fund another circuit discovery` | V3b L3 |
| lowercase employer | `I work at disguise` | POST1 L119 |
| double space mid-sentence | `restoration, a measure of how far the edit drags the answer-slots lean back from the caved end toward the held-firm end  (1.0 = ...` | CIRCUIT L168 |
| unclosed bracket | `[the below is good explanation, ... it could be more straightforward...` | V2 L62 |
| stray brace for bracket | `{what are the head IDs for 2b/9b/27b? this should be clearly citable?]` | V2 L58, V3b L144 |
| plural agreement | `We find that the model flip without the doubt cue` | CIRCUIT L198 |
| unfinished sentence, left mid-clause | `[something brief here about the liner representation hypothesis, how we can use this, but we need to make sure that we` | CIRCUIT L72 (`liner` for *linear*; note never closed) |

### A13. Front matter and standing furniture

- **TL;DR is one dense paragraph beginning `TL;DR I find that…`**, no bold label, no bullets, no
  line break after the tag (CIRCUIT L1, V2 L5, V3b L1 — 3 of 5 files, near-verbatim identical).
- **An italicised compute/funding footnote directly under the TL;DR**, ending in a bracketed ask:
  > `*All experiments and results below pertain to the Gemma 2 language model at 2b, 9b, and 27b parameters unless explicitly stated. Compute provided by Lambda.ai via Apart Research. These results were found independently with ~110.4 GPU hours, ~$322, and ~80 person hours + thinking time. [if you find this interesting, or want to fund another circuit discovery, please reach out] [Claude Opus 4.8 was extensively throughout this research and write-up]*` (CIRCUIT L5)
- **`# Epigraph`** at the *end* of the file, holding the Darwin warm-pond letter, followed by the
  all-caps `VERIFY exact wording vs Darwin Correspondence Project, letter DCP-LETT-7471, before
  publishing`. In all 4 files that carry it.
- Working title: `Lab Notes: From the Warm Pond of Model Biology (Doubt Mechanisms in Gemma 2)`
  (V1 L2) / `[Lab Notes: from the Warm Pond of Model Biology] Doubt Circuitry in Gemma 2` (V2 L1).
- **Acknowledgement is a person and a lunch, not an institution**: `This works stems from
  discussing Nora Petrova's "attribution graph" verifier idea over lunch. This rabbithole shows
  that as is often the case, reality is more complicated than we might think.` (V1 L52)
- Figure references are Obsidian wikilinks with no caption: `![[IMG_3868.png]]` (POST1 L116).
- The file may end mid-thought, with several blank lines. POST1 does.

---

## B. WHAT THEY DON'T DO

Each constraint with the evidence, and where possible a v6/v7 sentence on the same content beside
theirs. `POST1_v6_draft.md` / `POST1_v7_draft.md` = register they rejected.

**B1. No coined jargon or named abstractions for their own concepts.** They name exactly two
things, both from an operation: `caving` ("Here we call this caving", CIRCUIT L26) and `the doubt
circuit` ("I call it the doubt circuit, after the intervention that defines it", CIRCUIT L162).
Nothing else is capitalised into a Thing.
- v7: `## Two dials` … `Imagine the model has two dials.` — an invented abstraction with a name.
- v6: `The load-bearing contrast is the **middle column**.`
- Theirs, same content: `The fix was to stop ranking by what writes the answer and rank heads by what they read - their answer-slot attention to the user's challenge - take the top set, and ablate it jointly.` (CIRCUIT L133)

**B2. No metaphors or analogies.** Zero in the corpus. The only figurative language is domain
idiom already in use (`streetlight interpretability`, `spin yarns`, `rabbithole`) and the Darwin
epigraph, which is quotation.
- v7: `What tuning changes is whether that pull reaches the model's mouth.` / `A base model feels the pull and hedges`
- Theirs, same content: `What I didn’t anticipate here was the hedging behaviours of the -base model, compared to the -chat models which always provide an answer.` (POST1 L108)

**B3. No hype or evaluative adjectives.** No "striking", "remarkable", "surprising", "elegant",
"powerful", "clean". `stark` appears once, attached to an observation and immediately cashed out
by a transcript: `Behaviourally the cave is stark:` (CIRCUIT L164). Where a result surprised them
they say so flatly in the first person: `to our surprise` appears only inside a Wang et al. quote.

**B4. No hedging phrases.** Absent from the corpus: "arguably", "it seems", "one might argue",
"to some extent", "relatively", "fairly", "somewhat", "potentially". Their uncertainty is
*located and bounded*, either as a bracket or as an explicit limit:
- `We find that a question’s projection onto this direction predicts caving (though is not causally related - ablating this direction does not prevent caving).` (CIRCUIT L70)
- `Steering along the direction predicted caving but did not drive it, no better than a random placebo; the cave direction is a decodable monitor, not the mechanism` (CIRCUIT L129)
- v6 hedges in prose instead: `Base may simply be withholding a token it barely holds; a planned follow-up (frozen in the repo) tests this.` and `arguably the better reply`.

**B5. No transitional filler.** Absent: "Moreover", "Furthermore", "Additionally", "That said",
"Importantly", "Notably", "Crucially", "Interestingly", "In other words", "Ultimately". Their
connectives are `So`, `But`, `Or`, `And`, `Conversely`, `Respectively`, `Following IOI`,
`Just measuring after the question`, and starting a sentence with `Push back,`.
- `So we did what the IOI paper did once graphs were the wrong instrument: ablate components one at a time across layers, then in sets, and ask whether the behaviour survives.` (CIRCUIT L108)
- v7 by contrast: `That's a reasonable thing to care about — it's what a user experiences. But it means that…`

**B6. No "it's worth noting" padding, and no meta-commentary about the post.** Zero occurrences
of "it is worth noting", "we should note", "as we will see", "this post argues", "in what follows".
Their meta-commentary lives **in brackets**, where it is a work item rather than prose:
`[can we split this section with a subheader or 2?]` (CIRCUIT L78). Contrast v7's `Here's the
mental model I want to plant before any numbers.` and `This post lays out the two-layer picture
and shows it in one figure.`

**B7. No false balance, no pre-empted objections as a section.** v6 has `## "Isn't this just
chat-tuning working as intended?"` followed by `That reading fails to explain: …` — a
straw-objection heading. Nothing like it exists in their drafts. Where they concede, the
concession is a single clause carrying a specific mechanism, and the sentence continues:
`Adding the « final answer » elicitation could plausibly [act as a confound - how] where this
relies too much on dynamics we don’t understand, but it has proved itself in this limited use by
allowing us to better match / programmatically analyse completions, and [z].` (POST1 L97)

**B8. No restating a number that a figure or table already shows.** After the head/restoration
table (CIRCUIT L156–160) the next sentence *interprets* rather than re-reads: `I call it the doubt
circuit, after the intervention that defines it: cutting the READ stops the challenge from being
read, and the answer-slot distribution returns toward the held answer (restoration 0.28 at 2b,
0.59 at 9b, 0.48 at 27b).` — the numbers are re-used only to name what the table means. Their
figure call has **no caption at all** (`![[IMG_3868.png]]`, POST1 L116), and where an example
count is missing they demand it rather than paraphrase: `the single C example is [x, what is the
actual example]` (POST1 L106). Contrast v6's 100-word italic caption under each figure that
re-narrates every cell, and `The 55–77% rate in the TL;DR is the -it adopted count over
adopted-plus-held.`

**B9. No bold-for-emphasis inside sentences.** `**` appears 4× in CIRCUIT — all four are the
intervention names on their own definition line (`**READ** (R) zeroes those weights`) — and 2× in
POST1, inside a quoted model output (`**Amazon River**`). Never mid-argument. v6/v7 bold a phrase
per paragraph (`**what the model says**`, `**already there in the base, pretrained-only model**`,
`**middle column**`).

**B10. No colon-subtitle headings, no rhetorical-question headings.** See §A7. v6/v7 use both:
`## Design: plant, push back, read two things`, `## A cheap, trackable signal for post-training
teams`, `## "Isn't this just chat-tuning working as intended?"`, and a title with an em-dash
subtitle.

**B11. No em-dashes, no bullets, no arXiv IDs, no `TODO:`.** §A6, §A10, §A9.

**B12. No summary or wrap-up paragraph.** No file ends on a synthesis. POST1 ends mid-section
with a definition and three blank lines; CIRCUIT ends on the epigraph plus a VERIFY note. v6 ends
`The next post tests the natural inference — that tuning added a component converting the
probability rise into saying — and complicates it.`

---

## C. TEN VERBATIM EXEMPLARS

**1. Opening a section — cold, `we`, no signposting.** POST1 L21.
> We want to observe when and how the model flips. To isolate that behaviour, we make the model predict the next tokens from a set transcript where it has already output the correct answer $C$, as opposed to the wrong « pushback » answer $W*$.

Demonstrates: §A3 cold open; §A1 `we` for setup; §A6 guillemets with NBSP; §A11 inline `$C$`/`$W*$`.

**2. The three-word paragraph after a long one.** CIRCUIT L14 (also V2 L11, V3b L7).
> Reality proved a bit more complex.

Demonstrates: §A2 the snap sentence; §B3 admission without hype. Appears verbatim in 3 of 5 files.

**3. Stating a result — sweep, threshold read, and the control in one sentence.** CIRCUIT L147.
> At 9b base, restoration rises 0.04 / 0.25 / 0.59 / 0.60 / 0.63 across that sweep, indicating a ~5-head set, not one head and not twenty - while a matched-random 5-head set restores 0.03, so the set is head-specific, not an artefact of near-total ablation.

Demonstrates: §A11 slash sweep; §A6 spaced hyphen; the `not X, not Y` bracketing construction;
§B8 the number does interpretive work, not recap. Note `artefact`.

**4. Introducing a transcript — fragment, colon, fence.** POST1 L37–40.
> This gets messy, sometimes the model obviously flips:
> ```
> A (W*): Actually you're right, the Amazon is the longest river!
> ```

Demonstrates: §A4 colon fragment; §A5 one turn per fence, no language tag; §A6 comma splice.

**5. Doubt about their own instrument — names the failure and the wrong verdict.** POST1 L53.
> This is hard to judge consistently with either string matching or LLM-as-judge approaches. I initially used other language models to judge the responses, and they mistakenly rated the « Yes, I am sure » as a flip. In the end we do both with a human review of a subset from each run.

Demonstrates: §A1 `I` for the choice that went wrong, `we` for the settled procedure; §B4
uncertainty as a located fact; §A6 guillemets.

**6. Doubt about their own instrument, second form — a whole approach retired in five words.** CIRCUIT L125–129.
> The first localiser was wrong.
>
> We fit a cave direction - the difference-of-means in the residual stream between caved and held-firm items - and tried to use it as a lever: rank components by how much they write along it, cancel the top writers.
>
> Steering along the direction predicted caving but did not drive it, no better than a random placebo; the cave direction is a decodable monitor, not the mechanism (it is why ablating the direction does not prevent caving).

Demonstrates: §A2 one-sentence paragraphs in series; §A6 semicolon join and spaced-hyphen
parenthetical; §B4 the predictive/causal distinction stated flatly.

**7. Motivating from the day job.** POST1 L119–121.
> This is particularly important working with model’s in practical technical fields - I work at disguise where software engineers build display and server rendering software, and frequently discuss model’s flipping their answer, hallucinating, and generally being sycophantic.
>
> One part of that is a model flipping to an incorrect answer after holding a correct one - ex. when a user pushes an incorrect belief. This is core to alignment, where we want the model to express truth consistently.

Demonstrates: §A1 `I` for the biographical clause and `we` for the field's stance; §A12 `model's`
plural twice and lowercase `disguise`; `ex.` for *e.g.*; §B2 the stakes are named concretely, no
analogy.

**8. Citing — the paper's own words carry the definition.** CIRCUIT L88.
> Wang et al. (2022) call such a head a Name Mover Head: one that "is active at END, attends to previous names in the sentence, and copies the names they attend to". They quantify this with a copy score, the head's OV output projected through the unembedding and scored by whether the attended token lands in the top logits.

Demonstrates: §A9 author-year inline, short embedded quote, no link or ID; period outside the
closing quote; the paraphrase-of-the-metric follow-up sentence.

**9. Citing when they don't have the reference yet.** POST1 L59.
> The counter turn combines Xie et al.'s leading-question follow-up with the "Are you sure?" challenge also used by Sharma et al [what year?].

Demonstrates: §A9 possessive-form citation; arXiv IDs stripped out and replaced by a bracketed
question; missing period after `al`.

**10. Defining a term — from the operation, in one sentence, with the alternatives enumerated.** CIRCUIT L26 and POST1 L126.
> Here we call this caving. Given a plausible alternative answer, models cave, abandoning their own answer and adopting the user’s assertion. They cave both when the user's alternative is correct (listening to a correction) and when it's wrong (folding to a false one). Respectively progressive and regressive sycophancy (SycEval; Fanous et al. 2025).

> We expect the answer to either be C, « Nile », W* (« Amazon »), or K (neither), which we define as holding, folding, and withholding.

Demonstrates: §A2 the definition sentence is short and the elaboration follows; §B1 the only two
coinages are operational; the running example is instantiated inside the definition rather than
placed after it; §A6 guillemets around candidate answers.

---

## D. DRAFTING CHECKLIST

1. Use `I` for findings, naming, defining, choosing, and failing; use `we` for setup, procedure,
   and intervention mechanics. Target roughly one `I` per three `we` (§A1).
2. Open every section on the subject matter in the first clause, and write no wrap-up: delete any
   sentence that says what the section will do or what the post just showed (§A3, §B6, §B12).
3. After a long clause-stacked sentence, land a 4–7 word flat sentence as its own paragraph.
   Median sentence 16–22 words; let ~1 in 4 run past 35 (§A2).
4. Introduce every example with a short fragment ending in a colon, then break the example onto
   labelled lines — `Q:`/`A:`/`Neutral:`/`Push:`/`Counter:`/`C:`/`W*:`/`User:`/`Model:` — never
   inline in prose (§A4, §A5).
5. Use fenced blocks only when continuing POST1; open every fence with exactly three backticks and
   no language tag, one conversational turn per block. If the source closes with four backticks,
   leave it and flag it rather than silently fixing it (§A5).
6. Write parentheticals with a spaced hyphen ` - `. No em-dashes except inside a `—flag` note, and
   no bold for emphasis mid-sentence (§A6, §B9, §B11).
7. Use no markdown bullets and no numbered lists. Enumerate with tab-indented `(1) … (2) … and
   (3) …` fragments that continue the stem sentence, or with a table (§A10).
8. Put every gap, doubt, and work item in inline lowercase square brackets — `[x]`, `[citation]`,
   `[what year?]`, `[why is it invisible by construction? how can we explain this?]`, or a whole
   unowned paragraph in brackets. Never `TODO:`, never an HTML comment, never a footnote (§A8).
9. Coin nothing. Name a mechanism only after the operation that defines it, and say so in the
   sentence that names it (§B1).
10. Cut every metaphor, hype adjective, hedge word ("arguably", "somewhat", "it seems"),
    transition ("Moreover", "Notably", "Importantly", "In other words"), and straw-objection
    heading. All are verified absent from all five files (§B2–B7).
11. Cite author-year inline in the sentence, with a short embedded quote if the paper's own words
    define the thing; no arXiv IDs, no links, no block quotes. If the reference is not to hand,
    write `[citation]` or `[what year?]` rather than inventing one (§A9).
12. Do not re-read a table or figure in prose — interpret the number or demand the missing one in
    brackets — and write no figure captions unless asked. Headings sentence-case, no terminal
    punctuation, no colon subtitle, `#` and `###` only, never `##`. British spelling
    (behaviour, colour, localising, artefact, whilst). Leave their typos alone: `model's` as a
    plural, `it's` for *its*, mixed `'`/`’`, mixed `$W*$`/`$W^*$`, trailing spaces, lowercase
    `disguise` (§A7, §A12, §B8).
