# PATCH_intro_syceval — intro L20, the `[?]` closed against the paper

Target: `/home/hal/Documents/Remote/interp/DARWIN.md_post1_user_intro.md` (READ ONLY — do not write to
the vault). Live state re-verified immediately before writing: md5 `74533ee96ac2795bf6ebd6ceeaea3918`,
29 lines, 5854 B. That is **not** the state `PATCH_intro.md` was built against (`dcb8db8e…`, 27 lines,
5596 B) — the researcher has edited the file since — but the SycEval sentence is still on **L20** and
the anchor below is byte-identical to the one `PATCH_intro.md` §3.3 recorded, checked programmatically.

Invisible characters: every gap around the markdown link and around `_regressive_` / `_progressive_` is
a NO-BREAK SPACE (U+00A0). The anchor and the fill below both carry them; a careless copy will lose them.

This block **supersedes `PATCH_intro.md` §3.3**, whose fill was a holding bracket and whose RESIDUAL (2)
and (3) are exactly what this patch discharges. It also reverses that block's decision (b), which
deleted the DOI link: see the note under the fill.

---

### §3.3b — intro L20, the SycEval asymmetry

ANCHOR (verbatim; the sentence sits mid-line, after `This behavioural pattern has been studied extensively. ` and before ` In the sankey, we can see`):

```
What we call folding and listening is what [SycEval](https://doi.org/10.1609/aies.v8i1.36598) calls _regressive_ and _progressive_ sycophancy, and they also find that -chat models [?] revise toward truth more readily than toward falsehood.
```

FILL:

```
What we call folding and listening is what [SycEval](https://doi.org/10.1609/aies.v8i1.36598) calls _regressive_ and _progressive_ sycophancy, and they also find that -chat models - their three are ChatGPT-4o, Claude-Sonnet and Gemini-1.5-Pro, with no base arm - revise toward truth more readily than toward falsehood, 43.52% progressive against 14.66% regressive (Fanous et al. 2025) [carried by their maths set; on medical advice it reverses for Claude-Sonnet]. They push toward truth only where a model was first wrong and toward falsehood only where it was first right, so their two rates are counted over different items - we send the same 82 pairs through both directions.
```

Four things happen and each is deliberate. **(a)** Their claim is supported, not corrected — SycEval
does report the asymmetry as an explicit pair of numbers, so the `[?]` closes with the number rather
than with a demand. **(b)** The `[?]` sits between `-chat models` and `revise`, which is where a scope
qualifier belongs, so the answer to the scope question lands in that slot as a spaced-hyphen aside in
their own punctuation, and the rate lands after the claim it quantifies. **(c)** The one place the
asymmetry does not hold gets a bracket, not a silent omission; the sentence reads across it. **(d)**
The `doi.org/10.1609/aies.v8i1.36598` link **stays**. `PATCH_intro.md` §3.3 deleted it on the stated
ground that it was "absent from the ledger" and could not be certified. It is now certified — the DOI
302-redirects to `ojs.aaai.org/index.php/AIES/article/view/36598`, which is this paper's AIES
camera-ready — and the same block's own §S note says the document's five other verified links "are not
touched". Deleting only this one would single it out. Nothing I write adds a link or an identifier.

EVIDENCE:
  - `CITATIONS_post1_verified.md` :: SycEval entry appended this session (2026-07-28) :: DOI resolution,
    venue, the two rates, the per-model and per-dataset breakdown, and the denominator confound. All of
    it fetched from the paper, not from a repo note.
  - Fanous et al. 2025, Results, §"Sycophancy Rates Are High Across Models" :: "Our experiments showed
    that 58.19% of all samples exhibited sycophantic behavior, with progressive responses and regressive
    responses occurring at 43.52% and 14.66%, respectively." :: the number in the fill. Identical in the
    arXiv v4 HTML and in the AIES PDF (pp. 893–900).
  - Same section :: Gemini 53.22% / 9.25%, ChatGPT 42.32% / 14.40%, Claude-Sonnet 39.13% / 18.31% ::
    progressive exceeds regressive for **all three** models, which is what licenses the unqualified
    `-chat models` as the subject.
  - Fanous et al. 2025, Table 3 (MEDQuad) :: Claude in-context 302 prog / 383 regr, preemptive 275 / 375
    :: the reversal named in the bracket. Table 2 (AMPS Math) has the gap the other way for every cell
    (e.g. ChatGPT in-context 899 / 38), which is why the fill says "carried by their maths set".
  - Fanous et al. 2025, Methods, "Step 1" :: "We evaluate 3 models: ChatGPT-4o-(2024-05-13) … Claude-
    Sonnet and Gemini-1.5-Pro, both accessed through VertexAI, under default calibration settings." ::
    the model list in the aside; no pretrained checkpoint appears anywhere in the paper, hence "no base
    arm".
  - Fanous et al. 2025, Methods, "Step 2" :: "If the initial inquiry response was correct, we present
    evidence justifying an incorrect answer … If the initial inquiry response was incorrect, we present
    evidence justifying the correct answer." :: the second sentence of the fill. Both rates are then
    divided by the same 15,345 non-erroneous responses, and the initial correct/incorrect split is never
    reported, so neither rate is a per-opportunity rate.
  - `docs/drafts/figs/make_fig_outcome_alluvial.py` :: the six pinned ext2 cells, sourced to
    `foldlisten_judge_fl_*_ext2_summary.json` :: `-it` listen names C on 81 / 82 / 82 of 82 at 2b/9b/27b
    against fold naming W\* on 68 / 55 / 55 — the same-82-pairs design the fill's last clause asserts,
    and the researcher's own asymmetry, already carried by the L12 figure and by L17–L18.
  - `BRIEF_fill_agents.md` :: "author-year, parenthetical or narrative"; "no em-dashes … a spaced
    hyphen"; "inline lowercase square brackets" :: the form of the fill.

CRITERIA:
  F — 43.52 / 14.66, the three per-model pairs, and the MEDQuad counts all trace to named sections and
    tables of the paper, verified in both the arXiv v4 HTML and the AIES PDF.
  M — the intro states the asymmetry from their own data at L17–L18 and shows it at L12; the fill adds
    only what is external to that, which is the published rate and its limits.
  P — the aside answers the scope question, the number closes the `[?]`, the bracket names the one
    reversal, the last sentence is the reason the citation is a parallel and not a replacement for their
    own result. Removing any of the four loses something.
  1P — the fill's own claim about this post's design bottoms out in the ext2 summary JSONs via the
    figure script's pinned cells, not in a draft; the SycEval half bottoms out in the paper.
  R — spaced hyphens, no em-dash, British `maths`, lowercase in-flow bracket of 13 words, author-year
    parenthetical, no arXiv ID and no link added, U+00A0 run preserved byte-for-byte.
  C — SycEval now verified for the asymmetry as well as the vocabulary; their sentence stands, with the
    one condition bracketed rather than applied.
  S — L20's first cited sentence only. The rest of L20 (`In the sankey…`), and the five other links in
    the document, are untouched.

RESIDUAL:
  (1) `-chat` in this sentence now does double duty: it is the researcher's label for the Gemma 2
  instruction-tuned variants everywhere else in the post, and here it stands for three proprietary
  assistants. The aside makes the referents explicit rather than resolving the collision. If the
  researcher would rather keep `-chat` reserved for Gemma, the aside is the clause to rewrite.
  (2) SycEval's progressive and regressive rates share a denominator but not an opportunity set, and the
  paper never reports the initial correct/incorrect split, so a per-opportunity version of 43.52 / 14.66
  cannot be recovered from the published text. The fill states the asymmetry as they state it and names
  the counting difference; it does not attempt a corrected rate, and none should be invented.
  (3) `PATCH_intro.md` §3.3 must be marked superseded, or two patch files will propose different text
  for the same sentence.
