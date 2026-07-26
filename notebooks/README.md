# notebooks/

## `inspect_fold_completions.ipynb`

Read the fold-experiment completions yourself, item by item, and watch the headline numbers
get recomputed from the raw text.

**It needs no GPU, loads no model, and makes no network calls.** Every generation it shows
was produced on an A100 at greedy decoding and committed to JSON in this repo; the notebook
only reads those files and prints them. It runs in about two seconds on a laptop.

What it does, in order:

1. Loads the six ext2 cells (2b / 9b / 27b, base and -it) and asserts n=82 per cell, item-set
   identity against `verifier_family_ext2.json`, that each cell plants and pushes what it
   claims, and that the elicited final really was conditioned on the model's own free reply.
2. States the one piece of text surgery — the `\nQ:` cut that removes base-model runaway
   self-dialogue — shows the regex, counts how often it fires per file, and prints a full
   untruncated generation next to the truncated one. `SHOW_RAW = True` does the same inside
   every transcript.
3. Prints a seeded random sample of items as full four-turn transcripts, 9b-base and 9b-it
   on the same item, with the label assigned to each generated turn. Change `SEED` and
   re-run for a different sample; change `MODELS` to compare any two of the six cells.
4. Recomputes the headline counts for all six cells from the stored text, plus a column for
   how many free replies *literally* name an answer as opposed to being labelled via the
   confidence rule.
5. `browse()` — filter to a category and page through it, e.g. every item where 9b-it folded,
   every reply naming both answers, every base reply naming nothing.

All labelling goes through `controls/faithful_rescore.classify`. Nothing here reimplements
entity matching, and no count is read out of a summary block.

### Running it

```
pip install notebook          # or jupyterlab
jupyter notebook notebooks/inspect_fold_completions.ipynb
```

Outputs are committed, so it is readable on GitHub without being run.
