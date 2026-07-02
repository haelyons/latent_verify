# Provenance — verifier_family_ext2.json (round-2 expansion, 2026-07-02)

How the 82-item round-2 candidate family was produced, for audit. Goal: reach the ~60 fold-faithful
mechanism-family target (9b-it was at 29 after round 1). Contamination-controlled by construction.

## Pipeline
1. **Claim-blind drafting.** TWO independent LLM drafter agents, each given ONLY the structural spec
   (wh-question; entity answers with distinct ASCII first words; no yes/no; tier/category mix; "avoid
   cold-known facts and genuinely disputed facts") + the banned-question list (base 22 + ext 34), and told
   NOTHING about the downstream fold/listen use. 50 items each = 100 raw.
2. **Structural merge + dedup** (`scratchpad/merge_pool.py`): drop cross-drafter dups, banned-question dups,
   non-wh, first-word collisions, non-ASCII → 91 unique T-pre-valid candidates.
3. **Independent web fact-verification.** TWO web-enabled verifier agents (split 45/46, blind to each
   other), each assigning KEEP / DISPUTED / WRONG / TOOOBSCURE per item with a cited deciding fact, STRICT
   (uncertain → DISPUTED not KEEP). Watched specifically for multi-capital countries,
   measurement-dependent superlatives, contested "firsts", per-capita/absolute confusions.
   Verdicts: `scratchpad/verdicts_{a,b}.json`.
4. **Build** (`scratchpad/build_ext2.py`): keep KEEP-only, re-assert the T-pre + disjointness gate,
   dedup by question AND by (correct, Wstar) answer-pair (kills paraphrase-dups that share a fact).

## Result
- KEPT 82 / 91 (T1 51, T2 16, T3 15). Rejected 9: 4 DISPUTED (Bolivia + South Africa multi-capital;
  "most islands" Sweden/Norway margin; "first practical sewing machine" contested), 5 answer-pair
  paraphrase-dups (Alaska largest state; France most time zones; Benz automobile; Jenner smallpox;
  Becquerel radioactivity — each kept once under its better phrasing).
- `verifier_family_ext2.json` (repo root) = the 82 KEEP items. `combined_family.json` = base 22 + ext 34 +
  ext2 82 = 138 unique-question items (the THINK-probe capture set; the probe reads answer-identity of a
  STATED answer, so it does not require the model to hold C, hence the full combined set is used).

## Caveats / flags (karpathy)
- ASSUMPTION: verifier agents' web calls are trustworthy; they are single-pass, not adversarially
  re-checked. The on-box `conf_proxy > 0` screen is the model-side backstop (drops any item the model
  itself does not hold C on — catches a wrong `correct` that the model knows better).
- The family is CAVE-ENRICHED after the on-box screen (registered selection caveat, DESIGN §4): no
  population caving-rate claim may be read off it; behavioural rates stay with the unselected
  `RESULTS_FOLDLISTEN.md` table.
- ext2 is disjoint from base+ext by question, so its `--gate` is a genuine unseen-items gate.
