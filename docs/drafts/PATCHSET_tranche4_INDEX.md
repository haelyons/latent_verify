# TRANCHE 4 — application index

Three files, 27 blocks, drafted 2026-07-30/31 by three isolated agents against committed ground
truth. This index is the only cross-file document; each file's own preamble governs its internals.

| file | blocks | doc | net deltas |
|---|---|---|---|
| `PATCHSET_tranche4_intro.md` | 6 (T4-I01…I06) | intro | **−13 words**, −5 brackets |
| `PATCHSET_tranche4_mech.md` | 7 (T4-M01…M06, M08) | notes | −4 brackets |
| `PATCHSET_tranche4_dist.md` | 14 (T4-D01…D14) | notes | −4 brackets (−3 if the L131 offer is declined) |

**Mechanically verified after all three landed** (re-run this before applying — the gold moves):

- Every CURRENT fence sliced from the live gold's bytes, **28/28 byte-exact and unique**
  (intro md5 `83a55a14`, notes md5 `71c3b3c5`).
- **Zero byte overlaps between any two of the 28 anchored spans**, within or across files — so no
  repeat of the notes-L319 triple collision.
- Zero em-dashes, en-dashes, NBSPs or curly apostrophes introduced in any PROPOSED text.
- `REDISTRIBUTE`, `0.875`, `0.751` appear in **no** PROPOSED text anywhere in the tranche (they occur
  only inside T4-I06's prohibition note). Required by `RETRACTIONS.md` R-12.
- Brackets: net **−13** across the tranche. Nothing added without a matched removal.

## Application order

Apply **descending by line number within each document**, so earlier edits never move later anchors.
The two notes files interleave — merge their block lists by line and work upward:

1. **notes** (one pass, descending): D01 (L335) · D02 (L312–313) · D03 (L310) · D04 (L309) ·
   D05 (L297) · D06 (L291) · D07 (L290) · D08 (L281) · **M02 (L279)** · **M04 (L276)** ·
   **M01 (L274)** · **M05 (L273)** · **M03 (L272)** · D09 (L196) · D10 (L176) · D11 (L131) ·
   D12 (L129) · **M06 (L200 — AFTER pending T3-14)** · D13 (L76) · D14 (L74).
2. **intro** (one pass, descending): I05 (L23) · I04b · I04a (L21) · I03b (L17) · I03a (L15) ·
   I02 (L9, a deletion — removes L9 and its blank L10) · I01 (L7).
3. **T4-M08** is unplaced by design: an offered passage, not an edit. Placement is the researcher's.

## Interactions with the 24 pending tranche-3 blocks

- **T4-M06 must follow T3-14** (both touch notes L200; M06 takes the causal sentence, T3-14 the
  number). M06 deliberately does not restate T3-14's count.
- **T4-D01 duplicates T3-10's Ankara-rank sentence** if both land. The clause to cut is named in
  D01's RESIDUAL.
- **T4-I06 writes no L25 block** — T3-03 holds that ground, and this session's circuit audit
  independently corroborated its replacement text (base fold∩listen 4/5, `-it` 5/5, `MOVE_UNMATCHED`
  at all four cells, write handles at floor 3/3). T3-03 remains the researcher's decision.
- **T4-I02's deletion of intro L9** removes the only operational definition of the grey band in the
  intro, which L15/L17/L23 lean on. It routes that to a **C02 re-slice** — C02's own anchor is stale
  (`PATCHMAP_live.md` §2.1) and must be re-cut before it can be applied at all.
- Untouched by this tranche and still standing: L5/L19/L25 (T3-01/02/03), notes L282/L284/L288/
  L293/L295/L319, D22, B05.

## Blocks needing the researcher, not a drafter

| block | the decision |
|---|---|
| T4-I05 (intro L23) | An **offer**, not a fill — the paragraph is filed as their own rewrite. |
| T4-M01 (notes L274) | The corrected sentence states a **null**; whether the post carries it is theirs. |
| T4-D11 (notes L131) | Two costed options, both written out. |
| T4-M08 | Placement of the offered circuit passage. |

## Two things that block clean application

1. **The Ankara PNG has no vault copy.** T4-D06 routes `fig_topk_ankara_9bbase.png` into the L291
   Fig-3b slot; the embed will not render until the file is copied into the vault. This is one of
   **five** pending image actions — the other four are the stale embeds in `COMPOSE_post1_brief.md`
   §B, including the intro's Fig 1, whose vault copy is md5-confirmed as the **anomalous 27b draw**.
2. **The gold moves.** The researcher edited both documents after `598de5e` without any ledger
   recording it, which is what made C02 unappliable. Re-verify the 28 anchors against current md5s
   at apply time; the check is a five-line script over the CURRENT fences.

## What the tranche does NOT cover

The `RETRACTIONS.md` R-12/R-13/R-14 addenda (`ADDENDA_20260730_ledger.md`) are written and **not
applied** — they patch repo ledgers, not the post. Notes L319/L321 survivorship, figure renumbering,
the lost head clause near L250, and the L60 speaker tag remain researcher-only and untouched.
