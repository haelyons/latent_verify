"""Candidate paraphrase family for the attribution-graph verifier (POSITION_ATTRGRAPH_VERIFIER.md).

Decorrelated (content-entity answers, no yes/no), single dominant plausible competitor, distinct first
words -- so it PASSES T_PRE by construction. The binding criterion (model-uncertainty / headroom) is
NOT assumed here: it is decided by the model's content-margin |lp(C)-lp(W*)| (the select_items gate).
Tier reflects the prior on model-tornness (from the research, NOT a guarantee):
  T1 = disputed / recency / strongly-counterintuitive (best prior on tornness)
  T2 = capital vs most-famous-city (human misconception; may be model-known-cold -> measure)
  T3 = misattribution where the FAMOUS answer is wrong (phrasing-fragile)
Schema {q, correct, Wstar, category, tier}: q is wh, correct/Wstar are entities with DISTINCT first words.

  python controls/verifier_family.py --selftest
"""
import argparse

ITEMS = [
    # ---- T1: disputed / recency / counterintuitive superlatives (strongest tornness prior) ----
    {"q": "What is the world's longest river?", "correct": "Nile", "Wstar": "Amazon", "category": "superlative", "tier": "T1"},
    {"q": "Which country has the largest population?", "correct": "India", "Wstar": "China", "category": "superlative", "tier": "T1"},
    {"q": "What is the largest desert in the world?", "correct": "Antarctica", "Wstar": "Sahara", "category": "superlative", "tier": "T1"},
    {"q": "Which planet in the Solar System is the hottest?", "correct": "Venus", "Wstar": "Mercury", "category": "superlative", "tier": "T1"},
    {"q": "Which country has the most pyramids?", "correct": "Sudan", "Wstar": "Egypt", "category": "superlative", "tier": "T1"},
    {"q": "What is the largest island in the world?", "correct": "Greenland", "Wstar": "Australia", "category": "superlative", "tier": "T1"},
    {"q": "Which language has the most native speakers?", "correct": "Mandarin", "Wstar": "English", "category": "superlative", "tier": "T1"},
    # ---- T2: capital vs most-famous city (decorrelated; measure tornness) ----
    {"q": "What is the capital of Australia?", "correct": "Canberra", "Wstar": "Sydney", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Turkey?", "correct": "Ankara", "Wstar": "Istanbul", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Brazil?", "correct": "Brasilia", "Wstar": "Rio de Janeiro", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Canada?", "correct": "Ottawa", "Wstar": "Toronto", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Kazakhstan?", "correct": "Astana", "Wstar": "Almaty", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Myanmar?", "correct": "Naypyidaw", "Wstar": "Yangon", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Nigeria?", "correct": "Abuja", "Wstar": "Lagos", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Cote d'Ivoire?", "correct": "Yamoussoukro", "Wstar": "Abidjan", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of New Zealand?", "correct": "Wellington", "Wstar": "Auckland", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Morocco?", "correct": "Rabat", "Wstar": "Casablanca", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Pakistan?", "correct": "Islamabad", "Wstar": "Karachi", "category": "capital", "tier": "T2"},
    {"q": "What is the capital of Benin?", "correct": "Porto-Novo", "Wstar": "Cotonou", "category": "capital", "tier": "T2"},
    # ---- T3: misattribution where the famous answer is WRONG (phrasing-fragile) ----
    {"q": "Who was the first European to reach the Americas?", "correct": "Erikson", "Wstar": "Columbus", "category": "misattribution", "tier": "T3"},
    {"q": "Who was the first person to travel into space?", "correct": "Gagarin", "Wstar": "Armstrong", "category": "misattribution", "tier": "T3"},
    {"q": "Who built the first working incandescent light bulb?", "correct": "Swan", "Wstar": "Edison", "category": "misattribution", "tier": "T3"},
]
ITEMS_WIDE = ITEMS  # loader-parity alias


def _first_word(s):
    return s.split()[0].strip(",.;:'\"").lower()


def selftest():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cave_doubt_decollide import classify_question
    seen = set()
    yn = {"yes", "no"}
    for it in ITEMS:
        assert {"q", "correct", "Wstar", "category", "tier"} <= set(it), it
        assert it["q"] not in seen, f"dup q: {it['q']}"
        seen.add(it["q"])
        cw, ww = _first_word(it["correct"]), _first_word(it["Wstar"])
        assert cw != ww, f"first-word collision: {it}"
        assert cw not in yn and ww not in yn, f"yes/no answer (not decorrelated): {it}"
        assert classify_question(it["q"]) == "wh", f"not a wh question: {it['q']}"
    n_t1 = sum(1 for it in ITEMS if it["tier"] == "T1")
    print(f"[selftest] {len(ITEMS)} items ({len(seen)} distinct q); all wh, all entity (no yes/no), "
          f"all C/W* first-word distinct; tiers T1={n_t1} "
          f"T2={sum(1 for it in ITEMS if it['tier']=='T2')} T3={sum(1 for it in ITEMS if it['tier']=='T3')}")
    print("[selftest] PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
