# AML Name Screening (Transliteration-Aware)

Streamlit UI · XGBoost (primary) · n8n email alerts · Fuzzy name matching

---

## Features

- Arabic ⇄ Latin **transliteration** (normalize accents, prefixes, spacing)
- **Fuzzy similarity**: Jaro-Winkler, Levenshtein (normalized), Soundex
- **Composite score** (weighted blend) for robust matching
- **Decision**: ALLOWED · REVIEW · BLOCKED + explanation
- **XGBoost risk %** for the matched record
- **Streamlit** front-end
- **n8n** workflow sends **email** for REVIEW/BLOCKED

---

## How matching works (short)

- **Transliteration:** canonicalize Arabic/Latin forms (e.g., *Al/Ibn/Abu*, diacritics, spaces)
- **Similarity (0–1):**
  - **Jaro-Winkler** (string agreement / ordering)
  - **Levenshtein** (edit distance → normalized similarity)
  - **Soundex** (phonetic equality 1/0)
- **Composite:** `0.5*JaroWinkler + 0.3*Levenshtein + 0.2*Soundex`

---

## Models

| Model              | Role       | Notes                                                      |
|--------------------|-----------:|------------------------------------------------------------|
| **XGBoost**        | **Primary**| Best generalization; class imbalance handled; early stop.  |
| Random Forest      | Benchmark  | Strong baseline; OOB score available.                      |
| Logistic Regression| Benchmark  | Simple linear baseline.                                    |



---



