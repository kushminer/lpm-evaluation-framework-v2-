## Single-Cell LSFT Results

### Key Findings

- LSFT provides **large gains** for weak baselines (random, pretrained),
  but only **small gains** for self-trained PCA.
- This supports the view that **local geometry** in expression space
  is highly informative, even when global embeddings are poor.

Numbers below are adapted from:
- `SINGLE_CELL_METHODOLOGY_REPORT.md` (LSFT section)
- LSFT single-cell CSV outputs (when available)

---

### 1. Adamson

#### 1.1 LSFT Improvements (Illustrative)

| Baseline              | Top % | Baseline r | LSFT r | Δr     |
|-----------------------|-------|------------|--------|--------|
| Self-trained PCA      | 5%    | 0.396      | 0.399  | +0.003 |
| Self-trained PCA      | 10%   | 0.396      | 0.396  | +0.000 |
| scFoundation Gene Emb | 5%    | 0.257      | 0.381  | +0.124 |
| scFoundation Gene Emb | 10%   | 0.257      | 0.379  | +0.122 |
| scGPT Gene Emb        | 5%    | 0.312      | 0.389  | +0.077 |
| scGPT Gene Emb        | 10%   | 0.312      | 0.385  | +0.074 |
| Random Gene Emb       | 5%    | 0.205      | 0.384  | +0.179 |
| Random Gene Emb       | 10%   | 0.205      | 0.377  | +0.172 |

#### 1.2 Interpretation

- **Random Gene Embeddings** gain ~0.17–0.18 r, becoming competitive with
  self-trained PCA at small neighborhoods.
- **Pretrained embeddings** (scGPT, scFoundation) also see substantial
  gains (0.07–0.12 r).
- **Self-trained PCA** barely improves:
  - Already has good global geometry.
  - Local filtering adds only marginal benefit.

---

### 2. K562

For K562, LSFT is more challenging due to:
- Larger number of perturbations.
- Sparser GO coverage for GEARS.

Illustrative patterns:
- Self-trained PCA gains only small amounts from LSFT.
- Random/pretrained baselines can gain, but the absolute r values
  remain lower than Adamson due to task difficulty.

---

### 3. Summary Across Baselines

**Who benefits most from LSFT?**
- Random and pretrained embeddings benefit the most:
  - LSFT effectively recovers local neighborhoods that capture
    perturbation response similarity.
- Self-trained PCA benefits least:
  - PCA already positions similar perturbations/cells near each other
    in embedding space.

**Implications:**
- The **local manifold structure** is real and powerful:
  - Even when global embeddings are poor, nearest neighbors still
    contain strong predictive signal.
- **Global embedding quality** determines how much LSFT helps:
  - The worse the baseline, the more LSFT can rescue it.

For methodology, see:
- `methodology/lsft_single_cell.md`.


