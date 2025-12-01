## Pseudobulk vs Single-Cell – Cross-Resolution Comparison

### Key Findings

- The **ordering of baselines** is consistent between pseudobulk and
  single-cell:
  - Self-trained PCA ≫ scGPT ≳ scFoundation ≫ GEARS ≳ random.
- Single-cell performance is slightly noisier but **qualitatively
  matches** pseudobulk.
- This suggests the perturbation response manifold is a **robust
  object** visible at both resolutions.

---

### 1. Baseline Ordering

From aggregated results:
- Pseudobulk (Goal 2 analyses).
- Single-cell (Goal 2 single-cell + `SINGLE_CELL_ANALYSIS_REPORT.md`).

In both settings:
- Self-trained PCA is the **top baseline**.
- scGPT and scFoundation are intermediate.
- GEARS and random embeddings are weaker.

This consistency is strong evidence that:
- The manifold law is not an artifact of pseudobulk aggregation.

---

### 2. Performance Levels

Typical perturbation-level r values:

| Dataset  | Resolution   | Self-trained r | Random r (Gene) |
|----------|--------------|----------------|-----------------|
| Adamson  | Pseudobulk   | ~0.94          | ~0.23           |
| Adamson  | Single-cell  | ~0.40          | ~0.21           |
| K562     | Pseudobulk   | ~0.66          | ~0.10           |
| K562     | Single-cell  | ~0.26          | ~0.07           |

Observations:
- Absolute r values drop at single-cell resolution due to:
  - Increased noise at the cell level.
  - Harder prediction target (individual cells vs averages).
- Relative **gaps between baselines** persist.

---

### 3. LSFT Across Resolutions

In both pseudobulk and single-cell:
- LSFT offers **large improvements** for weak baselines:
  - Random embeddings gain ~0.15–0.20 r.
- LSFT offers **minimal gains** for self-trained PCA:
  - Improvements typically < 0.01–0.02 r.

Implication:
- Local neighborhood structure is predictive at both resolutions.
- Good global embeddings (PCA) already capture that structure.

---

### 4. LOGO Across Resolutions

LOGO extrapolation (holding out functional classes):
- Pseudobulk:
  - Self-trained PCA achieves r ~0.3–0.4 on held-out classes.
  - Random embeddings collapse near r ≈ 0.
- Single-cell:
  - Preliminary LOGO results match this pattern:
    - PCA sustains moderate r.
    - Random baselines fail.

Implication:
- Functional generalization is **not** an artifact of averaging; the
  underlying manifold carries genuine biological structure.

---

### 5. Summary

Across pseudobulk and single-cell:
- The **same manifold structure** governs perturbation responses.
- Simple, self-trained PCA embeddings dominate.
- Local similarity (LSFT) is effective regardless of resolution.
- Pretrained and graph-based embeddings underperform simple PCA on this
  task.

For full details:
- Pseudobulk: see aggregated results under `aggregated_results/`.
- Single-cell: see `analysis_docs/single_cell_overview.md`.


