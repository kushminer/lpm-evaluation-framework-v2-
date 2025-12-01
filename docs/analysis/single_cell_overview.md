## Single-Cell Analysis Overview

### Key Findings

- **Self-trained PCA dominates** single-cell baselines across datasets.
- **GEARS embeddings now diverge** from self-trained PCA after the path/validation fix.
- **Pretrained models (scGPT, scFoundation)** perform only modestly better than random.
- **Single-cell and pseudobulk results agree** on the ordering of baselines.

All numbers below come from:
- `results/single_cell_analysis/comparison/SINGLE_CELL_ANALYSIS_REPORT.md`
- `COMPREHENSIVE_SINGLE_CELL_REPORT.md`

---

### 1. Baseline Performance (Single-Cell)

#### 1.1 Per-Dataset Summary

| Dataset  | Baseline              | Perturbation r | L2     |
|----------|-----------------------|----------------|--------|
| Adamson  | Self-trained PCA      | 0.396          | 21.71  |
|          | scGPT Gene Emb        | 0.312          | 22.40  |
|          | scFoundation Gene Emb | 0.257          | 22.87  |
|          | GEARS Pert Emb        | 0.207          | 23.35  |
|          | Random Gene Emb       | 0.205          | 23.24  |
|          | Random Pert Emb       | 0.204          | 23.24  |
| K562     | Self-trained PCA      | 0.262          | 28.25  |
|          | scGPT Gene Emb        | 0.194          | 28.62  |
|          | scFoundation Gene Emb | 0.115          | 29.12  |
|          | GEARS Pert Emb        | 0.086          | 29.30  |
|          | Random Pert Emb       | 0.074          | 29.35  |
|          | Random Gene Emb       | 0.074          | 29.34  |
| RPE1     | GEARS Pert Emb        | 0.203          | 28.88  |

> Note: For RPE1, only GEARS is currently run in the single-cell pipeline.

#### 1.2 Average Performance Across Datasets

Average perturbation-level r:

- **Self-trained PCA**: 0.329
- **scGPT Gene Emb**: 0.253
- **scFoundation Gene Emb**: 0.186
- **GEARS Pert Emb**: 0.165
- **Random Gene Emb**: 0.139
- **Random Pert Emb**: 0.139

---

### 2. GEARS vs Self-Trained PCA (Single-Cell)

After fixing the GEARS CSV path and adding embedding validation:

- **Adamson**:
  - Self-trained: r = 0.396
  - GEARS: r = 0.207
  - Δr = −0.189 (GEARS worse)
- **K562**:
  - Self-trained: r = 0.262
  - GEARS: r = 0.086
  - Δr = −0.176 (GEARS worse)

Interpretation:
- GEARS captures GO graph structure but does **not** align as well with
  perturbation response manifolds as self-trained PCA.
- GEARS still provides a meaningful alternative geometry for comparison
  (no longer identical to PCA).

For a deeper dive, see:
- `analysis_docs/gears_comparison.md`

---

### 3. Single-Cell vs Pseudobulk

Even though the single-cell pipeline is noisier (cell-to-cell variance),
the **relative ordering** of baselines matches pseudobulk:

- Self-trained PCA ≫ scGPT ≳ scFoundation ≫ random.
- GEARS underperforms self-trained PCA but offers a distinct geometry.

Implications:
- The **Manifold Law** holds at both pseudobulk and single-cell
  resolution.
- Aggregation to pseudobulk does not destroy the key structure used for
  prediction.

For detailed cross-resolution analysis, see:
- `analysis_docs/cross_resolution_pseudobulk_vs_single_cell.md`


