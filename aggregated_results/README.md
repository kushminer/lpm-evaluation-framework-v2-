# Aggregated Research Results

**Purpose:** This folder contains unified, engineer-friendly CSV files aggregating results from both **pseudobulk** and **single-cell** analyses for easy comparison and analysis.

**Last Updated:** 2025-11-25

---

## 📁 Contents

| File | Description | Status |
|------|-------------|--------|
| `baseline_performance_all_analyses.csv` | All baseline results (pseudobulk + single-cell) | ✅ Single-cell complete |
| `best_baseline_per_dataset.csv` | Winning baseline per dataset/analysis | ✅ Available |
| `lsft_improvement_summary.csv` | LSFT lift analysis | ✅ Single-cell available |
| `logo_generalization_all_analyses.csv` | LOGO extrapolation results | ✅ Single-cell available |
| `ENGINEER_ANALYSIS_GUIDE.md` | Detailed analysis guide | ✅ Complete |

---

## 📊 Current Data Status

### ✅ Single-Cell Analysis (Complete)
- **Baselines:** 18 results across 3 datasets (adamson, k562, rpe1)
- **LSFT:** 5 baseline comparisons showing improvement from local similarity filtering
- **LOGO:** 15 results for functional class holdout evaluation

### ⏳ Pseudobulk Analysis (In Progress)
- Results will be added once pseudobulk experiments complete
- Script automatically detects and includes pseudobulk CSVs when available

---

## 🚀 Quick Start for Engineers

### 1. Load the Data
```python
import pandas as pd

# All baseline results
baselines = pd.read_csv('baseline_performance_all_analyses.csv')

# Best baseline per dataset
best = pd.read_csv('best_baseline_per_dataset.csv')

# LSFT improvements
lsft = pd.read_csv('lsft_improvement_summary.csv')

# LOGO generalization
logo = pd.read_csv('logo_generalization_all_analyses.csv')
```

### 2. Key Analyses

**Which baseline performs best?**
```python
baselines.groupby('baseline')['pearson_r'].mean().sort_values(ascending=False)
```

**How much does LSFT help?**
```python
lsft.sort_values('mean_delta_r', ascending=False)
```

**Does PCA maintain LOGO performance?**
```python
logo[logo['baseline'] == 'lpm_selftrained']
```

---

## 📈 File Descriptions

### `baseline_performance_all_analyses.csv`
Complete baseline performance metrics.

**Columns:**
- `dataset`: adamson, k562, or rpe1
- `baseline`: Baseline type identifier
- `pearson_r`: Pearson correlation coefficient (higher = better)
- `l2`: L2 distance (lower = better)
- `analysis_type`: 'pseudobulk' or 'single_cell'

**Key Findings (Single-Cell):**
- Self-trained PCA (lpm_selftrained) consistently wins: **0.40 r** (adamson), **0.26 r** (k562), **0.40 r** (rpe1)
- Random embeddings perform poorly: **~0.20 r** (adamson/rpe1), **~0.07 r** (k562)
- Foundation models (scGPT, scFoundation) show intermediate performance

### `best_baseline_per_dataset.csv`
Identifies the top-performing baseline for each dataset and analysis type.

**Use Case:** Quick reference for winners across experimental conditions.

### `lsft_improvement_summary.csv`
Quantifies how much Local Similarity-Filtered Training (LSFT) improves each baseline.

**Columns:**
- `mean_delta_r`: Average improvement from LSFT
- `max_delta_r`: Maximum improvement observed
- `mean_baseline_r`: Baseline performance before LSFT
- `mean_lsft_r`: Performance after LSFT

**Key Finding:** Random gene embeddings gain **+0.18 r** from LSFT (confirming geometry dominates), while PCA gains only **+0.003 r** (already aligned with manifold).

### `logo_generalization_all_analyses.csv`
Extrapolation performance when holding out functional classes (LOGO evaluation).

**Key Finding:** Self-trained PCA maintains performance (**~0.42 r** on Adamson) while random embeddings fail completely (**~0.23 r**).

---

## 🔍 Analysis Patterns to Explore

1. **Manifold Law Validation:** Does PCA consistently outperform random across both pseudobulk and single-cell?

2. **Geometry Dominance:** Do random embeddings gain more from LSFT than PCA? (Expected: Yes)

3. **Generalization Gap:** Does PCA maintain LOGO performance while foundation models degrade? (Expected: Yes)

4. **Aggregation Consistency:** Are baseline rankings consistent between pseudobulk and single-cell? (Expected: Yes)

---

## 📝 Notes

- **Missing pseudobulk data:** Will be automatically included when result files are detected
- **Single-cell focus:** Current results reflect single-cell analysis completion
- **Metrics:** All correlations are Pearson r (higher is better). L2 distances also available.
- **Datasets:** 
  - **Adamson:** 12 test perturbations (easy dataset)
  - **K562:** 163 test perturbations (medium-hard dataset)  
  - **RPE1:** 231 test perturbations (medium-hard dataset)

---

## 📚 Related Documentation

- **`ENGINEER_ANALYSIS_GUIDE.md`** - Detailed analysis workflow and interpretations
- **`../results/single_cell_analysis/comparison/SINGLE_CELL_ANALYSIS_REPORT.md`** - Full single-cell analysis report
- **`../docs/PIPELINE.md`** - Methodology and pipeline description
- **`../docs/DATA_SOURCES.md`** - Dataset information and sources

---

**Questions?** Refer to the analysis guide or check the source results in `../results/`


