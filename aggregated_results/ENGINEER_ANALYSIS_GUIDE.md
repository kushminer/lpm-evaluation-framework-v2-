# Aggregated Research Results - Engineer Analysis Guide

**Generated:** 2025-11-25 21:57:41

## Overview

This directory contains aggregated results from both **pseudobulk** and **single-cell** analyses of the Manifold Law research.

## Directory Structure

```
aggregated_results/
├── baseline_comparison_pseudobulk_vs_single_cell.csv     # Direct comparison
├── baseline_performance_all_analyses.csv                 # All baseline results
├── best_baseline_per_dataset.csv                         # Winners per dataset
├── lsft_improvement_summary.csv                          # LSFT lift analysis
├── logo_generalization_all_analyses.csv                  # Extrapolation results
└── ENGINEER_ANALYSIS_GUIDE.md                           # This file
```

## Key Files

### 1. `baseline_comparison_pseudobulk_vs_single_cell.csv`
**Purpose:** Direct comparison of baseline performance between pseudobulk and single-cell analyses.

**Columns:**
- `dataset`: Dataset name (adamson, k562, rpe1)
- `baseline`: Baseline type (e.g., lpm_selftrained, lpm_randomGeneEmb)
- `pseudobulk_r`: Pearson correlation (pseudobulk)
- `single_cell_r`: Pearson correlation (single-cell)
- `delta`: Difference (single_cell_r - pseudobulk_r)

**Key Questions:**
- Does the Manifold Law hold consistently across aggregation methods?
- Are there systematic differences between pseudobulk and single-cell?

### 2. `baseline_performance_all_analyses.csv`
**Purpose:** Complete baseline performance across all datasets and analysis types.

**Columns:**
- `dataset`: Dataset name
- `baseline`: Baseline type
- `pearson_r`: Pearson correlation coefficient
- `l2`: L2 distance
- `analysis_type`: 'pseudobulk' or 'single_cell'

**Key Questions:**
- Which baselines perform best overall?
- How does performance vary by dataset difficulty?

### 3. `best_baseline_per_dataset.csv`
**Purpose:** Identifies the winning baseline for each dataset and analysis type.

**Key Questions:**
- Is PCA (selftrained) consistently the winner?
- Do winners change between pseudobulk and single-cell?

### 4. `lsft_improvement_summary.csv`
**Purpose:** Quantifies how much LSFT (Local Similarity-Filtered Training) improves each baseline.

**Columns:**
- `dataset`: Dataset name
- `baseline`: Baseline type
- `mean_delta_r`: Average improvement from LSFT
- `max_delta_r`: Maximum improvement observed
- `mean_baseline_r`: Baseline performance before LSFT
- `mean_lsft_r`: Performance after LSFT

**Key Questions:**
- Which baselines benefit most from local similarity filtering?
- Does random embedding gain more than PCA (suggesting geometry dominates)?

### 5. `logo_generalization_all_analyses.csv`
**Purpose:** Extrapolation performance when holding out functional classes (LOGO).

**Key Questions:**
- Does PCA maintain performance when extrapolating to novel functions?
- How do foundation models (scGPT, scFoundation) compare?

## Analysis Workflow

### Step 1: Understand Baseline Performance
1. Load `baseline_performance_all_analyses.csv`
2. Group by `baseline` and compute statistics (mean, std, min, max)
3. Create visualizations comparing baselines across datasets

### Step 2: Compare Aggregation Methods
1. Load `baseline_comparison_pseudobulk_vs_single_cell.csv`
2. Compute correlation between pseudobulk_r and single_cell_r
3. Identify baselines with largest deltas (systematic differences)

### Step 3: Analyze LSFT Impact
1. Load `lsft_improvement_summary.csv`
2. Sort by `mean_delta_r` to find baselines that gain most
3. Test hypothesis: Random embeddings should gain more than PCA

### Step 4: Evaluate Generalization
1. Load `logo_generalization_all_analyses.csv`
2. Compare LOGO performance vs baseline performance
3. Identify baselines that maintain performance on novel functions

## Expected Findings

Based on the Manifold Law hypothesis:

1. **PCA (selftrained) should win baselines** across both pseudobulk and single-cell
2. **Random embeddings gain significantly from LSFT** (geometry lift)
3. **PCA maintains LOGO performance** while foundation models degrade
4. **Pseudobulk and single-cell should show consistent rankings** (law holds)

## Notes

- All metrics use Pearson correlation (r) unless otherwise specified
- L2 distances are also available in baseline files
- Single-cell results may have fewer baselines due to computational constraints
- Missing values indicate experiments not yet completed

## Questions or Issues?

Refer to:
- `docs/PIPELINE.md` for methodology
- `docs/DATA_SOURCES.md` for dataset information
- `results/single_cell_analysis/comparison/SINGLE_CELL_ANALYSIS_REPORT.md` for detailed findings
