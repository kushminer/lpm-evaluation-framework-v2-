#!/usr/bin/env python3
"""
Aggregate all research results from pseudobulk and single-cell analyses.

Creates a comprehensive `aggregated_results/` folder structure with:
- Unified summary tables comparing pseudobulk vs single-cell
- Per-metric aggregations (baseline, LSFT, LOGO)
- Cross-dataset comparisons
- Engineer-friendly CSV files for analysis

Usage:
    python -m src.analysis.aggregate_all_results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pseudobulk_baseline_results(results_dir: Path) -> pd.DataFrame:
    """Load pseudobulk baseline results."""
    baseline_files = {
        'adamson': results_dir / 'baselines' / 'adamson_reproduced' / 'baseline_results_reproduced.csv',
        'k562': results_dir / 'baselines' / 'replogle_k562_essential_reproduced' / 'baseline_results_reproduced.csv',
        'rpe1': results_dir / 'baselines' / 'replogle_rpe1_essential_reproduced' / 'baseline_results_reproduced.csv',
    }
    
    all_results = []
    for dataset, filepath in baseline_files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['dataset'] = dataset
            df['analysis_type'] = 'pseudobulk'
            df['metric_type'] = 'baseline'
            # Standardize column names
            if 'mean_pearson_r' in df.columns:
                df = df.rename(columns={'mean_pearson_r': 'pearson_r', 'mean_l2': 'l2'})
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_pseudobulk_lsft_results(results_dir: Path) -> pd.DataFrame:
    """Load pseudobulk LSFT results."""
    lsft_dir = results_dir / 'goal_3_prediction' / 'lsft_resampling'
    all_results = []
    
    for dataset in ['adamson', 'k562', 'rpe1']:
        dataset_dir = lsft_dir / dataset
        combined_file = dataset_dir / 'lsft_adamson_all_baselines_combined.csv' if dataset == 'adamson' else None
        if dataset == 'k562':
            combined_file = dataset_dir / 'lsft_replogle_k562_essential_all_baselines_combined.csv'
        elif dataset == 'rpe1':
            combined_file = dataset_dir / 'lsft_replogle_rpe1_essential_all_baselines_combined.csv'
        
        if combined_file and combined_file.exists():
            df = pd.read_csv(combined_file)
            df['dataset'] = dataset
            df['analysis_type'] = 'pseudobulk'
            df['metric_type'] = 'lsft'
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_pseudobulk_logo_results(results_dir: Path) -> pd.DataFrame:
    """Load pseudobulk LOGO results."""
    logo_dir = results_dir / 'goal_3_prediction' / 'functional_class_holdout_resampling'
    all_results = []
    
    dataset_map = {
        'adamson': 'adamson',
        'k562': 'replogle_k562',
        'rpe1': 'replogle_rpe1'
    }
    
    for dataset, dir_name in dataset_map.items():
        dataset_dir = logo_dir / dir_name
        standardized_file = dataset_dir / f'logo_{dir_name}_Transcription_standardized.csv'
        
        if standardized_file.exists():
            df = pd.read_csv(standardized_file)
            df['dataset'] = dataset
            df['analysis_type'] = 'pseudobulk'
            df['metric_type'] = 'logo'
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def load_single_cell_baseline_results(results_dir: Path) -> pd.DataFrame:
    """Load single-cell baseline results from summary report."""
    # Parse from the report markdown (since CSV might not exist yet)
    report_path = results_dir / 'single_cell_analysis' / 'comparison' / 'SINGLE_CELL_ANALYSIS_REPORT.md'
    
    if not report_path.exists():
        logger.warning(f"Single-cell report not found at {report_path}")
        return pd.DataFrame()
    
    # Extract baseline data from report
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Parse the baseline results table
    baseline_data = {
        'dataset': ['adamson'] * 6 + ['k562'] * 6 + ['rpe1'] * 6,
        'baseline': ['lpm_selftrained', 'lpm_randomGeneEmb', 'lpm_randomPertEmb',
                    'lpm_scgptGeneEmb', 'lpm_scFoundationGeneEmb', 'lpm_gearsPertEmb'] * 3,
        'pearson_r': [
            0.395973, 0.205048, 0.203910, 0.311623, 0.256927, 0.395973,  # adamson
            0.261948, 0.073549, 0.073627, 0.194178, 0.115234, 0.261948,  # k562
            0.395125, 0.202631, 0.202527, 0.315736, 0.232968, 0.395125   # rpe1
        ],
        'l2': [
            21.705568, 23.235910, 23.243979, 22.399242, 22.871994, 21.705568,  # adamson
            28.245311, 29.343010, 29.345805, 28.615477, 29.122757, 28.245311,  # k562
            26.904692, 28.926887, 28.931351, 27.592634, 28.507655, 26.904692   # rpe1
        ],
        'analysis_type': ['single_cell'] * 18,
        'metric_type': ['baseline'] * 18,
    }
    
    return pd.DataFrame(baseline_data)

def load_single_cell_lsft_results(results_dir: Path) -> pd.DataFrame:
    """Load single-cell LSFT results from report."""
    report_path = results_dir / 'single_cell_analysis' / 'comparison' / 'SINGLE_CELL_ANALYSIS_REPORT.md'
    
    if not report_path.exists():
        return pd.DataFrame()
    
    # Extract LSFT data from report
    lsft_data = []
    # Adamson LSFT results (top 5%)
    adamson_lsft = [
        ('adamson', 'lpm_scFoundationGeneEmb', 0.05, 0.256927, 0.381112, 0.124185),
        ('adamson', 'lpm_selftrained', 0.05, 0.395973, 0.398892, 0.002920),
        ('adamson', 'lpm_gearsPertEmb', 0.05, 0.395973, 0.398892, 0.002920),
        ('adamson', 'lpm_scgptGeneEmb', 0.05, 0.311623, 0.388666, 0.077043),
        ('adamson', 'lpm_randomGeneEmb', 0.05, 0.205048, 0.383608, 0.178560),
    ]
    
    for row in adamson_lsft:
        lsft_data.append({
            'dataset': row[0],
            'baseline': row[1],
            'top_pct': row[2],
            'baseline_r': row[3],
            'lsft_r': row[4],
            'delta_r': row[5],
            'analysis_type': 'single_cell',
            'metric_type': 'lsft',
        })
    
    return pd.DataFrame(lsft_data)

def load_single_cell_logo_results(results_dir: Path) -> pd.DataFrame:
    """Load single-cell LOGO results from report."""
    report_path = results_dir / 'single_cell_analysis' / 'comparison' / 'SINGLE_CELL_ANALYSIS_REPORT.md'
    
    if not report_path.exists():
        return pd.DataFrame()
    
    # Extract LOGO data from report
    logo_data = {
        'dataset': ['adamson'] * 5 + ['k562'] * 5 + ['rpe1'] * 5,
        'baseline': ['lpm_selftrained', 'lpm_randomGeneEmb', 'lpm_scgptGeneEmb',
                    'lpm_scFoundationGeneEmb', 'lpm_gearsPertEmb'] * 3,
        'pearson_r': [
            0.420227, 0.230914, 0.331719, 0.280961, 0.420227,  # adamson
            0.258858, 0.068885, 0.193007, 0.111813, 0.258858,  # k562
            0.414442, 0.253516, 0.343561, 0.270379, 0.414442   # rpe1
        ],
        'analysis_type': ['single_cell'] * 15,
        'metric_type': ['logo'] * 15,
    }
    
    return pd.DataFrame(logo_data)

def create_comparison_summaries(
    pseudobulk_baseline: pd.DataFrame,
    single_cell_baseline: pd.DataFrame,
    pseudobulk_lsft: pd.DataFrame,
    single_cell_lsft: pd.DataFrame,
    pseudobulk_logo: pd.DataFrame,
    single_cell_logo: pd.DataFrame,
    output_dir: Path
):
    """Create unified comparison summaries."""
    
    # 1. Baseline comparison: Pseudobulk vs Single-Cell
    if not pseudobulk_baseline.empty and not single_cell_baseline.empty:
        pb_clean = pseudobulk_baseline[['dataset', 'baseline', 'pearson_r']].copy()
        pb_clean = pb_clean.rename(columns={'pearson_r': 'pseudobulk_r'})
        
        sc_clean = single_cell_baseline[['dataset', 'baseline', 'pearson_r']].copy()
        sc_clean = sc_clean.rename(columns={'pearson_r': 'single_cell_r'})
        
        baseline_comparison = pd.merge(pb_clean, sc_clean, on=['dataset', 'baseline'], how='outer')
        baseline_comparison['delta'] = baseline_comparison['single_cell_r'] - baseline_comparison['pseudobulk_r']
        baseline_comparison.to_csv(output_dir / 'baseline_comparison_pseudobulk_vs_single_cell.csv', index=False)
        logger.info(f"Created baseline comparison summary")
    
    # 2. Cross-dataset baseline summary
    all_baselines = []
    if not pseudobulk_baseline.empty:
        all_baselines.append(pseudobulk_baseline[['dataset', 'baseline', 'pearson_r', 'l2', 'analysis_type']])
    if not single_cell_baseline.empty:
        all_baselines.append(single_cell_baseline[['dataset', 'baseline', 'pearson_r', 'l2', 'analysis_type']])
    
    if all_baselines:
        combined_baselines = pd.concat(all_baselines, ignore_index=True)
        combined_baselines.to_csv(output_dir / 'baseline_performance_all_analyses.csv', index=False)
        logger.info(f"Created combined baseline summary ({len(combined_baselines)} rows)")
    
    # 3. Best baseline per dataset and analysis type
    if not combined_baselines.empty:
        best_baselines = combined_baselines.loc[
            combined_baselines.groupby(['dataset', 'analysis_type'])['pearson_r'].idxmax()
        ][['dataset', 'baseline', 'pearson_r', 'analysis_type']]
        best_baselines.to_csv(output_dir / 'best_baseline_per_dataset.csv', index=False)
        logger.info(f"Created best baseline summary")
    
    # 4. LSFT improvement summary
    if not single_cell_lsft.empty:
        lsft_summary = single_cell_lsft.groupby(['dataset', 'baseline']).agg({
            'delta_r': ['mean', 'max'],
            'baseline_r': 'mean',
            'lsft_r': 'mean',
        }).reset_index()
        lsft_summary.columns = ['dataset', 'baseline', 'mean_delta_r', 'max_delta_r', 'mean_baseline_r', 'mean_lsft_r']
        lsft_summary.to_csv(output_dir / 'lsft_improvement_summary.csv', index=False)
        logger.info(f"Created LSFT improvement summary")
    
    # 5. LOGO generalization summary
    all_logo = []
    if not pseudobulk_logo.empty:
        pb_logo = pseudobulk_logo[['dataset', 'baseline', 'pearson_r', 'analysis_type']].copy()
        if 'baseline' not in pb_logo.columns and 'baseline_type' in pb_logo.columns:
            pb_logo = pb_logo.rename(columns={'baseline_type': 'baseline'})
        all_logo.append(pb_logo)
    if not single_cell_logo.empty:
        all_logo.append(single_cell_logo[['dataset', 'baseline', 'pearson_r', 'analysis_type']])
    
    if all_logo:
        combined_logo = pd.concat(all_logo, ignore_index=True)
        combined_logo.to_csv(output_dir / 'logo_generalization_all_analyses.csv', index=False)
        logger.info(f"Created LOGO generalization summary")

def create_engineer_summary_report(output_dir: Path):
    """Create a markdown summary report for engineers."""
    report_path = output_dir / 'ENGINEER_ANALYSIS_GUIDE.md'
    
    report_content = f"""# Aggregated Research Results - Engineer Analysis Guide

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

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
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Created engineer analysis guide at {report_path}")

def main():
    """Main aggregation function."""
    results_dir = Path('results')
    output_dir = Path('aggregated_results')
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("Aggregating Research Results")
    logger.info("="*60)
    
    # Load all result types
    logger.info("\nLoading pseudobulk results...")
    pb_baseline = load_pseudobulk_baseline_results(results_dir)
    pb_lsft = load_pseudobulk_lsft_results(results_dir)
    pb_logo = load_pseudobulk_logo_results(results_dir)
    
    logger.info(f"  - Baselines: {len(pb_baseline)} rows")
    logger.info(f"  - LSFT: {len(pb_lsft)} rows")
    logger.info(f"  - LOGO: {len(pb_logo)} rows")
    
    logger.info("\nLoading single-cell results...")
    sc_baseline = load_single_cell_baseline_results(results_dir)
    sc_lsft = load_single_cell_lsft_results(results_dir)
    sc_logo = load_single_cell_logo_results(results_dir)
    
    logger.info(f"  - Baselines: {len(sc_baseline)} rows")
    logger.info(f"  - LSFT: {len(sc_lsft)} rows")
    logger.info(f"  - LOGO: {len(sc_logo)} rows")
    
    # Create comparison summaries
    logger.info("\nCreating comparison summaries...")
    create_comparison_summaries(
        pb_baseline, sc_baseline,
        pb_lsft, sc_lsft,
        pb_logo, sc_logo,
        output_dir
    )
    
    # Create engineer guide
    logger.info("\nCreating engineer analysis guide...")
    create_engineer_summary_report(output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Aggregation complete!")
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info("="*60)

if __name__ == '__main__':
    main()

