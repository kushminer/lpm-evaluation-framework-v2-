#!/bin/bash
#
# Run single-cell baseline evaluation for Adamson and K562 datasets
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate gears_env2

# Data paths
ADAMSON_DATA="/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad"
K562_DATA="/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad"

# Split config paths (use existing or create)
RESULTS_DIR="$SCRIPT_DIR/results/single_cell_analysis"
mkdir -p "$RESULTS_DIR"

# Parameters
N_CELLS_PER_PERT=50
PCA_DIM=10
RIDGE_PENALTY=0.1
SEED=1

echo "=============================================="
echo "SINGLE-CELL BASELINE EVALUATION"
echo "=============================================="
echo "N cells per perturbation: $N_CELLS_PER_PERT"
echo "PCA dimension: $PCA_DIM"
echo "Ridge penalty: $RIDGE_PENALTY"
echo "Seed: $SEED"
echo ""

# Add src to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Run Python script
python3 << 'EOF'
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from goal_2_baselines.baseline_runner_single_cell import run_all_baselines_single_cell
from goal_2_baselines.baseline_types import BaselineType
from goal_2_baselines.split_logic import load_split_config, prepare_perturbation_splits
import anndata as ad
import json

# Configuration
datasets = {
    "adamson": {
        "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad",
    },
    "k562": {
        "adata_path": "/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad",
    },
}

n_cells_per_pert = 50
pca_dim = 10
ridge_penalty = 0.1
seed = 1

results_dir = Path("results/single_cell_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

# Baselines to run
baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
    BaselineType.RANDOM_PERT_EMB,
]

for dataset_name, config in datasets.items():
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Dataset: {dataset_name}")
    LOGGER.info(f"{'='*60}")
    
    adata_path = Path(config["adata_path"])
    
    if not adata_path.exists():
        LOGGER.warning(f"Data file not found: {adata_path}")
        continue
    
    # Create or load split config
    split_config_path = results_dir / f"{dataset_name}_split_config.json"
    
    if not split_config_path.exists():
        LOGGER.info("Creating train/test split...")
        adata = ad.read_h5ad(adata_path)
        
        # Get all perturbation conditions (excluding ctrl)
        conditions = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
        
        # Simple random 80/20 split
        import numpy as np
        np.random.seed(seed)
        np.random.shuffle(conditions)
        
        n_train = int(0.8 * len(conditions))
        train_conditions = conditions[:n_train]
        test_conditions = conditions[n_train:]
        
        # Include ctrl in all splits
        split_config = {
            "train": list(train_conditions) + ["ctrl"],
            "test": list(test_conditions),
        }
        
        with open(split_config_path, "w") as f:
            json.dump(split_config, f, indent=2)
        
        LOGGER.info(f"Created split: {len(train_conditions)} train, {len(test_conditions)} test")
    else:
        LOGGER.info(f"Using existing split config: {split_config_path}")
    
    # Run baselines
    output_dir = results_dir / dataset_name
    
    try:
        results_df = run_all_baselines_single_cell(
            adata_path=adata_path,
            split_config_path=split_config_path,
            output_dir=output_dir,
            baseline_types=baselines,
            pca_dim=pca_dim,
            ridge_penalty=ridge_penalty,
            seed=seed,
            n_cells_per_pert=n_cells_per_pert,
            cell_embedding_method="cell_pca",
        )
        
        LOGGER.info(f"\n{dataset_name} Results:")
        LOGGER.info(f"\n{results_df.to_string()}")
        
    except Exception as e:
        LOGGER.error(f"Failed to run {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

LOGGER.info("\n" + "="*60)
LOGGER.info("SINGLE-CELL BASELINE EVALUATION COMPLETE")
LOGGER.info("="*60)
EOF

echo ""
echo "Done! Results saved to $RESULTS_DIR"

