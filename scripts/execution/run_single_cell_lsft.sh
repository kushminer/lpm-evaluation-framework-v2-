#!/bin/bash
#
# Run single-cell LSFT evaluation for Adamson and K562 datasets
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================="
echo "SINGLE-CELL LSFT EVALUATION"
echo "=============================================="

export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

/opt/anaconda3/envs/gears_env2/bin/python << 'EOF'
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)

from goal_3_prediction.lsft.lsft_single_cell import run_lsft_single_cell_all_baselines
from goal_2_baselines.baseline_types import BaselineType

results_dir = Path('results/single_cell_analysis')

datasets = {
    'adamson': {
        'adata_path': '/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad',
        'split_config': results_dir / 'adamson_split_config.json',
    },
    'k562': {
        'adata_path': '/Users/samuelminer/Documents/classes/nih_research/data_replogle_k562_essential/perturb_processed.h5ad',
        'split_config': results_dir / 'k562_split_config.json',
    },
}

baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
]

top_pcts = [0.05, 0.10]  # 5% and 10%

for dataset_name, config in datasets.items():
    LOGGER.info(f'\n{"="*60}')
    LOGGER.info(f'Dataset: {dataset_name}')
    LOGGER.info(f'{"="*60}')
    
    output_dir = results_dir / dataset_name / 'lsft'
    
    results = run_lsft_single_cell_all_baselines(
        adata_path=Path(config['adata_path']),
        split_config_path=config['split_config'],
        output_dir=output_dir,
        dataset_name=dataset_name,
        baseline_types=baselines,
        top_pcts=top_pcts,
        pca_dim=10,
        ridge_penalty=0.1,
        seed=1,
        n_cells_per_pert=50,
    )
    
    LOGGER.info(f'\n{dataset_name} LSFT Results saved to {output_dir}')

LOGGER.info('\n' + '='*60)
LOGGER.info('SINGLE-CELL LSFT EVALUATION COMPLETE')
LOGGER.info('='*60)
EOF

echo "Done!"

