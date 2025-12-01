#!/bin/bash
#
# Run single-cell LOGO evaluation for Adamson and K562 datasets
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================="
echo "SINGLE-CELL LOGO EVALUATION"
echo "=============================================="

export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"

/opt/anaconda3/envs/gears_env2/bin/python << 'EOF'
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
LOGGER = logging.getLogger(__name__)

from goal_4_logo.logo_single_cell import run_logo_single_cell
from goal_2_baselines.baseline_types import BaselineType

results_dir = Path('results/single_cell_analysis')

datasets = {
    'adamson': {
        'adata_path': '/Users/samuelminer/Documents/classes/nih_research/data_adamson/perturb_processed.h5ad',
        'annotation_path': 'data/annotations/adamson_functional_classes_enriched.tsv',
    },
}

baselines = [
    BaselineType.SELFTRAINED,
    BaselineType.RANDOM_GENE_EMB,
]

for dataset_name, config in datasets.items():
    LOGGER.info(f'\n{"="*60}')
    LOGGER.info(f'Dataset: {dataset_name}')
    LOGGER.info(f'{"="*60}')
    
    output_dir = results_dir / dataset_name / 'logo'
    
    try:
        results = run_logo_single_cell(
            adata_path=Path(config['adata_path']),
            annotation_path=Path(config['annotation_path']),
            dataset_name=dataset_name,
            output_dir=output_dir,
            class_name='Transcription',
            baseline_types=baselines,
            pca_dim=10,
            ridge_penalty=0.1,
            seed=1,
            n_cells_per_pert=50,
        )
        
        LOGGER.info(f'\n{dataset_name} LOGO Results saved to {output_dir}')
    except Exception as e:
        LOGGER.error(f'Failed {dataset_name}: {e}')
        import traceback
        traceback.print_exc()

LOGGER.info('\n' + '='*60)
LOGGER.info('SINGLE-CELL LOGO EVALUATION COMPLETE')
LOGGER.info('='*60)
EOF

echo "Done!"

