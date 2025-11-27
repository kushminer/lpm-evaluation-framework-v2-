# Changelog

All notable changes are recorded here. Reverse chronological order.

## 2025-11-26
- **Cataloged historical documentation**: Organized archived docs into categorized subdirectories (`sprint_docs/`, `execution_docs/`, `status_reports/`, `setup_docs/`, `planning_docs/`) with comprehensive index.
- Consolidating documentation: Moved transient status reports, sprint summaries, and setup guides to `archive/docs/`.
- Organized `docs/` folder with `PIPELINE.md`, `DATA_SOURCES.md`, and `CHANGELOG.md`.
- Cleaned up root directory by archiving `WORKTREE_NOT_NEEDED.md` and `CLEANUP_PLAN.md`.
- Archived publication package setup documentation now that generation is complete.

## 2025-11-25
- Fixed single-cell baseline runner to load true GEARS perturbation
  embeddings; reran Adamson/K562/RPE1 baselines and LOGO evaluations.
- Launched expanded LSFT sweeps for all datasets/baselines (in progress).
- Added GEARS vs PCA audit tooling and visuals
  (`audits/single_cell_data_audit/`).
- Created repository refinement plan and top-level documentation.

## 2025-11-24
- Regenerated publication package figures and
  `SINGLE_CELL_ANALYSIS_REPORT.md` after fixing epic aggregation.
- Cleaned empty Epic 2 files and stabilized `run_all_epics_all_baselines.sh`.

## 2025-11-23
- Diagnosed worktree issues, restored missing modules for Cursor, and
  re-established Python 3.10 environment compatibility.
- Added initial single-cell baseline scripts and auditing utilities.
