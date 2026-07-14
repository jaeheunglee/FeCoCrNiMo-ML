# Fe-Co-Cr-Ni-Mo MEA: interpretable ML code

Code accompanying the machine-learning analysis in the manuscript. Scripts are
numbered in the order they are used in the pipeline.

## Contents

- `code/01_svc_training.py` — Trains the SVC model 1 and 2
  RBF-kernel SVC phase classifiers (γ vs. α'-containing). Hyperparameters are selected by
  grid search under repeated stratified 10-fold cross-validation (10 repeats).
  Saves trained models to `models/`.
- `code/02_shap_analysis.py` — Computes SHAP feature-importance values for both
  trained SVC models (exact Shapley-value computation) and saves beeswarm/bar
  summary plots to `figures/`.
- `code/03_correlation_analysis.py` — Pearson correlation analysis between
  elemental compositions and physical descriptors, used to assess
  feature redundancy.
- `code/04_symbolic_regression.py` — Builds a dense pseudo-ternary compositional
  grid from the measured powder feedstocks, evaluates the trained 
  SVC model 1 on the grid, and fits a closed-form symbolic expression for the
  γ/γ+α' decision boundary using PySR.

## Data

- `data/FeCoCrNiMo_SVC.xlsx` — 52-alloy dataset : Fe (at%), Co(at%), Cr(at%), Ni(at%), Mo(at%), phase label.
- `data/FeCoCrNiMo_SVC_2.xlsx` — same 52 alloys with additional physical
  descriptors : dG=ΔG_BCC-FCC (J mol⁻¹), Ms (°C), VEC, S=ΔS_conf/R, dR=δ.

## Setup

```bash
pip install -r requirements.txt
```

`04_symbolic_regression.py` additionally requires a working Julia installation
(installed automatically on first run via PySR; see the
[PySR documentation](https://github.com/MilesCranmer/PySR) for details).

## Usage

```bash
python code/01_svc_training.py
python code/02_shap_analysis.py
python code/03_correlation_analysis.py
python code/04_symbolic_regression.py
```

Outputs are written to `models/`, `figures/`, and `results/`.
