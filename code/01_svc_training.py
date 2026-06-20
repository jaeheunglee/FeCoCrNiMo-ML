"""Train the two RBF-kernel SVC phase classifiers (SVC model 1: composition
only; SVC model 2: composition + physical descriptors) with grid-search
hyperparameter optimization."""
import pathlib
import pandas as pd
import joblib

from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ── Load datasets ──────────────────────────────────────────────────────────────
data_comp = pd.read_excel(DATA / "FeCoCrNiMo_SVC.xlsx")     # Fe, Co, Cr, Ni, Mo, label
data_phys = pd.read_excel(DATA / "FeCoCrNiMo_SVC_2.xlsx")   # + dG, Ms, VEC, S, dR

X_comp, y_comp = data_comp.drop(columns=["idx", "label"]), data_comp["label"]
X_phys, y_phys = data_phys.drop(columns=["idx", "label"]), data_phys["label"]

# ── Repeated stratified 10-fold CV (10 repeats = 100 evaluations) ─────────────
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

param_grid = {
    "svc__kernel": ["rbf"],
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1, 1],
}

def train_and_evaluate(X, y, label):
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(probability=True))])
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=1, verbose=1)
    grid.fit(X, y)

    scores = cross_validate(
        grid.best_estimator_, X, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"], n_jobs=1,
    )

    print(f"\n===== {label} SVC =====")
    print("Best hyperparameters:", grid.best_params_)
    for metric, values in scores.items():
        if metric.startswith("test_"):
            print(f"{metric[5:]:10s}: {values.mean():.4f} +/- {values.std():.4f}")

    return grid.best_estimator_

final_comp = train_and_evaluate(X_comp, y_comp, "SVC model 1 (composition only)")
final_phys = train_and_evaluate(X_phys, y_phys, "SVC model 2 (composition + physical descriptors)")

joblib.dump(final_comp, str(MODELS / "comp_SVC.pkl"))
joblib.dump(final_phys, str(MODELS / "phys_SVC.pkl"))
print(f"\nModels saved to {MODELS}")
