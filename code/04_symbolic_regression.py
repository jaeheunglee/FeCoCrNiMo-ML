"""Symbolic regression (PySR) of the SVC model 1 decision boundary over a
dense pseudo-ternary powder-mixing grid (boundary points, |decision| < 1.0).
Requires PySR (https://github.com/MilesCranmer/PySR), which depends on Julia."""
import pathlib
import warnings
import numpy as np
import pandas as pd
import joblib
from pysr import PySRRegressor

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ── Measured powder compositions (at.%), Table 2 ──────────────────────────────
POWDER_A = {"Fe": 65.27, "Co": 24.73, "Cr": 0.00, "Ni": 0.00, "Mo": 10.00}   # Fe-Co-Mo
POWDER_B = {"Fe": 33.56, "Co": 0.00, "Cr": 32.94, "Ni": 33.50, "Mo": 0.00}  # Fe-Cr-Ni
POWDER_C = {"Fe": 61.27, "Co": 31.33, "Cr": 7.40, "Ni": 0.00, "Mo": 0.00}   # Fe-Co-Cr
ELEMENTS = ["Fe", "Co", "Cr", "Ni", "Mo"]

def ternary_composition(pA, pB, pC):
    return [POWDER_A[e] * pA + POWDER_B[e] * pB + POWDER_C[e] * pC for e in ELEMENTS]

# ── Build constrained pseudo-ternary grid (1% steps) ──────────────────────────
STEP = 0.01
rows = []
for pA in np.round(np.arange(0, 1 + STEP, STEP), 4):
    for pB in np.round(np.arange(0, 1 - pA + STEP, STEP), 4):
        pC = round(1 - pA - pB, 4)
        if pC < -1e-6:
            continue
        pC = max(pC, 0.0)
        # Each powder is constrained to a minimum of 10% so that all three
        # always remain present; the effective upper bounds are therefore
        # 80% / 60% / 80% for the Fe-Co-Mo / Fe-Cr-Ni / Fe-Co-Cr powders.
        if not (0.10 <= pA <= 0.80):
            continue
        if not (0.10 <= pB <= 0.60):
            continue
        if not (0.10 <= pC <= 0.80):
            continue
        comp = ternary_composition(pA, pB, pC)
        rows.append({"pA": pA, "pB": pB, "pC": pC, **dict(zip(ELEMENTS, comp))})

grid = pd.DataFrame(rows)
print(f"Pseudo-ternary grid: n={len(grid)}")

# ── Evaluate trained SVC model 1 on the grid ──────────────────────────────────
pipe = joblib.load(str(MODELS / "comp_SVC.pkl"))
grid["decision"] = pipe.decision_function(grid[ELEMENTS])
grid["prediction"] = pipe.predict(grid[ELEMENTS])
print(f"FCC={(grid['prediction']==0).sum()}  BCC={(grid['prediction']==1).sum()}")

# ── Restrict to points near the decision boundary ─────────────────────────────
THRESHOLD = 1.0
boundary = grid[grid["decision"].abs() < THRESHOLD].reset_index(drop=True)
print(f"Boundary points (|decision| < {THRESHOLD}): n={len(boundary)}")

# ── Symbolic regression ───────────────────────────────────────────────────────
X = boundary[ELEMENTS].values
y = boundary["decision"].values

model = PySRRegressor(
    niterations=500,
    population_size=50,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sqrt"],
    maxsize=25,
    maxdepth=10,
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    random_state=42,
)
model.fit(X, y, variable_names=ELEMENTS)

equations = model.equations_
equations.to_excel(str(OUT / "SR_equations_boundary.xlsx"), index=False)
print(equations[["complexity", "loss", "score", "equation"]].to_string(index=False))
print(f"\nSaved full Pareto front to {OUT / 'SR_equations_boundary.xlsx'}")
