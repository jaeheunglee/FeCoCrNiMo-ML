"""SHAP feature-importance analysis for both SVC models, computed on the
predicted probability of the BCC class with the exact Shapley-value algorithm."""
import pathlib
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ── SVC model 1 (composition only) ────────────────────────────────────────────
final_comp = joblib.load(str(MODELS / "comp_SVC.pkl"))
data_comp = pd.read_excel(DATA / "FeCoCrNiMo_SVC.xlsx")
X_comp = data_comp.drop(columns=["idx", "label"])

predict_fn_comp = lambda x: final_comp.predict_proba(x)[:, 1]   # P(BCC)
background_comp = shap.sample(X_comp, 20, random_state=42)
explainer_comp = shap.Explainer(predict_fn_comp, background_comp)
print("SVC model 1 explainer:", type(explainer_comp).__name__)
shap_values_comp = explainer_comp(X_comp)

plt.figure()
shap.summary_plot(shap_values_comp, X_comp, show=False)
plt.tight_layout()
plt.savefig(FIG / "SHAP_comp_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values_comp, X_comp, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(FIG / "SHAP_comp_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# ── SVC model 2 (composition + physical descriptors) ──────────────────────────
final_phys = joblib.load(str(MODELS / "phys_SVC.pkl"))
data_phys = pd.read_excel(DATA / "FeCoCrNiMo_SVC_2.xlsx")
X_phys = data_phys.drop(columns=["idx", "label"])

background_phys = shap.sample(X_phys, 50, random_state=42)
explainer_phys = shap.Explainer(final_phys.predict_proba, background_phys)
print("SVC model 2 explainer:", type(explainer_phys).__name__)
shap_values_phys = explainer_phys(X_phys)

plt.figure()
shap.summary_plot(shap_values_phys[:, :, 1], X_phys, show=False)   # class 1 = BCC
plt.tight_layout()
plt.savefig(FIG / "SHAP_phys_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values_phys[:, :, 1], X_phys, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(FIG / "SHAP_phys_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSHAP figures saved to {FIG}")
