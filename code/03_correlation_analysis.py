"""Pearson correlation analysis between the elemental compositions and the
physical descriptors used in SVC model 2, to assess feature redundancy."""
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

data_phys = pd.read_excel(DATA / "FeCoCrNiMo_SVC_2.xlsx")

features_comp = ["Fe", "Co", "Cr", "Ni", "Mo"]
physics_features = ["dG", "Ms", "VEC", "S", "dR"]
label_map = {"dG": r"$\Delta G$", "Ms": "$M_s$", "VEC": "VEC", "S": r"$\Delta S$", "dR": r"$\delta$"}

corr_cols = features_comp + physics_features
corr = data_phys[corr_cols].corr()
corr.index = corr.columns = [label_map.get(c, c) for c in corr_cols]

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, cbar_kws={"shrink": 0.85})
plt.tight_layout()
plt.savefig(FIG / "correlation_comp_physics.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Correlation heatmap saved to {FIG / 'correlation_comp_physics.png'}")
