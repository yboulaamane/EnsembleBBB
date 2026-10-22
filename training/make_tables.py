"""Turn the raw metrics into the tables the corrected manuscript needs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

df = pd.read_csv(RESULTS / "metrics.csv")

FP_ORDER = ["morgan", "maccs", "avalon", "topological_torsion"]
MODEL_ORDER = ["RF", "SVC", "kNN", "XGB", "Ensemble"]
FP_NAME = {
    "morgan": "Morgan", "maccs": "MACCS", "avalon": "Avalon",
    "topological_torsion": "Topological torsion",
}
PROTOCOL_NAME = {
    "A": "A  original (resample, then split)",
    "B": "B  random split, in-fold resampling",
    "C": "C  scaffold split, in-fold resampling",
}

df["fingerprint"] = pd.Categorical(df["fingerprint"], FP_ORDER, ordered=True)
df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
df = df.sort_values(["protocol", "fingerprint", "model"])

# ---------------------------------------------------------------- Table 1
# What the validation protocol alone is worth, averaged over fingerprints.
summary = (
    df.groupby(["protocol", "model"], observed=True)[["AUC", "MCC", "ACC", "BACC"]]
    .mean()
    .round(3)
    .reset_index()
)
summary.to_csv(RESULTS / "table_protocol_summary.csv", index=False)

print("=" * 72)
print("PROTOCOL COMPARISON  (mean over the four fingerprints)")
print("=" * 72)
for protocol in ["A", "B", "C"]:
    block = summary[summary["protocol"] == protocol]
    print(f"\n{PROTOCOL_NAME[protocol]}")
    print(block.drop(columns="protocol").to_string(index=False))

# ---------------------------------------------------------------- Table 2
# The replacement for the published Table 3: honest protocol, full metrics.
corrected = df[df["protocol"] == "C"][
    ["fingerprint", "model", "SE", "SP", "Q+", "Q-", "ACC", "BACC",
     "F1", "MCC", "kappa", "g_mean", "AUC", "mean_cv_auc", "std_cv_auc"]
].round(3)
corrected.to_csv(RESULTS / "table_corrected_performance.csv", index=False)

print("\n" + "=" * 72)
print("CORRECTED TABLE 3  (protocol C, scaffold split)")
print("=" * 72)
for fp in FP_ORDER:
    block = corrected[corrected["fingerprint"] == fp]
    print(f"\n{FP_NAME[fp]}")
    print(block.drop(columns="fingerprint").to_string(index=False))

# ---------------------------------------------------------------- Table 3
# Does the ensemble ever beat the best single model? Asked by both reviewers.
rows = []
for (protocol, fp), block in df.groupby(["protocol", "fingerprint"], observed=True):
    singles = block[block["model"] != "Ensemble"]
    ens = block[block["model"] == "Ensemble"].iloc[0]
    best = singles.loc[singles["MCC"].idxmax()]
    rows.append({
        "protocol": protocol,
        "fingerprint": FP_NAME[fp],
        "best_single": best["model"],
        "best_single_MCC": round(best["MCC"], 3),
        "ensemble_MCC": round(ens["MCC"], 3),
        "delta_MCC": round(ens["MCC"] - best["MCC"], 3),
        "best_single_AUC": round(best["AUC"], 3),
        "ensemble_AUC": round(ens["AUC"], 3),
        "delta_AUC": round(ens["AUC"] - best["AUC"], 3),
    })
verdict = pd.DataFrame(rows)
verdict.to_csv(RESULTS / "table_ensemble_vs_best.csv", index=False)

print("\n" + "=" * 72)
print("ENSEMBLE vs BEST SINGLE MODEL")
print("=" * 72)
print(verdict.to_string(index=False))

wins = int((verdict["delta_MCC"] > 0).sum())
print(f"\nensemble beats the best single model in {wins} of {len(verdict)} comparisons")
print(f"mean delta MCC: {verdict['delta_MCC'].mean():+.3f}")
print(f"mean delta AUC: {verdict['delta_AUC'].mean():+.3f}")

# ---------------------------------------------------------------- inflation
print("\n" + "=" * 72)
print("INFLATION ATTRIBUTABLE TO THE ORIGINAL PROTOCOL")
print("=" * 72)
pivot = df.pivot_table(
    index=["fingerprint", "model"], columns="protocol",
    values=["AUC", "MCC"], observed=True,
)
for metric in ["AUC", "MCC"]:
    a, c = pivot[(metric, "A")], pivot[(metric, "C")]
    print(f"  {metric}:  A {a.mean():.3f}  ->  C {c.mean():.3f}   "
          f"inflation {(a - c).mean():+.3f}  (max {(a - c).max():+.3f})")
