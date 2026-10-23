"""Is MACCS enough, or does a richer descriptor earn its place out of scope?

MACCS wins on average over repeated partitions, but it is only 166 bits. A
coarse representation may suit a diverse dataset while generalising worse to
chemotypes unlike anything in training - which is exactly where a user most
needs help. Averages hide that, so this splits accuracy by distance from the
training set.

Also tests whether combining the four descriptors helps, since the app serves
them separately and a user has no way to know which to pick.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
SEED = 42
FINGERPRINTS = ["maccs", "morgan", "avalon", "topological_torsion"]
BANDS = [("Out of scope", 0.0, 0.3), ("Edge of scope", 0.3, 0.5), ("In scope", 0.5, 1.01)]


def oversample(X, y, seed):
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    target = counts.max()
    px, py = [X], [y]
    for cls, count in enumerate(counts):
        if count < target:
            idx = np.where(y == cls)[0]
            extra = rng.choice(idx, size=target - count, replace=True)
            px.append(X[extra])
            py.append(y[extra])
    return np.vstack(px), np.concatenate(py)


y = np.load(DATA / "labels_dedup.npy")
mask = np.load(DATA / "scaffold_test_mask.npy")
molecules = pd.read_csv(DATA / "molecules_dedup.csv")

# Distance from the training set, on Morgan, as the app computes it.
bits = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
        for s in molecules["canonical"]]
train_bits = [b for b, t in zip(bits, mask) if not t]
similarity = np.array([max(DataStructs.BulkTanimotoSimilarity(b, train_bits))
                       for b, t in zip(bits, mask) if t])

probabilities: dict[str, np.ndarray] = {}
rows = []
for name in FINGERPRINTS:
    X = np.load(DATA / f"X_dedup_{name}.npy").astype(float)
    X_fit, y_fit = oversample(X[~mask], y[~mask], SEED)
    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X_fit, y_fit)
    prob = model.predict_proba(X[mask])[:, 1]
    probabilities[name] = prob
    y_test = y[mask]

    for band, low, high in BANDS:
        sel = (similarity >= low) & (similarity < high)
        if sel.sum() < 20:
            continue
        pred = (prob[sel] >= 0.5).astype(int)
        rows.append({
            "fingerprint": name, "band": band, "n": int(sel.sum()),
            "ACC": round(accuracy_score(y_test[sel], pred), 3),
            "MCC": round(matthews_corrcoef(y_test[sel], pred), 3),
            "AUC": round(roc_auc_score(y_test[sel], prob[sel]), 3),
        })

# Averaging the four probabilities costs nothing at prediction time and would
# spare the user a choice they have no basis for making.
combined = np.mean([probabilities[n] for n in FINGERPRINTS], axis=0)
probabilities["combined"] = combined
for band, low, high in BANDS:
    sel = (similarity >= low) & (similarity < high)
    if sel.sum() < 20:
        continue
    pred = (combined[sel] >= 0.5).astype(int)
    rows.append({
        "fingerprint": "combined", "band": band, "n": int(sel.sum()),
        "ACC": round(accuracy_score(y[mask][sel], pred), 3),
        "MCC": round(matthews_corrcoef(y[mask][sel], pred), 3),
        "AUC": round(roc_auc_score(y[mask][sel], combined[sel]), 3),
    })

frame = pd.DataFrame(rows)
frame.to_csv(RESULTS / "fingerprint_by_scope.csv", index=False)

print(f"{'band':<16}{'fingerprint':<22}{'n':>6}{'ACC':>8}{'MCC':>8}{'AUC':>8}")
print("-" * 68)
for band, _, _ in BANDS:
    block = frame[frame.band == band].sort_values("MCC", ascending=False)
    for _, r in block.iterrows():
        print(f"{r['band']:<16}{r['fingerprint']:<22}{r['n']:>6}"
              f"{r['ACC']:>8.3f}{r['MCC']:>8.3f}{r['AUC']:>8.3f}")
    print()

overall = {}
for name, prob in probabilities.items():
    pred = (prob >= 0.5).astype(int)
    overall[name] = {
        "ACC": round(accuracy_score(y[mask], pred), 3),
        "MCC": round(matthews_corrcoef(y[mask], pred), 3),
        "AUC": round(roc_auc_score(y[mask], prob), 3),
    }
print("overall, whole test set:")
for name, scores in sorted(overall.items(), key=lambda kv: -kv[1]["MCC"]):
    print(f"  {name:<22} MCC {scores['MCC']:.3f}  AUC {scores['AUC']:.3f}")

(RESULTS / "fingerprint_by_scope.json").write_text(
    json.dumps({"by_band": rows, "overall": overall}, indent=2))
