"""Evaluate the corrected model on the BBBP external set.

Reports performance on the set as published and on the genuinely external
subset, so the gap between the two is explicit.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, roc_auc_score,
)

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = HERE / "data"
RESULTS = HERE / "results"
SEED = 42


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


def metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": int(len(y_true)),
        "SE": round(se, 3), "SP": round(sp, 3),
        "ACC": round(accuracy_score(y_true, y_pred), 3),
        "BACC": round((se + sp) / 2, 3),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 3),
        "kappa": round(cohen_kappa_score(y_true, y_pred), 3),
        "g_mean": round(float(np.sqrt(se * sp)), 3),
        "AUC": round(roc_auc_score(y_true, y_prob), 3),
    }


# Train the recommended model on the scaffold-split training partition only.
X = np.load(DATA / "X_dedup_maccs.npy").astype(float)
y = np.load(DATA / "labels_dedup.npy")
test_mask = np.load(DATA / "scaffold_test_mask.npy")

# Recover which MACCS bits were retained, so external molecules are encoded
# in exactly the same feature space.
molecules = pd.read_csv(DATA / "molecules_dedup.csv")
full = np.vstack([np.array(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)))
                  for s in molecules["canonical"]])
retained, used = [], set()
for col in range(X.shape[1]):
    for bit in range(full.shape[1]):
        if bit not in used and np.array_equal(full[:, bit], X[:, col]):
            retained.append(bit)
            used.add(bit)
            break
assert len(retained) == X.shape[1], "could not recover the retained bit order"

X_fit, y_fit = oversample(X[~test_mask], y[~test_mask], SEED)
model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X_fit, y_fit)

annotated = pd.read_csv(DATA / "external_bbbp_annotated.csv")
feats, labels, keep = [], [], []
for _, row in annotated.iterrows():
    mol = Chem.MolFromSmiles(row["canonical"])
    if mol is None:
        continue
    bits = np.array(MACCSkeys.GenMACCSKeys(mol))
    feats.append(bits[retained])
    labels.append(int(row["label"]))
    keep.append(row)
feats = np.vstack(feats).astype(float)
labels = np.asarray(labels)
kept = pd.DataFrame(keep).reset_index(drop=True)

prob = model.predict_proba(feats)[:, 1]
pred = (prob >= 0.5).astype(int)

genuinely_external = (~kept["shared_scaffold"].to_numpy()) & (~kept["identical_structure"].to_numpy())
in_training = kept["identical_structure"].to_numpy()

report = {
    "as_published_whole_set": metrics(labels, pred, prob),
    "subset_already_in_training": metrics(
        labels[in_training], pred[in_training], prob[in_training]),
    "genuinely_external": metrics(
        labels[genuinely_external], pred[genuinely_external], prob[genuinely_external]),
}

print(json.dumps(report, indent=2))
(RESULTS / "external_benchmark.json").write_text(json.dumps(report, indent=2))

kept.assign(pred=pred, prob=prob.round(4)).to_csv(
    RESULTS / "external_predictions.csv", index=False)
print(f"\nwrote {RESULTS / 'external_benchmark.json'}")
