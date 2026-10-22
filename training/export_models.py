"""Train and export the corrected models for the web application.

Models are trained on the scaffold-split training partition only, with
oversampling applied inside that partition. The random forest is exported
because it is the best model under corrected validation; the soft-voting
consensus is exported alongside it so that the paper's negative result can be
checked in the application itself.

Also exports the packed training fingerprints the app needs to compute each
prediction's distance to the applicability domain.
"""
from __future__ import annotations

import gzip
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "repo" / "models"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

FP_BITS = {"morgan": 2048, "maccs": 167, "avalon": 1024, "topological_torsion": 2048}


def fingerprint(mol, kind):
    if kind == "morgan":
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if kind == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if kind == "avalon":
        return np.array(GetAvalonFP(mol, nBits=1024))
    return np.array(GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048))


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


def members():
    return [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
        ("svc", SVC(C=10, gamma=0.1, probability=True, random_state=SEED)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("xgb", XGBClassifier(n_estimators=100, objective="binary:logistic",
                              random_state=SEED, n_jobs=-1, eval_metric="logloss")),
    ]


molecules = pd.read_csv(DATA / "molecules_dedup.csv")
labels = np.load(DATA / "labels_dedup.npy")
test_mask = np.load(DATA / "scaffold_test_mask.npy")
mols = [Chem.MolFromSmiles(s) for s in molecules["canonical"]]

manifest = {
    "trained_on": "B3DB, scaffold-split training partition only",
    "n_training": int((~test_mask).sum()),
    "n_held_out": int(test_mask.sum()),
    "resampling": "random oversampling of the minority class, training partition only",
    "seed": SEED,
    "fingerprints": {},
}

for kind, nbits in FP_BITS.items():
    full = np.vstack([fingerprint(m, kind) for m in mols])

    # Match the feature selection used in the paper: drop constant columns,
    # then one of each pair correlated above 0.7.
    keep = full.std(axis=0) > 0
    cols = np.where(keep)[0]
    frame = pd.DataFrame(full[:, cols])
    corr = frame.corr().abs().to_numpy()
    drop = {cols[j] for i, j in zip(*np.where(np.triu(corr, k=1) > 0.7))}
    cols = np.array([c for c in cols if c not in drop])

    X = full[:, cols]
    X_train, y_train = X[~test_mask], labels[~test_mask]
    X_fit, y_fit = oversample(X_train, y_train, SEED)

    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_fit, y_fit)
    with gzip.open(OUT / f"rf_{kind}.pkl.gz", "wb") as fh:
        pickle.dump(rf, fh, protocol=4)

    consensus = VotingClassifier(estimators=members(), voting="soft", n_jobs=1)
    consensus.fit(X_fit, y_fit)
    with gzip.open(OUT / f"consensus_{kind}.pkl.gz", "wb") as fh:
        pickle.dump(consensus, fh, protocol=4)

    np.save(OUT / f"columns_{kind}.npy", cols)
    manifest["fingerprints"][kind] = {
        "n_bits": int(nbits),
        "n_retained": int(len(cols)),
    }
    print(f"{kind:>20}: {nbits} bits -> {len(cols)} retained; rf + consensus exported")

# Packed training Morgan fingerprints, for the applicability domain check.
train_morgan = np.vstack([fingerprint(m, "morgan") for m, t in zip(mols, test_mask) if not t])
np.save(OUT / "ad_train_morgan.npy", np.packbits(train_morgan.astype(np.uint8), axis=1))
manifest["applicability_domain"] = {
    "representation": "Morgan ECFP4, 2048 bits, packed",
    "n_reference": int(train_morgan.shape[0]),
    "metric": "maximum Tanimoto similarity to the training partition",
    "warn_below": 0.5,
    "note": "accuracy on held-out compounds falls from 94 % above 0.5 to 76 % below 0.3",
}

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nwrote {OUT}")
