"""Does the chirality fix cost accuracy? Random forest only.

The full five-model run was abandoned: the support vector classifier needs
eleven fits per protocol on 1 951 features with probability estimates enabled,
and the consensus refits it eleven times more, which put the run at four to ten
hours for columns the manuscript does not use. The claim concerns the
recommended model, so this reproduces that comparison alone and writes it to a
file rather than leaving the numbers in a log.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
SEED = 42
N_FOLDS = 10


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
        "SE": se, "SP": sp,
        "ACC": accuracy_score(y_true, y_pred),
        "BACC": (se + sp) / 2,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def evaluate(protocol, X, y, scaffold_mask):
    if protocol == "A":
        Xa, ya = oversample(X, y, SEED)
        rng = np.random.default_rng(SEED)
        mask = np.zeros(len(ya), dtype=bool)
        mask[rng.choice(len(ya), size=int(0.2 * len(ya)), replace=False)] = True
        X_train, y_train, X_test, y_test = Xa[~mask], ya[~mask], Xa[mask], ya[mask]
        X_fit, y_fit = X_train, y_train
    else:
        if protocol == "C":
            mask = scaffold_mask
        else:
            rng = np.random.default_rng(SEED)
            mask = np.zeros(len(y), dtype=bool)
            mask[rng.choice(len(y), size=int(scaffold_mask.sum()), replace=False)] = True
        X_train, y_train, X_test, y_test = X[~mask], y[~mask], X[mask], y[mask]
        X_fit, y_fit = oversample(X_train, y_train, SEED)

    model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X_fit, y_fit)
    prob = model.predict_proba(X_test)[:, 1]

    cv = []
    for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X_train, y_train):
        Xb, yb = ((X_train[tr], y_train[tr]) if protocol == "A"
                  else oversample(X_train[tr], y_train[tr], SEED))
        fold = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        fold.fit(Xb, yb)
        cv.append(roc_auc_score(y_train[va], fold.predict_proba(X_train[va])[:, 1]))

    return {**metrics(y_test, (prob >= 0.5).astype(int), prob),
            "mean_cv_auc": float(np.mean(cv)), "std_cv_auc": float(np.std(cv))}


def main() -> None:
    y = np.load(DATA / "labels_dedup.npy")
    scaffold_mask = np.load(DATA / "scaffold_test_mask.npy")
    X = np.load(DATA / "X_dedup_morgan_chiral.npy")
    blind = pd.read_csv(RESULTS / "metrics.csv")
    blind = blind[(blind.fingerprint == "morgan") & (blind.model == "RF")]
    print(f"chirality-aware Morgan: {X.shape[1]} retained descriptors "
          f"(blind variant retained 1951)\n", flush=True)

    rows = []
    for protocol in ["A", "B", "C"]:
        chiral = evaluate(protocol, X, y, scaffold_mask)
        reference = blind[blind.protocol == protocol].iloc[0]
        rows.append({
            "protocol": protocol,
            "blind_MCC": round(reference.MCC, 3), "chiral_MCC": round(chiral["MCC"], 3),
            "delta_MCC": round(chiral["MCC"] - reference.MCC, 3),
            "blind_AUC": round(reference.AUC, 3), "chiral_AUC": round(chiral["AUC"], 3),
            "delta_AUC": round(chiral["AUC"] - reference.AUC, 3),
            "chiral_ACC": round(chiral["ACC"], 3),
            "chiral_BACC": round(chiral["BACC"], 3),
            "chiral_cv_auc": round(chiral["mean_cv_auc"], 3),
        })
        print(f"[{protocol}] blind MCC {reference.MCC:.3f} -> chiral {chiral['MCC']:.3f} "
              f"({rows[-1]['delta_MCC']:+.3f});  AUC {reference.AUC:.3f} -> "
              f"{chiral['AUC']:.3f} ({rows[-1]['delta_AUC']:+.3f})", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "table_chirality_rf.csv", index=False)
    print("\n" + frame.to_string(index=False))

    summary = {
        "model": "random forest, 100 trees",
        "n_descriptors_chiral": int(X.shape[1]),
        "n_descriptors_blind": 1951,
        "note": ("leakage under random partitioning falls from 63.3 % to 1.4 % "
                 "with chirality enabled; this table gives the accuracy cost"),
        "scaffold_split_delta": {
            "MCC": frame[frame.protocol == "C"].delta_MCC.iloc[0],
            "AUC": frame[frame.protocol == "C"].delta_AUC.iloc[0],
        },
    }
    (RESULTS / "chirality_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {RESULTS / 'table_chirality_rf.csv'}")


if __name__ == "__main__":
    main()
