"""Train the published models under three validation protocols.

  A  original : oversample the whole set, then split at random  (what the paper did)
  B  random   : split at random, oversample inside training folds only
  C  scaffold : split by Murcko scaffold, oversample inside training folds only

Hyperparameters are the ones recovered from the published pickles, so the only
thing that changes between protocols is the validation discipline.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "revision" / "data"
RESULTS = PROJECT / "revision" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
FINGERPRINTS = ["morgan", "maccs", "avalon", "topological_torsion"]
N_FOLDS = 10


def build_models() -> dict:
    """Exactly the hyperparameters recovered from the published .pkl.gz files."""
    return {
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=2),
        "SVC": SVC(C=10, gamma=0.1, kernel="rbf", probability=True, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "XGB": XGBClassifier(
            n_estimators=100,
            objective="binary:logistic",
            random_state=SEED,
            n_jobs=2,
            eval_metric="logloss",
        ),
    }


def build_ensemble() -> VotingClassifier:
    return VotingClassifier(
        estimators=list(build_models().items()), voting="soft", n_jobs=1
    )


def oversample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random oversampling with replacement, matching the original pipeline."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    target = counts.max()
    parts_X, parts_y = [X], [y]
    for cls, count in enumerate(counts):
        if count < target:
            idx = np.where(y == cls)[0]
            extra = rng.choice(idx, size=target - count, replace=True)
            parts_X.append(X[extra])
            parts_y.append(y[extra])
    return np.vstack(parts_X), np.concatenate(parts_y)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return {
        "SE": se,
        "SP": sp,
        "Q+": tp / (tp + fp) if tp + fp else 0.0,
        "Q-": tn / (tn + fn) if tn + fn else 0.0,
        "ACC": accuracy_score(y_true, y_pred),
        "BACC": (se + sp) / 2,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "g_mean": float(np.sqrt(se * sp)),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def run_one(protocol: str, fingerprint: str) -> list[dict]:
    if protocol == "A":
        X = np.load(DATA / f"X_all_{fingerprint}.npy")
        y = np.load(DATA / "labels_all.npy")
        # The original error: resample everything, then split.
        X, y = oversample(X, y, SEED)
        rng = np.random.default_rng(SEED)
        test_idx = rng.choice(len(y), size=int(0.2 * len(y)), replace=False)
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[test_idx] = True
    else:
        X = np.load(DATA / f"X_dedup_{fingerprint}.npy")
        y = np.load(DATA / "labels_dedup.npy")
        if protocol == "C":
            test_mask = np.load(DATA / "scaffold_test_mask.npy")
        else:
            rng = np.random.default_rng(SEED)
            n_test = int(np.load(DATA / "scaffold_test_mask.npy").sum())
            test_mask = np.zeros(len(y), dtype=bool)
            test_mask[rng.choice(len(y), size=n_test, replace=False)] = True

    X_train, y_train = X[~test_mask], y[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Protocol A has already been resampled; B and C resample training data only.
    if protocol == "A":
        X_fit, y_fit = X_train, y_train
    else:
        X_fit, y_fit = oversample(X_train, y_train, SEED)

    rows = []
    candidates = {**build_models(), "Ensemble": build_ensemble()}
    held_out_probs = {}

    for name, model in candidates.items():
        model.fit(X_fit, y_fit)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        held_out_probs[name] = prob

        cv_aucs = []
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for tr, va in splitter.split(X_train, y_train):
            if protocol == "A":
                Xa, ya = X_train[tr], y_train[tr]
            else:
                Xa, ya = oversample(X_train[tr], y_train[tr], SEED)
            fold_model = (
                build_ensemble() if name == "Ensemble" else build_models()[name]
            )
            fold_model.fit(Xa, ya)
            cv_aucs.append(
                roc_auc_score(y_train[va], fold_model.predict_proba(X_train[va])[:, 1])
            )

        row = {
            "protocol": protocol,
            "fingerprint": fingerprint,
            "model": name,
            **metrics(y_test, pred, prob),
            "mean_cv_auc": float(np.mean(cv_aucs)),
            "std_cv_auc": float(np.std(cv_aucs)),
        }
        rows.append(row)
        print(
            f"[{protocol}/{fingerprint}] {name:>9}  "
            f"MCC {row['MCC']:.3f}  AUC {row['AUC']:.3f}  CV {row['mean_cv_auc']:.3f}",
            flush=True,
        )

    np.savez(
        RESULTS / f"probs_{protocol}_{fingerprint}.npz",
        y_test=y_test,
        **held_out_probs,
    )
    return rows


def main() -> None:
    jobs = [(p, f) for p in ["A", "B", "C"] for f in FINGERPRINTS]
    if len(sys.argv) > 1:  # allow a single protocol for a quick check
        jobs = [j for j in jobs if j[0] in sys.argv[1]]

    results = Parallel(n_jobs=6, verbose=10)(delayed(run_one)(p, f) for p, f in jobs)
    frame = pd.DataFrame([row for group in results for row in group])
    frame.to_csv(RESULTS / "metrics.csv", index=False)
    print(f"\nwrote {RESULTS / 'metrics.csv'}  ({len(frame)} rows)")


if __name__ == "__main__":
    main()
