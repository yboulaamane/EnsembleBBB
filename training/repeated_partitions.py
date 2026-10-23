"""Confidence intervals over repeated scaffold partitions.

The corrected figures elsewhere come from one partition, so they carry sampling
uncertainty that a single number hides. This repeats the partition ten times,
varying only which scaffold groups fall in the test set, and reports the spread.

Two questions are answered. Are the corrected performance figures stable across
partitions? And does the finding that consensus modelling confers no advantage
survive repetition, or was it an artefact of one partition?

Scope: the random forest across all four fingerprints, plus all five models on
MACCS keys. The support vector classifier is tractable on 118 MACCS features
but not on 1 951 Morgan ones, where it previously put a five-model run beyond
ten hours.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

N_REPEATS = 10
TEST_FRACTION = 0.2
FINGERPRINTS = ["morgan", "maccs", "avalon", "topological_torsion"]
MODEL_ORDER = ["RF", "SVC", "kNN", "XGB", "Consensus"]


def scaffold_partition(scaffolds: np.ndarray, seed: int) -> np.ndarray:
    """Assign whole scaffold groups to test until the target size is reached.

    The largest group always stays in training; the rest are shuffled per seed,
    which is what varies between repeats.
    """
    groups: dict[str, list[int]] = {}
    for index, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(index)
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    head, tail = ordered[:1], ordered[1:]

    rng = np.random.default_rng(seed)
    rng.shuffle(tail)

    target = int(round(TEST_FRACTION * len(scaffolds)))
    mask = np.zeros(len(scaffolds), dtype=bool)
    n_test = 0
    for _, indices in head + tail:
        if n_test >= target:
            break
        if len(indices) + n_test <= target:
            mask[indices] = True
            n_test += len(indices)
    return mask


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


def make_model(name: str, seed: int):
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2)
    if name == "SVC":
        return SVC(C=10, gamma=0.1, probability=True, random_state=seed)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, objective="binary:logistic",
                             random_state=seed, n_jobs=2, eval_metric="logloss")
    return VotingClassifier(
        estimators=[(n, make_model(n, seed)) for n in ("RF", "SVC", "kNN", "XGB")],
        voting="soft", n_jobs=1)


def score(y_true, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return {"SE": se, "SP": sp, "ACC": accuracy_score(y_true, pred),
            "BACC": (se + sp) / 2, "MCC": matthews_corrcoef(y_true, pred),
            "AUC": roc_auc_score(y_true, prob)}


def run_repeat(seed: int, scaffolds: np.ndarray, y: np.ndarray,
               features: dict[str, np.ndarray]) -> list[dict]:
    mask = scaffold_partition(scaffolds, seed)
    rows = []
    for fingerprint, X in features.items():
        # All five models on MACCS; the random forest alone elsewhere.
        names = MODEL_ORDER if fingerprint == "maccs" else ["RF"]
        X_fit, y_fit = oversample(X[~mask], y[~mask], seed)
        for name in names:
            model = make_model(name, seed)
            model.fit(X_fit, y_fit)
            prob = model.predict_proba(X[mask])[:, 1]
            rows.append({"seed": seed, "fingerprint": fingerprint, "model": name,
                         "n_test": int(mask.sum()), **score(y[mask], prob)})
    print(f"  partition {seed}: {int(mask.sum())} test compounds", flush=True)
    return rows


def interval(values: np.ndarray) -> tuple[float, float, float]:
    """Mean with a percentile interval; n = 10 is too small to assume normality."""
    return (float(np.mean(values)),
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)))


def main() -> None:
    molecules = pd.read_csv(DATA / "molecules_dedup.csv")
    scaffolds = molecules["scaffold"].to_numpy()
    y = np.load(DATA / "labels_dedup.npy")
    features = {f: np.load(DATA / f"X_dedup_{f}.npy") for f in FINGERPRINTS}
    print(f"{len(y)} molecules, {len(set(scaffolds))} scaffolds, "
          f"{N_REPEATS} partitions\n", flush=True)

    batches = Parallel(n_jobs=5, verbose=5)(
        delayed(run_repeat)(seed, scaffolds, y, features) for seed in range(N_REPEATS))
    frame = pd.DataFrame([r for batch in batches for r in batch])
    frame.to_csv(RESULTS / "repeated_partitions_raw.csv", index=False)

    summary = []
    for (fingerprint, model), block in frame.groupby(["fingerprint", "model"]):
        row = {"fingerprint": fingerprint, "model": model, "n_partitions": len(block)}
        for metric in ("MCC", "AUC", "ACC", "BACC"):
            mean, low, high = interval(block[metric].to_numpy())
            row[f"{metric}_mean"] = round(mean, 3)
            row[f"{metric}_lo"] = round(low, 3)
            row[f"{metric}_hi"] = round(high, 3)
            row[f"{metric}_sd"] = round(float(block[metric].std(ddof=1)), 3)
        summary.append(row)
    table = pd.DataFrame(summary)
    table.to_csv(RESULTS / "repeated_partitions_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("RANDOM FOREST ACROSS PARTITIONS")
    print("=" * 78)
    rf = table[table.model == "RF"]
    for _, r in rf.iterrows():
        print(f"{r['fingerprint']:>20}  MCC {r['MCC_mean']:.3f} "
              f"[{r['MCC_lo']:.3f}, {r['MCC_hi']:.3f}]  sd {r['MCC_sd']:.3f}   "
              f"AUC {r['AUC_mean']:.3f} [{r['AUC_lo']:.3f}, {r['AUC_hi']:.3f}]")

    print("\n" + "=" * 78)
    print("CONSENSUS AGAINST ITS BEST MEMBER, MACCS, PER PARTITION")
    print("=" * 78)
    maccs = frame[frame.fingerprint == "maccs"]
    deltas = []
    for seed, block in maccs.groupby("seed"):
        singles = block[block.model != "Consensus"]
        consensus = block[block.model == "Consensus"].iloc[0]
        best = singles.loc[singles["MCC"].idxmax()]
        deltas.append(consensus["MCC"] - best["MCC"])
        print(f"  partition {seed}: best {best['model']:<4} {best['MCC']:.3f}  "
              f"consensus {consensus['MCC']:.3f}  delta {deltas[-1]:+.3f}")
    deltas = np.asarray(deltas)
    mean, low, high = interval(deltas)
    print(f"\nmean delta MCC {mean:+.3f}  [{low:+.3f}, {high:+.3f}]")
    print(f"consensus wins {int((deltas > 0).sum())} of {len(deltas)} partitions")

    report = {
        "n_partitions": N_REPEATS,
        "test_fraction": TEST_FRACTION,
        "random_forest_maccs": {
            metric: [rf[rf.fingerprint == "maccs"][f"{metric}_{s}"].iloc[0]
                     for s in ("mean", "lo", "hi")]
            for metric in ("MCC", "AUC", "ACC", "BACC")},
        "consensus_minus_best_maccs": {
            "mean": round(mean, 3), "ci": [round(low, 3), round(high, 3)],
            "wins": int((deltas > 0).sum()), "of": len(deltas)},
    }
    (RESULTS / "repeated_partitions.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RESULTS / 'repeated_partitions_summary.csv'}")


if __name__ == "__main__":
    main()
