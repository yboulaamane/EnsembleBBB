"""Build fingerprints, quantify redundancy, and define the splits.

Three protocols are prepared so the corrected manuscript can show what the
original one measured and what an honest one measures:

  A  original : oversample the whole set, then split at random
  B  random   : deduplicate, split at random, oversample inside folds only
  C  scaffold : deduplicate, split by Murcko scaffold, oversample in-fold only
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "revision" / "data"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
CORRELATION_CUTOFF = 0.7


def fingerprints(mol):
    """The four fingerprints used in the published study."""
    return {
        "morgan": np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)),
        "maccs": np.array(MACCSkeys.GenMACCSKeys(mol)),
        "avalon": np.array(GetAvalonFP(mol, nBits=1024)),
        "topological_torsion": np.array(
            GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
        ),
    }


def load_molecules() -> pd.DataFrame:
    raw = pd.read_csv(PROJECT / "rdkit-bbb" / "b3db.csv")
    records = []
    for smiles, label in zip(raw["SMILES"], raw["Class"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        records.append(
            {
                "smiles": smiles,
                "canonical": Chem.MolToSmiles(mol),
                "label": int(label),
                "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
                "mol": mol,
            }
        )
    frame = pd.DataFrame(records)
    print(f"parsed {len(frame)} / {len(raw)} molecules")
    return frame


def drop_duplicate_molecules(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove repeated structures, and any structure carrying both labels."""
    before = len(frame)
    label_spread = frame.groupby("canonical")["label"].nunique()
    contradictory = set(label_spread[label_spread > 1].index)
    frame = frame[~frame["canonical"].isin(contradictory)]
    frame = frame.drop_duplicates(subset="canonical", keep="first").reset_index(drop=True)
    stats = {
        "input_molecules": before,
        "contradictory_structures_removed": len(contradictory),
        "duplicate_structures_removed": before - len(frame) - len(contradictory),
        "unique_molecules": len(frame),
    }
    print(json.dumps(stats, indent=2))
    return frame, stats


def select_features(matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Variance filter then correlation pruning, as described in the manuscript."""
    selector = VarianceThreshold(threshold=0.0)
    kept = selector.fit_transform(matrix)
    names = [n for n, keep in zip(names, selector.get_support()) if keep]

    frame = pd.DataFrame(kept, columns=names)
    corr = frame.corr().abs().to_numpy()
    upper = np.triu(corr, k=1)
    drop = {names[j] for i, j in zip(*np.where(upper > CORRELATION_CUTOFF))}
    names = [n for n in names if n not in drop]
    return frame[names].to_numpy(dtype=np.uint8), names


def scaffold_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> np.ndarray:
    """Assign whole scaffold groups to test, largest groups to train first."""
    groups = frame.groupby("scaffold").indices
    ordered = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    target = int(round(test_fraction * len(frame)))

    is_test = np.zeros(len(frame), dtype=bool)
    n_test = 0
    rng = np.random.default_rng(SEED)
    # Largest scaffold families stay in training; the rest are shuffled so the
    # test set is not simply the alphabetical tail of the singletons.
    head, tail = ordered[:1], ordered[1:]
    rng.shuffle(tail)
    for _, idx in head + tail:
        if n_test >= target:
            break
        if len(idx) + n_test <= target:
            is_test[idx] = True
            n_test += len(idx)
    return is_test


def main() -> None:
    frame = load_molecules()

    collisions = {}
    raw_matrices = {}
    for name in ["morgan", "maccs", "avalon", "topological_torsion"]:
        stacked = np.vstack([fingerprints(m)[name] for m in frame["mol"]])
        raw_matrices[name] = stacked
        unique = len(np.unique(stacked, axis=0))
        collisions[name] = {
            "molecules": int(len(stacked)),
            "unique_fingerprints": int(unique),
            "collision_rate": round(1 - unique / len(stacked), 4),
        }
        print(f"{name:>20}: {unique} unique / {len(stacked)} molecules")

    deduped, dedupe_stats = drop_duplicate_molecules(frame)
    keep_idx = frame.index.isin(
        frame.reset_index().merge(deduped[["canonical"]], on="canonical")["index"]
    )

    is_test = scaffold_split(deduped)
    print(f"\nscaffold split: {int((~is_test).sum())} train / {int(is_test.sum())} test")
    print(f"unique scaffolds: {deduped['scaffold'].nunique()}")

    np.save(OUT / "labels_all.npy", frame["label"].to_numpy())
    np.save(OUT / "labels_dedup.npy", deduped["label"].to_numpy())
    np.save(OUT / "scaffold_test_mask.npy", is_test)
    deduped.drop(columns="mol").to_csv(OUT / "molecules_dedup.csv", index=False)

    feature_report = {}
    for name, stacked in raw_matrices.items():
        names = [f"{name}_{i}" for i in range(stacked.shape[1])]
        selected_all, kept_names = select_features(stacked, names)
        np.save(OUT / f"X_all_{name}.npy", selected_all)
        np.save(OUT / f"X_dedup_{name}.npy", selected_all[keep_idx])
        feature_report[name] = {
            "raw_bits": int(stacked.shape[1]),
            "after_selection": int(selected_all.shape[1]),
        }
        print(f"{name:>20}: {stacked.shape[1]} bits -> {selected_all.shape[1]} retained")

    report = {
        "seed": SEED,
        "correlation_cutoff": CORRELATION_CUTOFF,
        "fingerprint_collisions": collisions,
        "deduplication": dedupe_stats,
        "feature_selection": feature_report,
        "scaffold_split": {
            "train": int((~is_test).sum()),
            "test": int(is_test.sum()),
            "unique_scaffolds": int(deduped["scaffold"].nunique()),
        },
    }
    (OUT / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'prepare_report.json'}")


if __name__ == "__main__":
    main()
