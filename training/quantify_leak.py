"""Quantify how much of a test set is already memorisable from training.

A test molecule leaks when some training molecule has an identical feature
vector. Compare a random split against the scaffold split, for all four
fingerprints, and check whether stereochemistry-aware fingerprints remove it.
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

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "revision" / "data"
SEED = 42

raw = pd.read_csv(PROJECT / "rdkit-bbb" / "b3db.csv")
mols, labels = [], []
for smiles, label in zip(raw["SMILES"], raw["Class"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mols.append(mol)
        labels.append(int(label))
labels = np.asarray(labels)

VARIANTS = {
    "morgan": lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048),
    "morgan_chiral": lambda m: AllChem.GetMorganFingerprintAsBitVect(
        m, 2, nBits=2048, useChirality=True
    ),
    "maccs": MACCSkeys.GenMACCSKeys,
    "avalon": lambda m: GetAvalonFP(m, nBits=1024),
    "topological_torsion": lambda m: GetHashedTopologicalTorsionFingerprintAsBitVect(
        m, nBits=2048
    ),
}

scaffolds = np.array(
    [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in mols]
)
test_mask_scaffold = np.load(DATA / "scaffold_test_mask.npy")

rng = np.random.default_rng(SEED)
test_mask_random = np.zeros(len(mols), dtype=bool)
test_mask_random[rng.choice(len(mols), size=int(test_mask_scaffold.sum()), replace=False)] = True


def leak_rate(matrix: np.ndarray, test_mask: np.ndarray) -> dict:
    train_rows = {row.tobytes() for row in matrix[~test_mask]}
    hits = [row.tobytes() in train_rows for row in matrix[test_mask]]
    # Of the leaked ones, how many carry the same label as their twin?
    return {
        "test_size": int(test_mask.sum()),
        "leaked": int(sum(hits)),
        "leak_rate": round(float(np.mean(hits)), 4),
    }


report = {}
for name, func in VARIANTS.items():
    matrix = np.vstack([np.array(func(m)) for m in mols])
    unique = len(np.unique(matrix, axis=0))
    report[name] = {
        "unique_fingerprints": int(unique),
        "collision_rate": round(1 - unique / len(matrix), 4),
        "random_split": leak_rate(matrix, test_mask_random),
        "scaffold_split": leak_rate(matrix, test_mask_scaffold),
    }
    r = report[name]
    print(
        f"{name:>20}  unique {unique:>5}  "
        f"random-split leak {r['random_split']['leak_rate']:.1%}  "
        f"scaffold-split leak {r['scaffold_split']['leak_rate']:.1%}"
    )

# Do stereoisomer families actually share a Murcko scaffold?
morgan = np.vstack([np.array(VARIANTS["morgan"](m)) for m in mols])
buckets: dict[bytes, list[int]] = {}
for i, row in enumerate(morgan):
    buckets.setdefault(row.tobytes(), []).append(i)
multi = [idx for idx in buckets.values() if len(idx) > 1]
same_scaffold = sum(len(set(scaffolds[idx])) == 1 for idx in multi)
same_label = sum(len(set(labels[idx])) == 1 for idx in multi)
print(f"\ncolliding groups: {len(multi)}")
print(f"  sharing one Murcko scaffold: {same_scaffold} ({same_scaffold / len(multi):.1%})")
print(f"  sharing one BBB label:       {same_label} ({same_label / len(multi):.1%})")

report["collision_groups"] = {
    "n_groups": len(multi),
    "share_scaffold": same_scaffold,
    "share_label": same_label,
}
(DATA / "leak_report.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {DATA / 'leak_report.json'}")
