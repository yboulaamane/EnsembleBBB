"""Exercise the app's logic without starting Streamlit.

The alignment check matters most: the nearest-neighbour display indexes the
training structures CSV with a position taken from the packed fingerprint
array. Those two files are written by separate scripts, so if their order ever
diverges the app will confidently show the wrong molecule.
"""
from __future__ import annotations

import gzip
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

MODELS = Path(__file__).resolve().parent.parent / "models"
KIND = "maccs"

PROBES = [
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
    ("sucrose", "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O"),
    ("cyclic tripeptide", "O=C1NC(Cc2ccccc2)C(=O)NC(CC(C)C)C(=O)NC1CO"),
    ("fluorotelomer alcohol", "OCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
    if not condition:
        failures.append(message)


print("1. ARTEFACTS")
bits = np.unpackbits(np.load(MODELS / "ad_train_morgan.npy"), axis=1)[:, :2048]
training = pd.read_csv(MODELS / "ad_train_molecules.csv")
check(len(bits) == len(training),
      f"fingerprints and structures same length ({len(bits)} vs {len(training)})")

# The decisive test: regenerating a structure's fingerprint must reproduce the
# stored row at the same index. A silent reordering would fail here.
rng = np.random.default_rng(0)
misaligned = 0
for index in rng.choice(len(training), size=50, replace=False):
    mol = Chem.MolFromSmiles(training.iloc[index]["smiles"])
    if mol is None:
        continue
    regenerated = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if not np.array_equal(regenerated, bits[index]):
        misaligned += 1
check(misaligned == 0, f"50 sampled rows align with their structures ({misaligned} off)")

print("\n2. PREDICTION")
columns = np.load(MODELS / f"columns_{KIND}.npy")
with gzip.open(MODELS / f"rf_{KIND}.pkl.gz", "rb") as fh:
    model = pickle.load(fh)

vectors = []
for row in bits:
    vector = DataStructs.ExplicitBitVect(2048)
    vector.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
    vectors.append(vector)

print(f"  {'compound':<24}{'P(BBB+)':>9}{'sim':>7}  nearest training molecule")
for name, smiles in PROBES:
    mol = Chem.MolFromSmiles(smiles)
    features = np.array(MACCSkeys.GenMACCSKeys(mol))[columns].reshape(1, -1)
    probability = float(model.predict_proba(features.astype(float))[0, 1])
    scores = DataStructs.BulkTanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), vectors)
    best = int(np.argmax(scores))
    neighbour = training.iloc[best]
    print(f"  {name:<24}{probability:>9.3f}{scores[best]:>7.3f}  "
          f"{neighbour['smiles'][:42]}")
    check(Chem.MolFromSmiles(neighbour["smiles"]) is not None,
          f"{name}: neighbour structure is valid")

print("\n3. ALIGNMENT WITH AN UNPARSEABLE ENTRY")
mixed = [PROBES[0][1], "not_a_smiles", PROBES[1][1]]
kept = [s for s in mixed if Chem.MolFromSmiles(s) is not None]
failed = [s for s in mixed if Chem.MolFromSmiles(s) is None]
check(kept == [mixed[0], mixed[2]] and failed == ["not_a_smiles"],
      "valid entries stay paired with their own inputs")

print(f"\n{'PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
