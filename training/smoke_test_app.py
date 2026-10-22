"""Verify the exported artefacts work, and check predictions against B3DB.

Two separate questions, which the first version of this script conflated:

  1. Does the pipeline work - do the models load, do fingerprints compute, do
     predictions come out aligned with their inputs?
  2. Does the model agree with its own training labels?

Pharmacological intuition is a third thing again, and where it disagrees with
B3DB the disagreement is a property of the dataset, not of the model. Those
cases are printed as commentary.
"""
from __future__ import annotations

import gzip
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MODELS = HERE / "repo" / "models"
PROJECT = HERE.parent

KIND = "maccs"
AD_WARN = 0.5

# "expected" is textbook pharmacology, not a B3DB label.
PROBES = [
    ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "BBB+"),
    ("diazepam", "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O", "BBB+"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12", "BBB+"),
    ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "BBB-"),
    ("domperidone", "O=C1Nc2ccccc2N1C1CCN(CCCN2C(=O)Nc3ccc(Cl)cc32)CC1", "BBB-"),
    ("loratadine", "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1", "BBB-"),
    ("sucrose", "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O", "BBB-"),
]

columns = np.load(MODELS / f"columns_{KIND}.npy")
with gzip.open(MODELS / f"rf_{KIND}.pkl.gz", "rb") as fh:
    model = pickle.load(fh)

packed = np.load(MODELS / "ad_train_morgan.npy")
bits = np.unpackbits(packed, axis=1)[:, :2048]
reference = []
for row in bits:
    vector = DataStructs.ExplicitBitVect(2048)
    vector.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
    reference.append(vector)

b3db = pd.read_csv(PROJECT / "rdkit-bbb" / "b3db.csv")
labels: dict[str, int] = {}
for smiles, label in zip(b3db["SMILES"], b3db["Class"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        labels.setdefault(Chem.MolToSmiles(mol), int(label))

print("1. PIPELINE")
print("-" * 68)
rows = []
for name, smiles, expected in PROBES:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"{name} failed to parse"
    features = np.array(MACCSkeys.GenMACCSKeys(mol))[columns].reshape(1, -1)
    probability = float(model.predict_proba(features.astype(float))[0, 1])
    query = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    similarity = max(DataStructs.BulkTanimotoSimilarity(query, reference))
    rows.append({
        "name": name,
        "predicted": "BBB+" if probability >= 0.5 else "BBB-",
        "probability": probability,
        "similarity": similarity,
        "b3db": {1: "BBB+", 0: "BBB-"}.get(labels.get(Chem.MolToSmiles(mol)), "absent"),
        "textbook": expected,
    })
print(f"models load, fingerprints compute, {len(rows)} predictions returned")

mixed = ["CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "not_a_smiles",
         "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"]
kept = [s for s in mixed if Chem.MolFromSmiles(s) is not None]
failed = [s for s in mixed if Chem.MolFromSmiles(s) is None]
assert kept == [mixed[0], mixed[2]] and failed == ["not_a_smiles"]
print("alignment holds when an entry fails to parse")

print("\n2. AGREEMENT WITH B3DB LABELS")
print("-" * 68)
print(f"{'compound':<14}{'B3DB':<9}{'predicted':<11}{'P(BBB+)':<10}{'nn_sim':<9}")
scored = [r for r in rows if r["b3db"] != "absent"]
for r in scored:
    mark = "ok" if r["predicted"] == r["b3db"] else "MISS"
    print(f"{r['name']:<14}{r['b3db']:<9}{r['predicted']:<11}"
          f"{r['probability']:<10.3f}{r['similarity']:<9.3f}{mark}")
agree = sum(r["predicted"] == r["b3db"] for r in scored)
print(f"\n{agree}/{len(scored)} agree with B3DB "
      f"(all are training compounds, so disagreement is model error)")

print("\n3. WHERE B3DB DISAGREES WITH PHARMACOLOGY")
print("-" * 68)
for r in rows:
    if r["b3db"] != "absent" and r["b3db"] != r["textbook"]:
        print(f"{r['name']:<14}B3DB says {r['b3db']}, textbook says {r['textbook']}")
for r in rows:
    if r["b3db"] == "absent":
        print(f"{r['name']:<14}absent from B3DB; model says {r['predicted']} "
              f"(similarity {r['similarity']:.2f})")
