"""Choose example compounds that span the scope bands.

The default example was four familiar drugs, all of which sit inside the
training domain, so the feature the tool is named after never appeared on first
use. These candidates are scored so the example can show the full range.
"""
from __future__ import annotations

import gzip
import pickle
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

MODELS = Path(__file__).resolve().parent / "repo" / "models"

CANDIDATES = {
    "caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "sucrose": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
    "diazepam": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
    "perfluorooctanol": "OCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "cyclic tripeptide": "O=C1NC(Cc2ccccc2)C(=O)NC(CC(C)C)C(=O)NC1CO",
    "crown ether": "C1COCCOCCOCCOCCOCCO1",
    "adamantane amine": "NC12CC3CC(CC(C3)C1)C2",
    "ferrocene-like cage": "C1CC2CCC1(CC2)C1CCC2(CC1)CCCC2",
    "long alkyl sulfonate": "CCCCCCCCCCCCCCCCCCS(=O)(=O)O",
    "porphyrin fragment": "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3",
}

columns = np.load(MODELS / "columns_maccs.npy")
with gzip.open(MODELS / "rf_maccs.pkl.gz", "rb") as fh:
    model = pickle.load(fh)

bits = np.unpackbits(np.load(MODELS / "ad_train_morgan.npy"), axis=1)[:, :2048]
reference = []
for row in bits:
    vector = DataStructs.ExplicitBitVect(2048)
    vector.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
    reference.append(vector)


def scope(similarity: float) -> str:
    if similarity >= 0.5:
        return "In scope"
    return "Edge of scope" if similarity >= 0.3 else "Out of scope"


print(f"{'compound':<22}{'P(BBB+)':>9}{'similarity':>12}  scope")
print("-" * 62)
for name, smiles in CANDIDATES.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{name:<22}  unparseable")
        continue
    features = np.array(MACCSkeys.GenMACCSKeys(mol))[columns].reshape(1, -1)
    probability = float(model.predict_proba(features.astype(float))[0, 1])
    query = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    similarity = max(DataStructs.BulkTanimotoSimilarity(query, reference))
    print(f"{name:<22}{probability:>9.3f}{similarity:>12.3f}  {scope(similarity)}")
