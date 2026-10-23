"""Would a comparison against B3clf actually be fair?

B3clf is published by the same group as B3DB and is trained on it, so any
external compound that appears anywhere in B3DB is training data for B3clf even
when it is external to our model. Measure that before running the comparison.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = HERE / "data"

b3db = pd.read_csv(PROJECT / "rdkit-bbb" / "b3db.csv")
b3db_canon, b3db_scaffolds = set(), set()
for smi in b3db["SMILES"]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    b3db_canon.add(Chem.MolToSmiles(mol))
    b3db_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))

clean = pd.read_csv(DATA / "external_bbbp_clean.csv")
in_b3db = clean["canonical"].isin(b3db_canon)
scaffold_in_b3db = clean["scaffold"].isin(b3db_scaffolds)

report = {
    "clean_external_size": int(len(clean)),
    "also_in_full_B3DB": int(in_b3db.sum()),
    "scaffold_in_full_B3DB": int(scaffold_in_b3db.sum()),
    "pct_in_full_B3DB": round(100 * float(in_b3db.mean()), 2),
    "fair_for_B3clf": int((~in_b3db).sum()),
}

doubly_clean = clean[~in_b3db & ~scaffold_in_b3db].reset_index(drop=True)
report["doubly_clean_size"] = int(len(doubly_clean))
report["doubly_clean_balance"] = {
    "BBB+": int((doubly_clean["label"] == 1).sum()),
    "BBB-": int((doubly_clean["label"] == 0).sum()),
}

print(json.dumps(report, indent=2))
doubly_clean.to_csv(DATA / "external_bbbp_doubly_clean.csv", index=False)
(DATA / "b3clf_fairness.json").write_text(json.dumps(report, indent=2))
