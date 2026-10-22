"""How much of the BBBP external set is already in our training data?

The external benchmark in the published version is only meaningful if the
external compounds are genuinely external. Given this paper's own findings
about near-duplicates, overlap has to be measured at three levels: identical
structure, identical fingerprint, and shared Murcko scaffold.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = HERE / "data"


def load(smiles_iter, labels):
    out = []
    for smi, label in zip(smiles_iter, labels):
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol is None:
            continue
        out.append({
            "canonical": Chem.MolToSmiles(mol),
            "label": int(label),
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
            "fp": AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
        })
    return pd.DataFrame(out)


# Our training partition.
train = pd.read_csv(DATA / "molecules_dedup.csv")
test_mask = np.load(DATA / "scaffold_test_mask.npy")
train = train[~test_mask].reset_index(drop=True)
train_mols = [Chem.MolFromSmiles(s) for s in train["canonical"]]
train_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in train_mols]
train_canon = set(train["canonical"])
train_scaffolds = set(train["scaffold"])
train_bits = {np.array(f).tobytes() for f in train_fps}

# The BBBP external set used in the published version.
bbbp_raw = pd.read_csv(PROJECT / "rdkit-bbb" / "Unique_bbbp.csv")
bbbp = load(bbbp_raw["Smiles"], bbbp_raw["Actual"])
print(f"BBBP external set: {len(bbbp)} parsed of {len(bbbp_raw)}")

# Despite the file name, the set is not unique: distinct SMILES strings
# canonicalise onto the same structure. Deduplicate, and drop any structure
# that carries both labels.
n_before = len(bbbp)
spread = bbbp.groupby("canonical")["label"].nunique()
contradictory = set(spread[spread > 1].index)
bbbp = bbbp[~bbbp["canonical"].isin(contradictory)]
bbbp = bbbp.drop_duplicates(subset="canonical", keep="first").reset_index(drop=True)
print(f"  duplicate structures removed: {n_before - len(bbbp) - len(contradictory)}")
print(f"  contradictory structures removed: {len(contradictory)}")
print(f"  unique external compounds: {len(bbbp)}")
print(f"  class balance: {int((bbbp['label'] == 1).sum())} BBB+ / "
      f"{int((bbbp['label'] == 0).sum())} BBB-")

same_structure = bbbp["canonical"].isin(train_canon)
same_fp = bbbp["fp"].map(lambda f: np.array(f).tobytes() in train_bits)
same_scaffold = bbbp["scaffold"].isin(train_scaffolds)

nn = np.array([max(DataStructs.BulkTanimotoSimilarity(f, train_fps)) for f in bbbp["fp"]])

report = {
    "raw_rows": int(len(bbbp_raw)),
    "duplicate_structures_removed": int(n_before - len(bbbp) - len(contradictory)),
    "contradictory_structures_removed": int(len(contradictory)),
    "external_size": int(len(bbbp)),
    "identical_structure_in_training": int(same_structure.sum()),
    "identical_fingerprint_in_training": int(same_fp.sum()),
    "shared_scaffold_with_training": int(same_scaffold.sum()),
    "pct_identical_structure": round(100 * float(same_structure.mean()), 2),
    "pct_identical_fingerprint": round(100 * float(same_fp.mean()), 2),
    "pct_shared_scaffold": round(100 * float(same_scaffold.mean()), 2),
    "median_nn_tanimoto": round(float(np.median(nn)), 3),
}

# A genuinely external subset: no shared scaffold, no duplicate structure.
clean = bbbp[~same_scaffold & ~same_structure].reset_index(drop=True)
report["clean_external_size"] = int(len(clean))
report["clean_class_balance"] = {
    "BBB+": int((clean["label"] == 1).sum()),
    "BBB-": int((clean["label"] == 0).sum()),
}
clean_nn = nn[(~same_scaffold & ~same_structure).to_numpy()]
report["clean_median_nn_tanimoto"] = round(float(np.median(clean_nn)), 3)

print(json.dumps(report, indent=2))

clean_out = clean[["canonical", "label", "scaffold"]].copy()
clean_out.to_csv(DATA / "external_bbbp_clean.csv", index=False)
bbbp[["canonical", "label", "scaffold"]].assign(
    identical_structure=same_structure.to_numpy(),
    identical_fingerprint=same_fp.to_numpy(),
    shared_scaffold=same_scaffold.to_numpy(),
    nn_tanimoto=nn,
).to_csv(DATA / "external_bbbp_annotated.csv", index=False)
(DATA / "external_overlap.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {DATA / 'external_bbbp_clean.csv'} ({len(clean)} compounds)")
