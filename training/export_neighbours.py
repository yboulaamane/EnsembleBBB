"""Ship the training structures so the app can name the nearest neighbour.

The scope is a number until you can see what it is close to. Exporting the
training SMILES and labels lets the app show the molecule a prediction is
leaning on, which is what makes the domain concrete rather than abstract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "repo" / "models"

molecules = pd.read_csv(DATA / "molecules_dedup.csv")
labels = np.load(DATA / "labels_dedup.npy")
mask = np.load(DATA / "scaffold_test_mask.npy")

training = pd.DataFrame({
    "smiles": molecules.loc[~mask, "canonical"].to_numpy(),
    "label": labels[~mask],
})
assert len(training) == int((~mask).sum()), "row count does not match the partition"

# The app indexes this against ad_train_morgan.npy, which is written in the
# same order by export_models.py, so the two must stay aligned.
training.to_csv(OUT / "ad_train_molecules.csv", index=False)
print(f"wrote {OUT / 'ad_train_molecules.csv'}  ({len(training)} molecules, "
      f"{int(training.label.sum())} BBB+ / {int((1 - training.label).sum())} BBB-)")
