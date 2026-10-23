"""Banner for the web application and the repository README.

Two things earlier versions got wrong and this one does not. The molecule is
drawn by RDKit from a SMILES whose InChIKey has been checked against the
published key, rather than sketched by hand. And every number shown is
computed here from the shipped models, so the banner cannot drift away from
what the app actually returns.
"""
from __future__ import annotations

import gzip
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.image as mpimg                                  # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Polygon, Rectangle)
from rdkit import Chem, DataStructs, RDLogger                     # noqa: E402
from rdkit.Chem import AllChem, MACCSkeys                         # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D                           # noqa: E402
from rdkit.Chem.inchi import MolToInchiKey                        # noqa: E402

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"

BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c2352f"
INK, MUTED, RULE = "#13233a", "#5a6b82", "#cfdcec"
FAINT = "#dfe7f2"
FONT = {"fontname": "Liberation Sans"}

# Olanzapine: a marketed CNS drug, and a structure busy enough to be worth
# drawing. The key is the published one, so a mistyped ring would not match.
NAME = "olanzapine"
SMILES = "CC1=CC2=C(S1)NC3=CC=CC=C3N=C2N4CCN(C)CC4"
INCHIKEY = "KVWDHTXUZHCGIO-UHFFFAOYSA-N"

# Bands as the app defines them, with the accuracy measured for each.
BANDS = [(0.0, 0.3, RED, "out of scope", "76 %"),
         (0.3, 0.5, ORANGE, "edge", "88 %"),
         (0.5, 1.0, GREEN, "in scope", "94 %")]


def predict(smiles: str):
    """Probability, nearest-neighbour similarity and the MACCS bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    assert MolToInchiKey(mol) == INCHIKEY, "structure is not the intended one"

    maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
    columns = np.load(MODELS / "columns_maccs.npy")
    with gzip.open(MODELS / "rf_maccs.pkl.gz", "rb") as fh:
        forest = pickle.load(fh)
    probability = float(forest.predict_proba(
        maccs[columns].reshape(1, -1).astype(float))[0, 1])

    packed = np.unpackbits(np.load(MODELS / "ad_train_morgan.npy"), axis=1)
    vectors = []
    for row in packed[:, :2048]:
        vector = DataStructs.ExplicitBitVect(2048)
        vector.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
        vectors.append(vector)
    similarity = max(DataStructs.BulkTanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), vectors))

    return probability, similarity, maccs


def render_molecule(path: Path) -> None:
    mol = Chem.MolFromSmiles(SMILES)
    drawer = rdMolDraw2D.MolDraw2DCairo(560, 420)
    options = drawer.drawOptions()
    options.bondLineWidth = 3
    options.clearBackground = False
    options.setAtomPalette({
        6: (0.07, 0.14, 0.23), 7: (0.16, 0.47, 0.84),
        8: (0.92, 0.41, 0.20), 16: (0.72, 0.52, 0.04),
        -1: (0.07, 0.14, 0.23),
    })
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


probability, similarity, maccs = predict(SMILES)

fig, ax = plt.subplots(figsize=(12.0, 3.5), dpi=150)
ax.set_xlim(0, 120)
ax.set_ylim(0, 35)
ax.axis("off")
fig.patch.set_facecolor("white")


def card(x, y, w, h, edge=RULE, lw=1.6, face="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))


def arrow(x0, x1, y, above=None, below=None):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=15,
        linewidth=1.6, color=MUTED, zorder=3))
    mid = (x0 + x1) / 2
    if above:
        ax.text(mid, y + 2.2, above, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=INK, **FONT)
    if below:
        ax.text(mid, y - 2.2, below, ha="center", va="top", fontsize=8,
                color=MUTED, **FONT)


# --- wordmark -------------------------------------------------------------
# "Scope" is placed from the measured extent of "BBB" so the two colours meet
# exactly, whatever the font metrics turn out to be.
head = ax.text(3.5, 30.0, "BBB", fontsize=25, fontweight="bold", color=INK,
               ha="left", va="center", **FONT)
fig.canvas.draw()
edge_x = ax.transData.inverted().transform(
    (head.get_window_extent().x1, 0))[0]
ax.text(edge_x, 30.0, "Scope", fontsize=25, fontweight="bold", color=GREEN,
        ha="left", va="center", **FONT)
ax.add_line(Line2D([3.5, 9.0], [26.2, 26.2], color=GREEN, linewidth=2.6,
                   solid_capstyle="butt"))
ax.text(10.5, 26.2, "Blood-brain barrier permeability, reported with the "
                    "distance from the data it was trained on",
        fontsize=10.5, color=MUTED, ha="left", va="center", **FONT)

Y, H = 4.5, 16.5
CY = Y + H / 2

# --- the molecule ---------------------------------------------------------
card(3.0, Y, 21, H)
molecule_png = HERE / "_logo_molecule.png"
render_molecule(molecule_png)
image = mpimg.imread(molecule_png)
# The axes are square in display units, so the extent keeps the 4:3 drawing
# canvas undistorted: a stretched structure would read as wrong bond angles.
ax.imshow(image, extent=(6.5, 20.5, CY - 5.25, CY + 5.25), zorder=4,
          aspect="auto", interpolation="antialiased")
ax.text(13.5, Y + H - 2.0, NAME, ha="center", va="center", fontsize=10,
        fontweight="bold", color=INK, **FONT)
ax.text(13.5, Y + 1.9, "SMILES in", ha="center", va="center", fontsize=8.5,
        color=MUTED, **FONT)

arrow(25.0, 28.5, CY)

# --- the fingerprint, drawn bit by bit ------------------------------------
card(29.5, Y, 20, H)
ax.text(39.5, Y + H - 2.0, "MACCS keys", ha="center", va="center", fontsize=10,
        fontweight="bold", color=INK, **FONT)
COLS = 24
CELL, GAP = 0.62, 0.17
grid_w = COLS * (CELL + GAP) - GAP
grid_x = 39.5 - grid_w / 2
grid_y = CY + 3.0
for index, bit in enumerate(maccs):
    row, col = divmod(index, COLS)
    ax.add_patch(Rectangle(
        (grid_x + col * (CELL + GAP), grid_y - row * (CELL + GAP) - CELL),
        CELL, CELL, facecolor=BLUE if bit else FAINT, edgecolor="none",
        zorder=4))
ax.text(39.5, Y + 1.9, f"{int(maccs.sum())} of 167 keys set",
        ha="center", va="center", fontsize=8.5, color=MUTED, **FONT)

arrow(50.5, 54.0, CY)

# --- the model ------------------------------------------------------------
card(55.0, Y, 12, H)
ax.text(61.0, CY + 1.4, "Random", ha="center", va="center", fontsize=10,
        fontweight="bold", color=INK, **FONT)
ax.text(61.0, CY - 0.6, "forest", ha="center", va="center", fontsize=10,
        fontweight="bold", color=INK, **FONT)
ax.text(61.0, Y + 1.9, "scaffold-split", ha="center", va="center",
        fontsize=8.5, color=MUTED, **FONT)

arrow(68.0, 71.5, CY)

# --- the readout ----------------------------------------------------------
card(72.5, Y, 44.5, H, edge=GREEN, lw=1.8)
ax.add_line(Line2D([85.0, 85.0], [Y + 2.2, Y + H - 2.2], color=RULE,
                   linewidth=1.4))

ax.text(78.75, Y + H - 2.0, "prediction", ha="center", va="center",
        fontsize=8.5, color=MUTED, **FONT)
ax.text(78.75, CY + 0.9, "BBB+", ha="center", va="center", fontsize=21,
        fontweight="bold", color=GREEN, **FONT)
ax.text(78.75, Y + 2.6, f"p = {probability:.2f}", ha="center", va="center",
        fontsize=11, fontweight="bold", color=INK, **FONT)

ax.text(101.25, Y + H - 2.0, "nearest training molecule", ha="center",
        va="center", fontsize=8.5, color=MUTED, **FONT)
BAR_X0, BAR_X1, BAR_H = 87.5, 115.0, 2.4
BAR_Y = CY - 2.6
span = BAR_X1 - BAR_X0
for low, high, colour, label, accuracy in BANDS:
    x0 = BAR_X0 + low * span
    ax.add_patch(FancyBboxPatch(
        (x0 + 0.12, BAR_Y), (high - low) * span - 0.24, BAR_H,
        boxstyle="round,pad=0,rounding_size=0.5", facecolor=colour,
        edgecolor="none", alpha=0.92, zorder=4))
    centre = x0 + (high - low) * span / 2
    ax.text(centre, BAR_Y - 0.8, label, ha="center", va="top", fontsize=7.5,
            color=MUTED, **FONT)
    ax.text(centre, BAR_Y - 2.5, accuracy, ha="center", va="top", fontsize=8.5,
            fontweight="bold", color=colour, **FONT)

marker = BAR_X0 + similarity * span
ax.add_patch(Polygon([[marker, BAR_Y + BAR_H + 0.2],
                      [marker - 0.8, BAR_Y + BAR_H + 1.6],
                      [marker + 0.8, BAR_Y + BAR_H + 1.6]],
                     closed=True, facecolor=INK, edgecolor="none", zorder=5))
ax.text(marker, BAR_Y + BAR_H + 2.0, f"Tanimoto {similarity:.2f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold", color=INK,
        **FONT)

fig.savefig(HERE.parent / "logo.png", dpi=150, facecolor="white",
            bbox_inches="tight", pad_inches=0.14)
molecule_png.unlink(missing_ok=True)
print(f"wrote logo.png  ({NAME}: p={probability:.2f}, "
      f"Tanimoto={similarity:.3f}, {int(maccs.sum())} keys set)")
