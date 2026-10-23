"""BBBScope: blood-brain barrier permeability with an applicability domain.

Models are trained on a scaffold-partitioned split of B3DB, with resampling
confined to the training partition. Every prediction is returned with its
distance from the training domain - the scope - because accuracy depends
strongly on that distance and a prediction without it is not interpretable.
"""
from __future__ import annotations

import gzip
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

FINGERPRINTS = {
    "MACCS keys": "maccs",
    "Morgan (ECFP4)": "morgan",
    "Avalon": "avalon",
    "Topological torsion": "topological_torsion",
}

# Accuracy measured on held-out compounds, by distance from the training set.
# These bands are what the name refers to.
SCOPE_BANDS = [
    (0.50, "In scope", "about 94 % accurate", "#1baf7a"),
    (0.30, "Edge of scope", "about 88 % accurate", "#eb6834"),
    (0.00, "Out of scope", "about 76 % accurate", "#c2352f"),
]

# Chosen to span the scope bands rather than to flatter the model. The last
# entry is the instructive one: a confident-looking 0.86 on a structure the
# model has never seen anything like.
EXAMPLE = "\n".join([
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                           # caffeine, in scope
    "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",                # sucrose, in scope
    "O=C1NC(Cc2ccccc2)C(=O)NC(CC(C)C)C(=O)NC1CO",             # cyclic peptide, edge
    "OCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",   # fluorotelomer, out
])

st.set_page_config(page_title="BBBScope", page_icon="\U0001f9e0", layout="wide")


@st.cache_resource
def load_manifest() -> dict:
    return json.loads((MODELS / "manifest.json").read_text())


@st.cache_resource
def load_model(kind: str):
    with gzip.open(MODELS / f"rf_{kind}.pkl.gz", "rb") as fh:
        return pickle.load(fh)


@st.cache_resource
def load_columns(kind: str) -> np.ndarray:
    return np.load(MODELS / f"columns_{kind}.npy")


@st.cache_resource
def load_reference() -> list:
    """Training fingerprints, unpacked once, for the scope calculation."""
    bits = np.unpackbits(np.load(MODELS / "ad_train_morgan.npy"), axis=1)[:, :2048]
    reference = []
    for row in bits:
        vector = DataStructs.ExplicitBitVect(2048)
        vector.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
        reference.append(vector)
    return reference


def fingerprint(mol, kind: str) -> np.ndarray:
    if kind == "morgan":
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if kind == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if kind == "avalon":
        return np.array(GetAvalonFP(mol, nBits=1024))
    return np.array(GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048))


def parse(lines: list[str]) -> tuple[list, list[str], list[str]]:
    """Molecules and their SMILES, kept aligned, plus the ones that failed.

    Alignment is the point: an earlier version dropped unparseable entries from
    the feature matrix but not from the label column, so predictions were
    silently attached to the wrong molecules.
    """
    mols, kept, failed = [], [], []
    for line in lines:
        smiles = line.strip()
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(smiles)
        else:
            mols.append(mol)
            kept.append(smiles)
    return mols, kept, failed


def band(similarity: float) -> tuple[str, str, str]:
    for threshold, label, accuracy, colour in SCOPE_BANDS:
        if similarity >= threshold:
            return label, accuracy, colour
    return SCOPE_BANDS[-1][1:]


logo = HERE / "logo.png"
if logo.exists():
    st.image(Image.open(logo), use_container_width=True)

st.markdown("""
Predicts blood-brain barrier permeability from structure, and reports how far
each compound sits from the data the model was trained on.

**That distance is the point.** On held-out compounds, accuracy runs from about
**76 %** for structures unlike anything in training to about **95 %** for close
analogues. A prediction without that context is not interpretable, so BBBScope
returns it with every result.
""")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("CSV with a `Smiles` column", type=["csv"])
    if "typed" not in st.session_state:
        st.session_state.typed = ""
    if st.button("Load example", help="Four compounds spanning the scope bands"):
        st.session_state.typed = EXAMPLE
    typed = st.text_area("or paste SMILES, one per line", height=170, key="typed")

    st.header("Settings")
    fingerprint_name = st.selectbox("Fingerprint", list(FINGERPRINTS))
    show_properties = st.checkbox("Include physicochemical properties", value=False)
    st.caption(
        "Random forest, trained on a scaffold-partitioned split. Consensus "
        "modelling gave no reliable advantage over it and is not served; see the "
        "README."
    )
    run = st.button("Predict", type="primary")

if not run:
    st.info("Add input in the sidebar, then press Predict.")
    st.stop()

if uploaded is not None:
    frame = pd.read_csv(uploaded)
    if "Smiles" not in frame.columns:
        st.error("The CSV needs a column named `Smiles`.")
        st.stop()
    raw = frame["Smiles"].astype(str).tolist()
elif typed.strip():
    raw = typed.splitlines()
else:
    st.warning("No input given.")
    st.stop()

mols, kept, failed = parse(raw)
if not mols:
    st.error("None of the input could be parsed as SMILES.")
    st.stop()
if failed:
    st.warning(f"{len(failed)} entries could not be parsed and were skipped: "
               + ", ".join(failed[:5]) + ("…" if len(failed) > 5 else ""))

kind = FINGERPRINTS[fingerprint_name]

with st.spinner("Predicting…"):
    columns = load_columns(kind)
    features = np.vstack([fingerprint(m, kind) for m in mols])[:, columns]
    probability = load_model(kind).predict_proba(features.astype(float))[:, 1]
    reference = load_reference()
    similarity = np.array([
        max(DataStructs.BulkTanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), reference))
        for m in mols])

bands = [band(s) for s in similarity]
results = pd.DataFrame({
    "Smiles": kept,
    "Prediction": np.where(probability >= 0.5, "BBB+", "BBB−"),
    "P(BBB+)": probability.round(3),
    "Scope": [b[0] for b in bands],
    "Similarity": similarity.round(3),
    "Expected accuracy": [b[1] for b in bands],
})

if show_properties:
    results["MW"] = [round(Descriptors.MolWt(m), 1) for m in mols]
    results["cLogP"] = [round(Descriptors.MolLogP(m), 2) for m in mols]
    results["TPSA"] = [round(Descriptors.TPSA(m), 1) for m in mols]
    results["HBD"] = [Descriptors.NumHDonors(m) for m in mols]

counts = {label: sum(b[0] == label for b in bands)
          for _, label, _, _ in SCOPE_BANDS}
left, middle, right = st.columns(3)
for column, (_, label, _, colour) in zip((left, middle, right), SCOPE_BANDS):
    column.markdown(
        f"<div style='padding:0.6rem 0.9rem;border-left:5px solid {colour};"
        f"background:rgba(128,128,128,0.08);border-radius:4px'>"
        f"<div style='font-size:1.6rem;font-weight:700'>{counts[label]}</div>"
        f"<div style='opacity:0.8'>{label}</div></div>",
        unsafe_allow_html=True)

st.subheader("Predictions")
st.dataframe(
    results.style.apply(
        lambda row: [f"color: {band(row['Similarity'])[2]}"
                     if col in ("Scope", "Similarity") else "" for col in results.columns],
        axis=1),
    use_container_width=True)

outside = counts["Out of scope"] + counts["Edge of scope"]
if outside:
    st.warning(
        f"{outside} of {len(results)} compounds fall below Tanimoto 0.5 to the "
        "training set. Their predictions are less reliable, and should be read as "
        "indicative rather than as classifications.")

buffer = io.StringIO()
results.to_csv(buffer, index=False)
st.download_button("Download predictions (CSV)", data=buffer.getvalue(),
                   file_name="bbbscope_predictions.csv", mime="text/csv")

with st.expander("How the scope is defined"):
    st.markdown("""
Scope is the Tanimoto similarity between a query and its nearest neighbour in
the training partition, on 2048-bit Morgan fingerprints. Measured on held-out
compounds:

| Similarity | Scope | Accuracy |
|---|---|---|
| ≥ 0.5 | In scope | ~94 % |
| 0.3 – 0.5 | Edge of scope | ~88 % |
| < 0.3 | Out of scope | ~76 % |

A leverage-based domain was also evaluated. It behaves for MACCS keys but
degenerates for wider fingerprints, flagging 63 % of a test set at 1 951
descriptors, because the usual warning leverage assumes far fewer descriptors
than compounds. The similarity domain does not have that failure mode.
""")

with st.expander("Model details and performance"):
    st.json(load_manifest())
    st.markdown("""
Across ten repeated scaffold partitions, random forest on MACCS keys:
**AUC 0.934 (0.923–0.948), MCC 0.693 (0.666–0.734)**. On compounds genuinely
external to the training data, **AUC 0.954**.

Earlier releases of this app, under the name EnsembleBBB, reported substantially
higher figures. Those were inflated by two faults: the minority class was
oversampled before the data were partitioned, and the partition was random over
a dataset in which roughly half the molecules share a descriptor vector with
another molecule, because the fingerprints ignore stereochemistry. Both are
corrected here.
""")
