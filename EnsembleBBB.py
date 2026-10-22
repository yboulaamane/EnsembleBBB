"""EnsembleBBB: blood-brain barrier permeability prediction.

Models are trained on a scaffold-split partition of B3DB, with oversampling
applied inside the training partition only. Every prediction is reported with
its probability and with the distance from the training domain, because
accuracy depends strongly on that distance.
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
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprintAsBitVect

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

FINGERPRINTS = {
    "Morgan (ECFP4)": "morgan",
    "MACCS keys": "maccs",
    "Avalon": "avalon",
    "Topological torsion": "topological_torsion",
}
AD_WARN = 0.5

st.set_page_config(page_title="EnsembleBBB", layout="wide")


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
def load_domain_reference() -> list:
    packed = np.load(MODELS / "ad_train_morgan.npy")
    bits = np.unpackbits(packed, axis=1)[:, :2048]
    reference = []
    for row in bits:
        vector = DataStructs.ExplicitBitVect(2048)
        vector.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
        reference.append(vector)
    return reference


def compute_fingerprint(mol, kind: str) -> np.ndarray:
    if kind == "morgan":
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if kind == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if kind == "avalon":
        return np.array(GetAvalonFP(mol, nBits=1024))
    return np.array(GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048))


def parse(smiles_list: list[str]) -> tuple[list, list[str], list[str]]:
    """Return parsed molecules with their SMILES, and the ones that failed.

    Keeping these aligned is the point: an earlier version of this app dropped
    unparseable entries from the feature matrix but not from the label column,
    so predictions were silently attached to the wrong molecules.
    """
    mols, kept, failed = [], [], []
    for smiles in smiles_list:
        smiles = smiles.strip()
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failed.append(smiles)
        else:
            mols.append(mol)
            kept.append(smiles)
    return mols, kept, failed


def predict(mols, kind: str) -> tuple[np.ndarray, np.ndarray]:
    columns = load_columns(kind)
    features = np.vstack([compute_fingerprint(m, kind) for m in mols])[:, columns]
    probability = load_model(kind).predict_proba(features.astype(float))[:, 1]
    return probability, (probability >= 0.5).astype(int)


def domain_distance(mols) -> np.ndarray:
    reference = load_domain_reference()
    out = []
    for mol in mols:
        query = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        out.append(max(DataStructs.BulkTanimotoSimilarity(query, reference)))
    return np.asarray(out)


logo = MODELS.parent / "logo.png"
if logo.exists():
    st.image(Image.open(logo), use_container_width=True)

st.markdown("""
# EnsembleBBB

Predicts blood-brain barrier permeability from structure, using models trained
on a **scaffold-split** partition of the [B3DB dataset](https://github.com/theochem/B3DB).

Each prediction comes with a probability and with its Tanimoto similarity to the
nearest training molecule. That similarity matters: on held-out compounds,
accuracy is about **94 % above 0.5** similarity and about **76 % below 0.3**.
""")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("CSV with a `Smiles` column", type=["csv"])
    typed = st.text_area("or paste SMILES, one per line", height=180)

    st.header("Settings")
    fingerprint_name = st.selectbox("Fingerprint", list(FINGERPRINTS), index=1)
    st.caption(
        "Random forest. Consensus modelling gave no reliable advantage over it, "
        "so it is no longer served; see the README."
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
               + ", ".join(failed[:5]) + ("..." if len(failed) > 5 else ""))

kind = FINGERPRINTS[fingerprint_name]

with st.spinner("Predicting..."):
    probability, label = predict(mols, kind)
    similarity = domain_distance(mols)

results = pd.DataFrame({
    "Smiles": kept,
    "Prediction": np.where(label == 1, "BBB+", "BBB-"),
    "Probability_BBB+": probability.round(3),
    "Nearest_training_Tanimoto": similarity.round(3),
    "In_applicability_domain": np.where(similarity >= AD_WARN, "yes", "no"),
})

st.subheader("Predictions")
st.dataframe(results, use_container_width=True)

outside = int((similarity < AD_WARN).sum())
if outside:
    st.warning(
        f"{outside} of {len(results)} compounds fall below Tanimoto {AD_WARN} to the "
        "training set. Their predictions are less reliable; treat them as indicative."
    )

buffer = io.StringIO()
results.to_csv(buffer, index=False)
st.download_button(
    "Download predictions (CSV)",
    data=buffer.getvalue(),
    file_name="ensemblebbb_predictions.csv",
    mime="text/csv",
)

with st.expander("Model details"):
    manifest = load_manifest()
    st.json(manifest)
    st.markdown("""
Performance under scaffold-split validation, random forest on MACCS keys:
**AUC 0.943, MCC 0.742**. On compounds genuinely external to the training data,
**AUC 0.954**.

Earlier releases of this app reported substantially higher figures. Those were
inflated by two faults in the original pipeline: the minority class was
oversampled before the data were partitioned, and the partition was random over
a dataset in which roughly half the molecules share a descriptor vector with
another molecule, because the fingerprints ignore stereochemistry. Both are
corrected here.
""")
