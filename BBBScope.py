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

# Structure depiction needs RDKit's Cairo backend, which in turn needs system
# libraries (libXrender, libcairo) that slim container images often omit.
# packages.txt installs them, but a prediction tool should not fail to start
# over a drawing library, so the import is optional and the app degrades to a
# table if it is unavailable.
try:
    from rdkit.Chem.Draw import rdMolDraw2D
    DEPICTION = True
    DEPICTION_ERROR = ""
except ImportError as exc:                       # pragma: no cover
    rdMolDraw2D = None
    DEPICTION = False
    DEPICTION_ERROR = str(exc)

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

# Accuracy by descriptor and distance from training, measured on the held-out
# scaffold partition. Morgan is the worst choice where a user most needs help,
# so the app says so rather than presenting the four as interchangeable.
FINGERPRINTS = {
    "MACCS keys": ("maccs", "Recommended. Best overall and at the edge of scope."),
    "Avalon": ("avalon", "Best for compounds far from the training set."),
    "Topological torsion": ("topological_torsion", "Comparable to MACCS in scope."),
    "Morgan (ECFP4)": ("morgan", "Matches MACCS in scope, but degrades sharply outside it."),
}

SCOPE_BANDS = [
    (0.50, "In scope", "~94 % accurate", "#1baf7a"),
    (0.30, "Edge of scope", "~88 % accurate", "#eb6834"),
    (0.00, "Out of scope", "~76 % accurate", "#c2352f"),
]

EXAMPLES = {
    "Spanning the scope": [
        ("caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"),
        ("sucrose", "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O"),
        ("cyclic tripeptide", "O=C1NC(Cc2ccccc2)C(=O)NC(CC(C)C)C(=O)NC1CO"),
        ("fluorotelomer alcohol", "OCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
    ],
    "CNS drugs": [
        ("diazepam", "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O"),
        ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
        ("haloperidol", "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"),
        ("levodopa", "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O"),
        ("donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"),
    ],
    "Peripherally restricted": [
        ("atenolol", "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
        ("loratadine", "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1"),
        ("domperidone", "O=C1Nc2ccccc2N1C1CCN(CCCN2C(=O)Nc3ccc(Cl)cc32)CC1"),
        ("cetirizine", "OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1"),
    ],
    "Novel chemotypes": [
        ("crown ether", "C1COCCOCCOCCOCCOCCO1"),
        ("porphyrin core", "c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3"),
        ("adamantanamine", "NC12CC3CC(CC(C3)C1)C2"),
        ("alkyl sulfonate", "CCCCCCCCCCCCCCCCCCS(=O)(=O)O"),
    ],
}

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
def load_reference() -> tuple[list, pd.DataFrame]:
    """Training fingerprints and the structures they belong to.

    Both are written by the export scripts in the same order; the app relies
    on that alignment to name the nearest neighbour.
    """
    bits = np.unpackbits(np.load(MODELS / "ad_train_morgan.npy"), axis=1)[:, :2048]
    vectors = []
    for row in bits:
        vector = DataStructs.ExplicitBitVect(2048)
        vector.SetBitsFromList([int(i) for i in np.flatnonzero(row)])
        vectors.append(vector)
    molecules = pd.read_csv(MODELS / "ad_train_molecules.csv")
    if len(molecules) != len(vectors):
        raise ValueError("training fingerprints and structures are out of step")
    return vectors, molecules


def fingerprint(mol, kind: str) -> np.ndarray:
    if kind == "morgan":
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if kind == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if kind == "avalon":
        return np.array(GetAvalonFP(mol, nBits=1024))
    return np.array(GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048))


def parse(lines: list[str]) -> tuple[list, list[str], list[str]]:
    """Molecules and their SMILES, kept aligned, plus the ones that failed."""
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


def depict(mol, size=(320, 240)) -> bytes | None:
    if not DEPICTION:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().bondLineWidth = 2
        drawer.drawOptions().clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:                            # pragma: no cover
        return None


logo = HERE / "logo.png"
if logo.exists():
    st.image(Image.open(logo), width="stretch")

st.markdown("""
Predicts blood-brain barrier permeability from structure, and reports how far
each compound sits from the data the model was trained on.

**That distance is the point.** On held-out compounds, accuracy runs from about
**76 %** for structures unlike anything in training to about **94 %** for close
analogues. A prediction without that context is not interpretable, so BBBScope
returns it with every result.
""")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("CSV with a `Smiles` column", type=["csv"])
    if "typed" not in st.session_state:
        st.session_state.typed = ""
    example_set = st.selectbox("Example set", list(EXAMPLES))
    if st.button("Load example"):
        st.session_state.typed = "\n".join(s for _, s in EXAMPLES[example_set])
    typed = st.text_area("or paste SMILES, one per line", height=150, key="typed")

    st.header("Descriptor")
    fingerprint_name = st.selectbox("Fingerprint", list(FINGERPRINTS))
    st.caption(FINGERPRINTS[fingerprint_name][1])
    show_detail = st.checkbox("Show per-compound detail", value=True)
    run = st.button("Predict", type="primary")

if not run:
    st.info("Pick an example or paste SMILES in the sidebar, then press Predict.")
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

kind = FINGERPRINTS[fingerprint_name][0]

with st.spinner("Predicting…"):
    columns = load_columns(kind)
    features = np.vstack([fingerprint(m, kind) for m in mols])[:, columns]
    probability = load_model(kind).predict_proba(features.astype(float))[:, 1]

    vectors, training = load_reference()
    similarity, neighbour_index = [], []
    for mol in mols:
        scores = DataStructs.BulkTanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), vectors)
        best = int(np.argmax(scores))
        similarity.append(scores[best])
        neighbour_index.append(best)
    similarity = np.asarray(similarity)

bands = [band(s) for s in similarity]
results = pd.DataFrame({
    "Smiles": kept,
    "Prediction": np.where(probability >= 0.5, "BBB+", "BBB−"),
    "P(BBB+)": probability.round(3),
    "Scope": [b[0] for b in bands],
    "Similarity": similarity.round(3),
    "Nearest training molecule": training.loc[neighbour_index, "smiles"].to_numpy(),
    "Its label": ["BBB+" if v else "BBB−"
                  for v in training.loc[neighbour_index, "label"]],
    "MW": [round(Descriptors.MolWt(m), 1) for m in mols],
    "cLogP": [round(Descriptors.MolLogP(m), 2) for m in mols],
    "TPSA": [round(Descriptors.TPSA(m), 1) for m in mols],
    "HBD": [Descriptors.NumHDonors(m) for m in mols],
})

counts = {label: sum(b[0] == label for b in bands) for _, label, _, _ in SCOPE_BANDS}
for column, (_, label, accuracy, colour) in zip(st.columns(3), SCOPE_BANDS):
    column.markdown(
        f"<div style='padding:0.6rem 0.9rem;border-left:5px solid {colour};"
        f"background:rgba(128,128,128,0.08);border-radius:4px'>"
        f"<div style='font-size:1.6rem;font-weight:700'>{counts[label]}</div>"
        f"<div style='opacity:0.85'>{label}</div>"
        f"<div style='opacity:0.6;font-size:0.8rem'>{accuracy}</div></div>",
        unsafe_allow_html=True)

st.subheader("Predictions")
st.dataframe(results, width="stretch", hide_index=True)

outside = counts["Out of scope"] + counts["Edge of scope"]
if outside:
    st.warning(
        f"{outside} of {len(results)} compounds fall below Tanimoto 0.5 to the "
        "training set. Their predictions are less reliable and should be read as "
        "indicative rather than as classifications.")

buffer = io.StringIO()
results.to_csv(buffer, index=False)
st.download_button("Download predictions (CSV)", data=buffer.getvalue(),
                   file_name="bbbscope_predictions.csv", mime="text/csv")

if show_detail:
    st.subheader("Per-compound detail")
    if not DEPICTION:
        st.caption(
            "Structure depiction is unavailable in this deployment, so the "
            "comparison below is shown as SMILES. Everything else is unaffected.")
    for i, mol in enumerate(mols):
        label, accuracy, colour = bands[i]
        neighbour = training.iloc[neighbour_index[i]]
        neighbour_label = "BBB+" if neighbour["label"] else "BBB−"
        with st.container(border=True):
            query_col, arrow_col, neighbour_col, text_col = st.columns([3, 1, 3, 5])
            query_image = depict(mol)
            if query_image:
                query_col.image(query_image, caption=f"Query {i + 1}")
            else:
                query_col.caption(f"Query {i + 1}")
                query_col.code(kept[i], language=None)
            arrow_col.markdown(
                "<div style='text-align:center;padding-top:3.2rem;font-size:1.4rem;"
                "opacity:0.45'>↔</div>", unsafe_allow_html=True)
            neighbour_mol = Chem.MolFromSmiles(neighbour["smiles"])
            neighbour_image = depict(neighbour_mol) if neighbour_mol else None
            if neighbour_image:
                neighbour_col.image(
                    neighbour_image,
                    caption=f"Nearest in training · {neighbour_label}")
            else:
                neighbour_col.caption(f"Nearest in training · {neighbour_label}")
                neighbour_col.code(neighbour["smiles"], language=None)
            text_col.markdown(
                f"**{results.loc[i, 'Prediction']}** &nbsp; P(BBB+) = "
                f"{probability[i]:.3f}\n\n"
                f"<span style='color:{colour};font-weight:600'>{label}</span> "
                f"&nbsp;·&nbsp; Tanimoto {similarity[i]:.3f} &nbsp;·&nbsp; {accuracy}\n\n"
                f"MW {results.loc[i, 'MW']} &nbsp;·&nbsp; "
                f"cLogP {results.loc[i, 'cLogP']} &nbsp;·&nbsp; "
                f"TPSA {results.loc[i, 'TPSA']} &nbsp;·&nbsp; "
                f"HBD {results.loc[i, 'HBD']}",
                unsafe_allow_html=True)
            if similarity[i] < 0.5:
                text_col.caption(
                    "The nearest training molecule is not a close analogue, so this "
                    "prediction is an extrapolation.")

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
than compounds. The similarity domain has no such failure mode.
""")

with st.expander("Which descriptor to use"):
    st.markdown("""
The four are not interchangeable, and the differences appear where they matter
most. Matthews correlation coefficient on the held-out partition, by distance
from training:

| Descriptor | In scope | Edge of scope | Out of scope |
|---|---|---|---|
| **MACCS keys** | **0.860** | **0.696** | 0.405 |
| Avalon | 0.811 | 0.614 | **0.458** |
| Topological torsion | 0.852 | 0.547 | 0.383 |
| Morgan (ECFP4) | 0.858 | 0.512 | 0.231 |

Morgan matches MACCS on close analogues and then collapses, which is the
opposite of what a richer descriptor is usually assumed to do. Morgan encodes
specific atom environments, so a novel chemotype sets bits the model has never
seen. MACCS encodes generic features — counts of oxygens, hydrogen-bond donors,
simple connectivity — which transfer across chemotypes.

MACCS is the default. Avalon is worth trying for compounds far outside the
training set.
""")

with st.expander("Model details and performance"):
    st.json(load_manifest())
    st.markdown("""
Across ten repeated scaffold partitions, random forest on MACCS keys:
**AUC 0.934 (0.923–0.948), MCC 0.693 (0.666–0.734)**. On compounds genuinely
external to the training data, **AUC 0.954**.

Earlier releases, under the name EnsembleBBB, reported substantially higher
figures. Those were inflated by two faults: the minority class was oversampled
before the data were partitioned, and the partition was random over a dataset in
which roughly half the molecules share a descriptor vector with another
molecule, because the fingerprints ignore stereochemistry. Both are corrected
here.
""")
