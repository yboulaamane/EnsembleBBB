"""Matched molecular pair analysis: which single changes gain permeability?

Attribution says which substructures associate with permeability. It does not
tell a chemist what to do. A matched pair differs by one substituent, so a pair
whose members differ in label isolates that change with the rest of the molecule
held constant.

Two cuts are used, which gives a constant core and a variable substituent in the
Hussain-Rea sense. An earlier version used a single cut with a size filter that
required the variable part to be under 40 % of the molecule; a single cut splits
a molecule roughly in half, so that filter discarded almost every pair and the
analysis returned nothing. The dataset supports this comfortably: 2.45 million
matched pairs, 36 % of them label-discordant.

Because many transformations are tested, p-values from a two-sided binomial test
against an even split are corrected by Benjamini-Hochberg, and only
transformations surviving at 5 % false discovery are reported as directional.
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA
from scipy.stats import binomtest

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# A single cut gives one core and one substituent, which is what makes a
# transformation recur across different molecules. Two cuts produce a central
# core with two attachment points, so the "variable part" is a pair of
# substituents taken together - far too specific to repeat, which is why an
# earlier run found only seven transformations across 2.45 million pairs.
MAX_CUTS = 1
MAX_VARIABLE_HEAVY = 13   # the changing group must be a substituent
MIN_CORE_RATIO = 2.0      # and the core must be substantially larger than it
MIN_PAIRS = 10            # discordant pairs needed before a rate is reported
MAX_CORE_MEMBERS = 300    # vast cores are generic and uninformative
FDR = 0.05


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    n = len(pvalues)
    thresholds = alpha * np.arange(1, n + 1) / n
    passing = ranked <= thresholds
    cutoff = np.max(np.flatnonzero(passing)) if passing.any() else -1
    keep = np.zeros(n, dtype=bool)
    if cutoff >= 0:
        keep[order[: cutoff + 1]] = True
    return keep


def heavy_atoms(fragment: str) -> int:
    mol = Chem.MolFromSmiles(fragment.replace("[*:1]", "[H]").replace("[*:2]", "[H]"))
    return mol.GetNumHeavyAtoms() if mol else 0


def index_by_core(records):
    """Map each constant core to the molecules sharing it and their substituent."""
    by_core: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for canonical, label, mol in records:
        try:
            cuts = rdMMPA.FragmentMol(mol, maxCuts=MAX_CUTS, resultsAsMols=False)
        except Exception:
            continue
        seen = set()
        for core, chains in cuts:
            pieces = (chains.split(".") if not core else [core, chains])
            if len(pieces) != 2:
                continue
            sizes = [heavy_atoms(p) for p in pieces]
            large, small = (0, 1) if sizes[0] >= sizes[1] else (1, 0)
            if sizes[small] == 0 or sizes[small] > MAX_VARIABLE_HEAVY:
                continue
            if sizes[large] < MIN_CORE_RATIO * sizes[small]:
                continue
            key, variable = pieces[large], pieces[small]
            if (key, variable) in seen:
                continue
            seen.add((key, variable))
            by_core[key].append((canonical, label, variable))
    return by_core


def main() -> None:
    raw = pd.read_csv(PROJECT / "rdkit-bbb" / "b3db.csv")
    records = []
    for smiles, label in zip(raw["SMILES"], raw["Class"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            records.append((Chem.MolToSmiles(mol), int(label), mol))
    print(f"molecules: {len(records)}", flush=True)

    by_core = index_by_core(records)
    print(f"distinct cores: {len(by_core)}", flush=True)

    counts: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"gain": 0, "loss": 0, "examples": []})

    for members in by_core.values():
        if not 1 < len(members) <= MAX_CORE_MEMBERS:
            continue
        for (smi_a, lab_a, var_a), (smi_b, lab_b, var_b) in combinations(members, 2):
            if var_a == var_b or smi_a == smi_b or lab_a == lab_b:
                continue
            # Orient so the transformation always reads non-permeable to permeable.
            if lab_a == 0:
                src, dst, example = var_a, var_b, (smi_a, smi_b)
            else:
                src, dst, example = var_b, var_a, (smi_b, smi_a)
            key = (src, dst) if src < dst else (dst, src)
            entry = counts[key]
            if (src, dst) == key:
                entry["gain"] += 1
                if len(entry["examples"]) < 2:
                    entry["examples"].append(example)
            else:
                entry["loss"] += 1

    rows = []
    for (src, dst), entry in counts.items():
        gain, loss = entry["gain"], entry["loss"]
        total = gain + loss
        if total < MIN_PAIRS:
            continue
        lo, hi = wilson(gain, total)
        rows.append({
            "from": src, "to": dst,
            "pairs": total, "gained": gain, "lost": loss,
            "gain_rate": gain / total,
            "ci_low": lo, "ci_high": hi,
            "p_value": binomtest(gain, total, 0.5, alternative="two-sided").pvalue,
            "example_from": entry["examples"][0][0] if entry["examples"] else "",
            "example_to": entry["examples"][0][1] if entry["examples"] else "",
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        print(f"no transformation reached {MIN_PAIRS} discordant pairs")
        return

    frame["significant"] = benjamini_hochberg(frame["p_value"].to_numpy(), FDR)
    frame = frame.sort_values(["significant", "gain_rate", "pairs"],
                              ascending=[False, False, False])
    for column in ("gain_rate", "ci_low", "ci_high"):
        frame[column] = frame[column].round(3)
    frame["p_value"] = frame["p_value"].map(lambda v: f"{v:.2e}")
    frame.to_csv(RESULTS / "table_mmp.csv", index=False)

    n_sig = int(frame["significant"].sum())
    print(f"\ntransformations with >= {MIN_PAIRS} discordant pairs: {len(frame)}")
    print(f"significant at {FDR:.0%} false discovery: {n_sig}", flush=True)

    gains = frame[frame["significant"] & (frame["gain_rate"] > 0.5)].head(15)
    losses = frame[frame["significant"] & (frame["gain_rate"] < 0.5)].head(10)

    def show(title, block, column):
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        print(f"{'from':<30}{'to':<30}{'pairs':>7}{'rate':>8}   95 % CI")
        for _, r in block.iterrows():
            rate = r["gain_rate"] if column == "gain" else 1 - r["gain_rate"]
            lo, hi = ((r["ci_low"], r["ci_high"]) if column == "gain"
                      else (1 - r["ci_high"], 1 - r["ci_low"]))
            src, dst = ((r["from"], r["to"]) if column == "gain"
                        else (r["to"], r["from"]))
            print(f"{src[:29]:<30}{dst[:29]:<30}{r['pairs']:>7}{rate:>8.2f}"
                  f"   [{lo:.2f}, {hi:.2f}]")

    show("TRANSFORMATIONS THAT GAIN PERMEABILITY", gains, "gain")
    show("TRANSFORMATIONS THAT LOSE PERMEABILITY", losses, "loss")

    summary = {
        "molecules": len(records),
        "max_cuts": MAX_CUTS,
        "distinct_cores": len(by_core),
        "transformations_tested": int(len(frame)),
        "significant_at_fdr": n_sig,
        "min_pairs": MIN_PAIRS,
        "fdr": FDR,
    }
    (RESULTS / "mmp_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {RESULTS / 'table_mmp.csv'}")


if __name__ == "__main__":
    main()
