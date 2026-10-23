"""How often does any single transformation actually recur in B3DB?

The matched-pair analysis found nothing at a threshold of ten discordant pairs.
That is either because the threshold is slightly too strict or because no
transformation recurs at all. Those call for different conclusions, so report
the distribution rather than lowering the threshold until something appears.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = HERE / "results"

MAX_VARIABLE_HEAVY = 13
MIN_CORE_RATIO = 2.0
MAX_CORE_MEMBERS = 300


def heavy_atoms(fragment: str) -> int:
    mol = Chem.MolFromSmiles(fragment.replace("[*:1]", "[H]").replace("[*:2]", "[H]"))
    return mol.GetNumHeavyAtoms() if mol else 0


raw = pd.read_csv(PROJECT / "rdkit-bbb" / "b3db.csv")
records = []
for smiles, label in zip(raw["SMILES"], raw["Class"]):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        records.append((Chem.MolToSmiles(mol), int(label), mol))

by_core: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
for canonical, label, mol in records:
    try:
        cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
    except Exception:
        continue
    seen = set()
    for core, chains in cuts:
        pieces = chains.split(".") if not core else [core, chains]
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

transform_pairs: Counter = Counter()
transform_discordant: Counter = Counter()

for members in by_core.values():
    if not 1 < len(members) <= MAX_CORE_MEMBERS:
        continue
    for (smi_a, lab_a, var_a), (smi_b, lab_b, var_b) in combinations(members, 2):
        if var_a == var_b or smi_a == smi_b:
            continue
        key = (var_a, var_b) if var_a < var_b else (var_b, var_a)
        transform_pairs[key] += 1
        if lab_a != lab_b:
            transform_discordant[key] += 1

all_counts = Counter(transform_pairs.values())
disc_counts = Counter(transform_discordant.values())

report = {
    "cores": len(by_core),
    "cores_with_multiple_members": sum(1 for v in by_core.values() if len(v) > 1),
    "distinct_transformations": len(transform_pairs),
    "transformations_with_any_discordant_pair": len(transform_discordant),
    "max_pairs_for_one_transformation": max(transform_pairs.values()) if transform_pairs else 0,
    "max_discordant_for_one_transformation": (
        max(transform_discordant.values()) if transform_discordant else 0),
    "discordant_count_distribution": {
        str(k): disc_counts.get(k, 0) for k in sorted(disc_counts)[:15]},
    "transformations_with_at_least": {
        str(n): sum(1 for v in transform_discordant.values() if v >= n)
        for n in (2, 3, 5, 8, 10, 15, 20)},
}

print(json.dumps(report, indent=2))
(RESULTS / "mmp_distribution.json").write_text(json.dumps(report, indent=2))

print("\nmost frequently recurring transformations (by discordant pairs):")
for (src, dst), n in transform_discordant.most_common(12):
    total = transform_pairs[(src, dst)]
    print(f"  {src[:34]:<36} -> {dst[:34]:<36} {n:>4} discordant of {total}")
