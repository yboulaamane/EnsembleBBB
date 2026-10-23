# BBBScope

![BBBScope: olanzapine in, MACCS keys, random forest, BBB+ with its distance
from the training data](logo.png)

Blood-brain barrier permeability prediction that reports **how far each compound
sits from the data the model was trained on** — and what that costs in accuracy.

**Live app:** https://bbbscope.streamlit.app/

Every number in the banner above is computed from the shipped model when the
image is built, so it cannot drift away from what the app returns.

---

## Why the scope matters

Most blood-brain barrier predictors return a class, or a class and a
probability. Neither tells you whether the model has ever seen anything like
your compound. Measured on held-out compounds:

| Tanimoto to nearest training molecule | Scope | Accuracy |
|---|---|---|
| ≥ 0.5 | In scope | ~94 % |
| 0.3 – 0.5 | Edge of scope | ~88 % |
| < 0.3 | Out of scope | ~76 % |

An eighteen-point accuracy range is not a footnote. BBBScope returns that
distance with every prediction, so a result on a novel chemotype can be read as
the extrapolation it is.

A leverage-based applicability domain was evaluated too. It behaves for MACCS
keys but degenerates for wider fingerprints, flagging 63 % of a test set at
1 951 descriptors, because the usual warning leverage *h\** = 3(p+1)/n assumes
far fewer descriptors than compounds. The similarity domain has no such failure
mode.

## Which descriptor to use

The four are not interchangeable, and the differences appear where they matter
most. Matthews correlation coefficient on the held-out partition, split by
distance from training:

| Descriptor | In scope | Edge of scope | Out of scope |
|---|---|---|---|
| **MACCS keys** | **0.860** | **0.696** | 0.405 |
| Avalon | 0.811 | 0.614 | **0.458** |
| Topological torsion | 0.852 | 0.547 | 0.383 |
| Morgan (ECFP4) | 0.858 | 0.512 | 0.231 |

**Morgan matches MACCS on close analogues and then collapses** — the opposite of
what a richer descriptor is usually assumed to do. Morgan encodes specific atom
environments, so a novel chemotype sets bits the model has never seen. MACCS
encodes generic features — counts of oxygens, hydrogen-bond donors, simple
connectivity — which transfer across chemotypes. That matches the attribution
analysis, where the strongest signals are cumulative polarity counts rather than
particular functional groups.

MACCS is the default. Avalon is worth trying for compounds far outside the
training set.

## Performance

Random forest on MACCS keys, across ten repeated scaffold partitions:

| | AUC | MCC | Accuracy |
|---|---|---|---|
| Held out, 10 partitions | **0.934** (0.923–0.948) | **0.693** (0.666–0.734) | 0.866 |
| Genuinely external compounds | 0.954 | 0.775 | 0.909 |

Those two agreeing is the useful part: scaffold-partitioned internal validation
predicts genuine external performance, so you can trust the estimate without an
external set — provided it is taken over repeated partitions rather than one.

## History: this project was wrong, and here is how

This repository was previously **EnsembleBBB**, and reported AUC 0.95–0.96 with
95 % accuracy on an "external" set. Those figures were inflated. Three faults:

**1. Oversampling before partitioning.** The minority class was oversampled
across the whole dataset and only then split, so copies of the same molecule sat
on both sides of every split and every cross-validation fold.

**2. Random partitioning over a stereoisomer-rich dataset.** B3DB contains large
families of stereoisomers, and Morgan, MACCS and Avalon fingerprints ignore
stereochemistry by default. 48–51 % of molecules therefore share a descriptor
vector with another molecule. Under a random split, **63 % of test compounds had
a bit-for-bit identical twin in training**, and the median test compound had
Tanimoto similarity **1.00** to its nearest training neighbour.

**3. The external set was not external.** **67.6 %** of the BBBP evaluation set
is present in B3DB as identical structures. That benchmark was largely measuring
memorisation.

Correcting the protocol costs **0.057 AUC and 0.174 MCC**. Two one-line remedies
remove the leakage: partition by Bemis–Murcko scaffold, or enable stereochemistry
in the fingerprint — the latter costing 0.006 MCC, within fold-to-fold noise.

**Consensus modelling does not help here.** Across ten partitions, soft voting
over four classifiers beat its best member in 4 of 10, mean ΔMCC −0.001
(−0.014 to +0.015). Indistinguishable. The app serves a single random forest;
`training/` rebuilds the consensus if you want to check.

## Usage

Upload a CSV with a `Smiles` column, or paste SMILES one per line. Output gives
the predicted class, the probability, the similarity to the nearest training
molecule, the scope band and its expected accuracy.

![The app with the built-in example set loaded: four compounds split across the
three scope bands, each with its prediction, similarity and nearest training
molecule](docs/screenshot.png)

The example set above is chosen to span the scope deliberately — caffeine sits
on top of its own training entry, a perfluorinated alcohol sits outside
anything in B3DB — so the difference the distance makes is visible in one
screen.

```bash
pip install -r requirements.txt
streamlit run BBBScope.py
```

## Reproducing everything

`data/` holds the curated structures, their labels and the scaffold partition
indices; `results/` holds every metric reported in the manuscript, including
the per-protocol predicted probabilities, so the figures can be redrawn without
refitting anything. The fingerprint matrices are not tracked because
`prepare_data.py` regenerates them from `data/molecules_dedup.csv`.
`training/requirements-lock.txt` pins the versions the published numbers were
produced under.

`training/` holds the full pipeline — curation, fingerprints, scaffold
partitioning, the three validation protocols, repeated partitions, the external
overlap analysis and model export.

```bash
python training/prepare_data.py          # curate, fingerprint, partition
python training/train_protocols.py       # three protocols, all models
python training/repeated_partitions.py   # confidence intervals
python training/check_external_overlap.py
python training/export_models.py         # export what the app serves
python training/make_logo.py             # rebuild the banner (needs matplotlib)
```

Hyperparameters are those of the originally published models, so the comparison
isolates validation protocol from modelling choices: random forest with 100
trees, SVC with C = 10 and γ = 0.1, k-nearest neighbours with k = 5, XGBoost with
100 rounds. No tuning was performed.

## Data

[B3DB](https://github.com/theochem/B3DB) (Meng et al., *Sci Data* **8** (2021)
289), 7 794 molecules after curation: 4 948 BBB+ and 2 846 BBB−, imbalance ratio
1.74.

One caveat inherited from the endpoint: binary labels derived from brain-to-plasma
ratios conflate passive permeability with active efflux. Atenolol and loratadine,
both poorly brain-penetrant in practice, are labelled permeable in B3DB. No
classifier trained on these labels can distinguish a compound that does not cross
from one that crosses and is pumped back out.

## Citation

A manuscript reporting the corrected analysis is in preparation. The preprint
([Research Square, 2023](https://www.researchsquare.com/article/rs-3752408/v1))
reports the uncorrected figures and is superseded.

## Licence

MIT. See [LICENSE](LICENSE).
