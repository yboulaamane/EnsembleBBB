# EnsembleBBB

Blood-brain barrier permeability prediction from chemical structure, with an
applicability domain and honest validation.

**Live app:** http://ensemblebbb.streamlit.app/

---

## What changed in this release

Earlier versions of this repository reported AUC around 0.95–0.96 and 95 %
accuracy on an "external" set. Those figures were inflated, and we corrected
them. Two faults were responsible:

1. **The minority class was oversampled before the data were partitioned.**
   Copies of the same molecule therefore appeared on both sides of every split
   and every cross-validation fold.

2. **The partition was random**, over a dataset in which roughly half the
   molecules share a descriptor vector with another molecule. B3DB contains
   large families of stereoisomers, and Morgan, MACCS and Avalon fingerprints
   are computed without stereochemistry by default. Under a random split, 63 %
   of test compounds had a bit-for-bit identical twin in training, and the
   median test compound had Tanimoto similarity **1.00** to its nearest
   training neighbour.

A third problem affected the external evaluation: **68 % of the BBBP "external"
set is present in B3DB as identical structures**, so that benchmark was mostly
measuring memorisation.

Corrected figures, random forest on MACCS keys with scaffold-split validation:

| | AUC | MCC | Accuracy |
|---|---|---|---|
| Held-out (scaffold split) | 0.943 | 0.742 | 0.894 |
| Genuinely external compounds | 0.954 | 0.775 | 0.909 |

We also found that **consensus modelling does not help here.** Soft voting over
four classifiers beats its own best member in only 3 of 12 comparisons, with a
mean MCC difference of −0.006. The random forest alone is the better model, and
it is what the app now serves.

The consensus models are no longer shipped. Besides adding nothing, they are
not practical to distribute: the k-nearest-neighbours member stores the entire
oversampled training array, roughly 150 MB per fingerprint. `training/` rebuilds
them in full for anyone who wants to verify the comparison.

## Applicability domain

Accuracy depends strongly on how close a compound is to the training set:

| Tanimoto to nearest training molecule | Accuracy |
|---|---|
| < 0.3 | 76 % |
| 0.3 – 0.5 | 88 % |
| > 0.8 | 95 % |

Every prediction reports this similarity, and anything below 0.5 is flagged.

## Usage

Upload a CSV with a `Smiles` column, or paste SMILES one per line. Choose a
fingerprint and model, then press Predict. Output includes the predicted class,
the probability, the similarity to the nearest training molecule, and whether
the compound falls inside the applicability domain.

## Running locally

```bash
pip install -r requirements.txt
streamlit run EnsembleBBB.py
```

## Reproducing the models

`training/` contains the full pipeline: curation, fingerprint generation,
scaffold partitioning, the three validation protocols, and model export.

```bash
python training/prepare_data.py      # curate, fingerprint, partition
python training/train_protocols.py   # all three protocols, all models
python training/export_models.py     # export what the app serves
```

Hyperparameters are those of the originally published models, so the comparison
isolates validation protocol from modelling choices: random forest with 100
trees, SVC with C = 10 and gamma = 0.1, k-nearest neighbours with k = 5, and
XGBoost with 100 rounds. No tuning was performed.

## Data

[B3DB](https://github.com/theochem/B3DB) (Meng et al.), 7 794 molecules after
curation: 4 948 BBB+ and 2 846 BBB−, an imbalance ratio of 1.74.

## Citation

A manuscript reporting the corrected analysis is in preparation. Until it
appears, cite the repository and note that the preprint
([Research Square, 2023](https://www.researchsquare.com/article/rs-3752408/v1))
reports the uncorrected figures and has been superseded.

## Licence

MIT. See [LICENSE](LICENSE).
