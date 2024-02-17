## EnsembleBBB: BBB Permeability Prediction App
![image](https://github.com/yboulaamane/EnsembleBBB/assets/7014404/b3485a62-1219-42f1-8484-aa1fb3d71b18)

EnsembleBBB is a web-based application built with Streamlit that enables users to predict the blood-brain barrier (BBB) permeability of drug molecules. Try the app live at: http://ensemblebbb.streamlit.app/. It utilizes an ensemble of machine learning models (Random Forest, Support Vector Machine, k-Nearest Neighbors, and XGBoost) trained on the [B3DB dataset](https://github.com/theochem/B3DB/blob/main/B3DB) and calculates Morgan (ECFP4), MACCS, Avalon, and Topological Torsion fingerprints as molecular descriptors.

## Key Features

Provides a simple interface to:
- Upload a CSV file containing the SMILES representations of molecules.
- Enter SMILES manually for quick predictions.
- Select the desired fingerprint calculation method.
- Accurate Predictions: Leverages the predictive power of ensemble modeling techniques.
- Downloadable Results: Enables users to download a CSV file containing their BBB permeability predictions for further analysis or record-keeping.

## Usage

- Upload Data: Upload a CSV file with a "Smiles" column, or enter SMILES in the designated text area.
- Choose Fingerprint Type: Select the desired fingerprint (Morgan, MACCS, Avalon, or Topological Torsion).
- Predict: Click the "Predict" button.
- Download (Optional): Download the generated CSV file containing predicted BBB permeability.
  
## Technology

- Python: Core programming language
- Streamlit: Building the web application
- scikit-learn: Machine learning models
- RDKit: Fingerprint calculation
- Pandas: Data handling

## Data

The ensemble models are trained on the B3DB dataset https://www.nature.com/articles/s41597-021-01069-5.

## Cite Our Work

If you find this app useful, please cite our preprint:

Boulaamane, Y., & Maurady, A. (2023). EnsembleBBB: Enhanced accuracy in predicting drug blood-brain barrier permeability with a Machine Learning Ensemble model.

## Contributions

We welcome contributions! If you have ideas for improvements or bug fixes, please open an issue or a pull request on the GitHub repository [https://github.com/yboulaamane/EnsembleBBB].
