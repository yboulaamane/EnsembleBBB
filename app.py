import streamlit as st
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
import base64
import pickle

# Feature calculation using RDKit
def calculate_fingerprints(smiles, fingerprint_type):
    fingerprints = []

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            if fingerprint_type == 'Morgan fingerprints':
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            elif fingerprint_type == 'MACCS fingerprints':
                fingerprint = AllChem.GetMACCSKeysFingerprint(mol)
            else:
                st.error("Invalid fingerprint type selected.")
                return
            fingerprint = list(fingerprint.ToBitString())
        else:
            fingerprint = [None] * 2048  # Placeholder for invalid molecules
        fingerprints.append(fingerprint)

    return fingerprints

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model prediction
def predict_model(features, model):
    load_model = pickle.load(open(model, 'rb'))
    prediction = load_model.predict(features)
    return prediction

# Logo image
image = Image.open('logo.png')
st.image(image, use_column_width=True)

# Page title
st.markdown("""
# BBB Permeability Prediction App

This app allows you to predict the blood-brain barrier permeability of drugs based on Morgan or MACCS fingerprints. The ensemble models were built using four classifiers on the B3DB dataset.

**Credits**
- App built in `Python` + `Streamlit` by Yassir Boulaamane.
- Descriptor calculated using [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html) for fingerprint generation.  

Cite our preprint:  
Boulaamane, Y., & Maurady, A. (2023). EnsembleBBB: Enhanced accuracy in predicting drug blood-brain barrier permeability with a Machine Learning Ensemble model.

---
""")



# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['csv'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/yboulaamane/EnsembleBBB/main/example.csv)
""")
    fingerprint_type = st.sidebar.selectbox('2. Choose fingerprint type', ['Morgan fingerprints', 'MACCS fingerprints'])

# Adding an input box for smiles
smiles_input = st.sidebar.text_area("Enter smiles of molecules (one per line)", height=200)

if st.sidebar.button('Predict'):
    # Check if either the file or smiles_input is provided
    if uploaded_file is not None or smiles_input:
        # If file is uploaded, load the data from the file, else create a DataFrame from the smiles_input
        if uploaded_file is not None:
            load_data = pd.read_csv(uploaded_file)
        else:
            smiles_list = smiles_input.split("\n")
            smiles_list = [smile.strip() for smile in smiles_list if smile.strip()]  # Remove empty lines
            load_data = pd.DataFrame({'Smiles': smiles_list})

        if 'Smiles' not in load_data.columns:
            st.error("Input file must contain 'Smiles' column.")
            st.stop()

        st.header('**Original input data**')
        st.write(load_data)

        # Calculate descriptors for the loaded data
        with st.spinner("Calculating descriptors..."):
            features = calculate_fingerprints(load_data['Smiles'], fingerprint_type)

        # Convert features to DataFrame
        if fingerprint_type == 'Morgan fingerprints':
            num_bits = 2048
            fingerprint_type_name = 'Morgan'
        elif fingerprint_type == 'MACCS fingerprints':
            num_bits = 167
            fingerprint_type_name = 'MACCS'
        else:
            st.error("Invalid fingerprint type selected.")
            st.stop()

        # Create feature names consistent with the model's expectations
        feature_names = [f'{fingerprint_type_name}FP_{i}' for i in range(num_bits)]

        # Create DataFrame outside the if-else block
        features_df = pd.DataFrame(features, columns=feature_names)

        # Save descriptors to CSV file
        descriptors_output_file = "descriptors_output.csv"
        features_df.to_csv(descriptors_output_file, index=False)
        st.success(f"Descriptors saved to {descriptors_output_file}")

        # Read descriptor list used in previously built model
        if fingerprint_type == 'Morgan fingerprints':
            model = 'ensemble_morgan.pkl'
        elif fingerprint_type == 'MACCS fingerprints':
            model = 'ensemble_maccs.pkl'
        else:
            st.error("Invalid fingerprint type selected.")
            st.stop()

        # Apply trained model to make prediction on query compounds
        prediction = predict_model(features_df, model)
        
        # Map numeric predictions to BBB labels
        prediction_labels = ['BBB+' if pred == 1 else 'BBB-' for pred in prediction]

        # Prepare output DataFrame
        output_df = pd.DataFrame({
            'Smiles': load_data['Smiles'],
            'Prediction': prediction_labels
        })

        # Prepare output DataFrame
        output_df = pd.DataFrame({
            'Smiles': load_data['Smiles'],
            'BBB Class': prediction_labels
        })

        st.header('**Prediction output**')
        st.write(output_df)
        st.markdown(filedownload(output_df), unsafe_allow_html=True)
    else:
        st.warning("Please upload a CSV file or provide smiles of molecules.")
else:
    st.info('Upload input data in the sidebar to start!')
