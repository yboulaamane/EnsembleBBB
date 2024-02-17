import streamlit as st
import pandas as pd
import requests
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
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
                fingerprint = list(fingerprint.ToBitString())
            elif fingerprint_type == 'MACCS fingerprints':
                fingerprint = MACCSkeys.GenMACCSKeys(mol)
                fingerprint = list(fingerprint.ToBitString())
            elif fingerprint_type == 'Avalon fingerprints':
                fingerprint = GetAvalonFP(mol, nBits=2048)      
                fingerprint = list(fingerprint.ToBitString())
            elif fingerprint_type == 'Topological Torsion fingerprints':
                fingerprint = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)    
                fingerprint = list(fingerprint.ToBitString())               
            else:
                st.error("Invalid fingerprint type selected.")
                return
        else:
            fingerprint = [None] * 2048  # Placeholder for invalid molecules
        fingerprints.append(fingerprint)

    return fingerprints

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="ensemblebbb_prediction_results.csv">Download Predictions</a>'
    return href

# Model download
def load_model():
    model_url = 'https://www.dropbox.com/scl/fi/...ensemble_maccs.pkl?rlkey=...&dl=1'  # your URL
    
    st.write("Downloading model...")
    response = requests.get(model_url, stream=True) 

    if response.status_code == 200:
        with open('model.pkl', 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        st.write("Model downloaded successfully!")
        return pickle.load(open('model.pkl', 'rb')) 
    else:
        st.error("Failed to download model.")
        return None

# Model prediction
def predict_model(features, model):
    model = load_model()
    if model is not None:
        prediction = load_model.predict(features)
        return prediction

# Logo image
image = Image.open('logo.png')
st.image(image, use_column_width=True)

# Page title
st.markdown("""
# BBB Permeability Prediction App

This app allows you to predict the blood-brain barrier permeability of drugs based on Morgan (ECFP4), MACCS, Avalon or Topological Torsion fingerprints. The ensemble models were built using four machine learning classifiers (RF, SVC, kNN, and XGB) using the [B3DB dataset](https://www.nature.com/articles/s41597-021-01069-5) with 7996 molecules.

**Credits**
- App built in `Python` + `Streamlit` by [Yassir Boulaamane](https://yboulaamane.github.io/).
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

# Adding an input box for smiles
smiles_input = st.sidebar.text_area("Enter list of SMILES of molecules (one per line)", height=200)

st.sidebar.markdown("""
Example:
""")
st.sidebar.markdown("""
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
""")

st.sidebar.header('2. Choose fingerprint type')
fingerprint_type = st.sidebar.selectbox('Choose fingerprint type', ['Morgan fingerprints', 'MACCS fingerprints', 'Avalon fingerprints', 'Topological Torsion fingerprints'])


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
        
        # Read descriptor list based on the selected fingerprint type
        if fingerprint_type == 'Morgan fingerprints':
            descriptor_list = pd.read_csv('b3db_morgan_fp.csv')
        elif fingerprint_type == 'MACCS fingerprints':
            descriptor_list = pd.read_csv('b3db_maccs_fp.csv')
        elif fingerprint_type == 'Avalon fingerprints':
            descriptor_list = pd.read_csv('b3db_avalon_fp.csv')
        elif fingerprint_type == 'Topological Torsion fingerprints':
            descriptor_list = pd.read_csv('b3db_topological_torsion_fp.csv')
        else:
            st.error("Invalid fingerprint type selected.")
            st.stop()
            
        # Display the DataFrame of the descriptor lists
        st.subheader(f'Descriptor List for {fingerprint_type}')
        st.write(descriptor_list)
        st.write("Shape of Descriptors:")
        st.write(descriptor_list.shape)        

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
        elif fingerprint_type == 'Avalon fingerprints':
            num_bits = 2048
            fingerprint_type_name = 'Avalon'
        elif fingerprint_type == 'Topological Torsion fingerprints':
            num_bits = 2048
            fingerprint_type_name = 'TopologicalTorsion'
        else:
            st.error("Invalid fingerprint type selected.")
            st.stop()

        # Create feature names consistent with the model's expectations
        
        feature_names = [f'{fingerprint_type_name}_{i}' for i in range(num_bits)]
        

        # Create DataFrame outside the if-else block
        features_df = pd.DataFrame(features, columns=feature_names)

        # Apply trained model to make prediction on query compounds
        Xlist = list(descriptor_list.columns)  # Store feature names for subsetting

        
        # Select subset of descriptors based on the chosen fingerprint type
        st.subheader(f'Calculating Descriptors for input molecules')
        desc_subset = features_df[Xlist]
        # Convert columns to numeric types
        desc_subset = desc_subset.apply(pd.to_numeric, errors='coerce')

        # Display subset of descriptors and its shape
        st.write("Subset of Descriptors:")
        st.write(desc_subset)
        st.write("Shape of Subset:")
        st.write(desc_subset.shape)
        
        # Save descriptors to CSV file
        descriptors_output_file = "descriptors_output.csv"
        features_df.to_csv(descriptors_output_file, index=False)
        st.success(f"Descriptors saved to {descriptors_output_file}")

        # Load previously built ensemble model
        if fingerprint_type == 'Morgan fingerprints':
            model = 'ensemble_morgan.pkl'
        elif fingerprint_type == 'MACCS fingerprints':
            model = 'https://www.dropbox.com/scl/fi/vrgatw9p5bo4tmli3oiyz/ensemble_maccs.pkl?rlkey=u644udi20ajjv4x11xyvln5td&dl=1'
        elif fingerprint_type == 'Avalon fingerprints':
            model = 'ensemble_avalon.pkl'
        elif fingerprint_type == 'Topological Torsion fingerprints':
            model = 'ensemble_topological_torsion.pkl'
        else:
            st.error("Invalid fingerprint type selected.")
            st.stop()

        # Apply trained model to make prediction on query compounds
        prediction = predict_model(desc_subset, model)
        
        # Map numeric predictions to BBB labels
        prediction_labels = ['BBB+' if pred == 1 else 'BBB-' for pred in prediction]

        # Prepare output DataFrame
        output_df = pd.DataFrame({
            'Smiles': load_data['Smiles'],
            'Prediction': prediction_labels
        })

        st.header('**Prediction output**')
        st.write(output_df)
        st.markdown(filedownload(output_df), unsafe_allow_html=True)
    else:
        st.warning("Please upload a CSV file or provide smiles of molecules.")
else:
    st.info('Upload input data in the sidebar to start!')
