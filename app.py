import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
import pandas as pd
from padelpy import padeldescriptor



# Molecular descriptor calculator using padelpy
def calculate_descriptors(fingerprint_type):
    try:
        fp = {
            'Substructure': 'SubstructureFingerprinter.xml',
            'MACCS': 'MACCSFingerprinter.xml',
            'PubChem': 'PubChemFingerprinter.xml',
            'AtomPairs2D': 'AtomPairs2DFingerprinter.xml',
            'CDKextended': 'ExtendedFingerprinter.xml',
            'EState': 'EStateFingerprinter.xml'
        }
        fingerprint_output_file = f'{fingerprint_type}_output.csv'

        padeldescriptor(mol_dir='dataset.smi',
                        d_file=fingerprint_output_file,
                        descriptortypes=fp[fingerprint_type],
                        detectaromaticity=True,
                        standardizenitro=True,
                        standardizetautomers=True,
                        threads=2,
                        removesalt=True,
                        log=True,
                        fingerprints=True)

    except Exception as e:
        st.error(f"An error occurred during descriptor calculation: {str(e)}")
        st.stop()



# File download
def file_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    try:
        # Reads in EnsembleBBB
        model_filename = f'ensemble_{fingerprint_type}.pkl'
        load_model = pickle.load(open(model_filename, 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='BBB_class')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(file_download(df), unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model file exists.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during model prediction: {str(e)}")
        st.stop()

# Logo image
image = Image.open('logo.png')

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# EnsembleBBB: Enhanced accuracy in predicting drug blood-brain barrier permeability with a Machine Learning Ensemble model

This app allows you to predict the BBB permeability of small-molecules using 880 binary descriptors included in PubChem fingerprints.

**Credits**
- App built in `Python` + `Streamlit` by Yassir Boulaamane.
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).
---
""")

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

    fingerprint_type = st.sidebar.selectbox("Select Fingerprint Type", ["Substructure", "MACCS", "PubChem", "AtomPairs2D", "CDKextended", "EState"])


if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        load_data.to_csv('dataset.smi', sep='\t', header=False, index=False)

        st.header('**Original input data**')
        st.write(load_data)

        with st.spinner("Calculating descriptors..."):
            calculate_descriptors(fingerprint_type)

        # Read in calculated descriptors and display the dataframe
        st.header('**Calculated molecular descriptors**')
        desc = pd.read_csv(f'{fingerprint_type}_output.csv')
        descriptor_list_filename = f'{fingerprint_type}.csv'
        st.write(desc)
        st.write(desc.shape)

        # Read descriptor list used in previously built model
        st.header('**Subset of descriptors from previously built models**')
        Xlist = list(pd.read_csv(descriptor_list_filename, index_col=0).columns)
        desc_subset = desc[Xlist]
        st.write(desc_subset)
        st.write(desc_subset.shape)
    

        # Apply trained model to make prediction on query compounds
        build_model(desc_subset)
    else:
        st.warning('Please upload a file to proceed.')
else:
    st.info('Upload input data in the sidebar to start!')
