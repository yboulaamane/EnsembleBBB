import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
import jdk

# Molecular descriptor calculator
def calculate_descriptors():
    try:
        java_path = "java"  # Replace with your actual path
        bash_command = "f{java_path} -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            st.error(f"Error in descriptor calculation: {error.decode()}")
            st.stop()
        os.remove('molecule.smi')
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
        # Dropbox shareable link to the model file
        dropbox_link = 'https://www.dropbox.com/scl/fi/6h79b2crwxwo1zpi9kx6a/ensembleBBB.pkl?rlkey=pcy8l762k0mzqhl415f61ed8f&dl=0'

        # Download the model file
        response = requests.get(dropbox_link)
        with open('ensembleBBB.pkl', 'wb') as file:
            file.write(response.content)

        # Load the model
        load_model = pickle.load(open('ensembleBBB.pkl', 'rb'))
        load_model = pickle.load(open('ensembleBBB.pkl', 'rb'))
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

if st.sidebar.button('Predict'):
    if uploaded_file is not None:
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

        st.header('**Original input data**')
        st.write(load_data)

        with st.spinner("Calculating descriptors..."):
            calculate_descriptors()

        # Read in calculated descriptors and display the dataframe
        st.header('**Calculated molecular descriptors**')
        desc = pd.read_csv('descriptors_output.csv')
        st.write(desc)
        st.write(desc.shape)

        # Read descriptor list used in previously built model
        st.header('**Subset of descriptors from previously built models**')
        Xlist = list(pd.read_csv('descriptor_list.csv').columns)
        desc_subset = desc[Xlist]
        st.write(desc_subset)
        st.write(desc_subset.shape)

        # Apply trained model to make prediction on query compounds
        build_model(desc_subset)
    else:
        st.warning('Please upload a file to proceed.')
else:
    st.info('Upload input data in the sidebar to start!')
