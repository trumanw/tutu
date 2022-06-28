import glob
import random
import os
from os import listdir
from os.path import isfile, join

import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from rdkit.Chem import PandasTools

from tutu.surfactant import predict

# Streamlit entrypoint
st.sidebar.markdown("# CMC Prediction")
st.write("""
# Surfactants CMC Prediction App
This app predicts the **surfactants CMC** via GCN model!
""")
st.write('---')

# Loads model & datasets
def load_cmc_models(model_dir='models/surfactant/cmc', model_suffix=".pth.tar"):
    prefix_abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # iterate all the .pth.tar files
    model_ckpts = [join(model_dir, f) for f in listdir(os.path.join(prefix_abspath, model_dir)) if isfile(join(model_dir, f)) and f.endswith(model_suffix)]
    return model_ckpts

model_select = st.selectbox(
    label ="CMC Model select",
    options=load_cmc_models()
)

# Visualization 
uploaded_smiles_file = st.file_uploader("Choose a file")
if uploaded_smiles_file is not None:
    # To read file as bytes:
    smiles_df = pd.read_csv(uploaded_smiles_file)
    st.write(smiles_df)

    # show prediction 
    if st.button('Run Prediction'):
        st.header('CMC Prediction')
        st.write('Using model: ', model_select)
        pred_cmc_vals = []
        pred_smi_rdkit_img = []
        for smi in smiles_df.smi:
            pred_cmc = predict(smi, model_select)[0]
            pred_cmc_vals.append(float('%.8f'%pred_cmc))
        smiles_df['pred_log_cmc'] = pred_cmc_vals
        PandasTools.RenderImagesInAllDataFrames(images=True)
        PandasTools.AddMoleculeColumnToFrame(smiles_df, 'smi', includeFingerprints=False)
        st.write(smiles_df.to_html(escape=False), unsafe_allow_html=True)