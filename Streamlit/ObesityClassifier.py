#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:45:18 2023

@author: Danesh Moradigaravand
"""

import streamlit as st
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import shap
import joblib
import numpy as np
#function for taxa
from PIL import Image


def read_filtered_file(file_inp):
    try:
        filtered_lines =[]
        max_pipe_count=2
        #file=file.split("\n")
        #with open(file_inp, 'r') as file:
            # Iterate over the lines and filter lines starting with "k__Bacteria"
        filtered_lines = [line for line in file_inp if line.startswith('k__Bacteria') and line.count('|') <= max_pipe_count]

        # Create two separate lists for column names and values, respectively
        df_data = []
        for item in filtered_lines:
            item = item.strip()
            if '|' in item:
                idx = item.rindex('|')
                strain = item[idx+1:item.index('\t')]
            else:
                strain = item[:item.index('\t')]
            frequency = float(item[item.index('\t')+1:])
            df_data.append({'strain': strain, 'frequency': frequency})

        # Return the DataFrame
        return pd.DataFrame(df_data)
    except FileNotFoundError:
        print("Error: The specified file path does not exist.")
        return None
    except Exception as e:
        print("Error:", e)
        return None
    


def get_unique_first_elements(file):
    unique_values = set()
    for line in file:
                # Ignore lines starting with #, UNINTEGRATED, and UNMAPPED
        if not line.startswith("#") and not line.startswith("UNINTEGRATED") and not line.startswith("UNMAPPED"):
                    # Extract the first element separated by : from other fields
            first_element = line.split(":")[0]
                    # Add the first element to the set of unique values
            unique_values.add(first_element)
    return unique_values



def process_data(taxa_file, pathway_file, predcitor_features_file) -> pd.DataFrame:

    taxa_input = read_filtered_file(taxa_file)
    unique_pathways = get_unique_first_elements(pathway_file)
        #predictor_features_file = pd.read_csv(predcitor_features_file)
    predictor_features_file = predcitor_features_file

    
    predictor_features_list = list(predictor_features_file.columns[2:])
    indices_pathway = [i for i, elem in enumerate(predictor_features_list) if "__" not in elem]
    predictor_features_list_pathway = [predictor_features_list[i] for i in indices_pathway]
    list_pathway = [int(elem in unique_pathways) for elem in predictor_features_list_pathway]
    indices_taxa = [i for i, elem in enumerate(predictor_features_list) if "__" in elem]
    predictor_features_list_taxa = [predictor_features_list[i] for i in indices_taxa]
    
    freq_list = [taxa_input.loc[taxa_input["strain"] == elem, "frequency"].values[0] if elem in taxa_input["strain"].tolist() else 0 for elem in predictor_features_list_taxa]

    values = list_pathway + freq_list
    output = pd.DataFrame([values], columns=predictor_features_file.columns[2:])
    return output

# Set title and subheader
st.title("Machine learning prediction of Obesity from Gut Microbiota Taxa and Pathway Profiles")
photo_path = "gut_microbiota_microbes.jpg"
photo_image = Image.open(photo_path)
st.image(photo_image, caption="Developed by Danesh Moradigaravand")

# Upload two input files
with st.sidebar:
    taxa_file = st.file_uploader("Upload Taxa File")
    pathway_file = st.file_uploader("Upload Pathway File")
    # Load and display the image on the front page
   


# Create button to process data
if st.sidebar.button("Prediction"):
    st.empty()
    if taxa_file is not None and pathway_file is not None:
        # Convert file objects to file paths
        taxa_bytes = taxa_file.read()
        
        stringio_taxa = StringIO(taxa_file.getvalue().decode("utf-8"))
        uploaded_file_input_taxa=stringio_taxa.read().splitlines()
        
        stringio_pathway = StringIO(pathway_file.getvalue().decode("utf-8"))
        uploaded_file_input_pathway=stringio_pathway.read().splitlines()
        
        # Load the third file
        terms_features_file = "terms_features.csv"

        # Read the third file
        terms_features_data = pd.read_csv(terms_features_file)
        model = joblib.load("model.joblib")
        processed_data = process_data(uploaded_file_input_taxa,uploaded_file_input_pathway,terms_features_data)
        
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_data)

        local_plot = shap.force_plot(explainer.expected_value, 
                             shap_values[0], 
                 features=processed_data.loc[0],
                 feature_names=processed_data.columns,
                 show=False, matplotlib=True)
        plt.savefig("force_plot.png",dpi=1000)
        
        shap_df=pd.DataFrame(list(np.squeeze(shap_values)), index=processed_data.columns,columns=['SHAP'])
        shap_df = shap_df.loc[shap_df['SHAP'] != 0]
        
        shap_df.to_csv("SHAP_output.csv")

        tmp_df=pd.DataFrame(model.predict_proba(processed_data))
        tmp_df.columns=['Non-Obese Probability','Obese Probability']
        

        #tmp_df.columns=['Non-Obese', 'Obese']
        st.dataframe(tmp_df)
    else:
        st.warning("Please upload both files.")

if st.sidebar.button("Show SHAP Importance Plot"):
        # Load and display the image
    st.empty()
    image = "force_plot.png"
    st.image(image, caption="Force Plot", width=1000)
        
if st.sidebar.button("Display Feature Importance Values"):
    st.empty()
    shap_data = pd.read_csv("SHAP_output.csv")  
    
    st.write("Predictors for Obesity:")
    filtered_df = shap_data.loc[shap_data['SHAP'] > 0]
    sorted_df = filtered_df.sort_values('SHAP', ascending=False)
    st.dataframe(sorted_df)
    
    st.write("Predictors for Non-Obesity:")
    filtered_df = shap_data.loc[shap_data['SHAP'] < 0]
    sorted_df = filtered_df.sort_values('SHAP', ascending=True)
    st.dataframe(sorted_df)







