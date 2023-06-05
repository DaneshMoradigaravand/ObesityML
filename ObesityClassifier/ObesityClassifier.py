#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:39:49 2023

@author: Danesh Moradigaravand
"""

import argparse
import pandas as pd
import shap
import matplotlib.pyplot as plt
#function for pathways
import os
import joblib
import numpy as np
#function for taxa
import warnings

# Disable the warning for the deprecation of ntree_limit
warnings.filterwarnings("ignore", category=DeprecationWarning, module="xgboost", message=".*ntree_limit is deprecated.*")

def read_filtered_file(file_path):
    try:
        filtered_lines =[]
        max_pipe_count=2
        with open(file_path, 'r') as file:
            # Iterate over the lines and filter lines starting with "k__Bacteria"
            filtered_lines = [line for line in file if line.startswith('k__Bacteria') and line.count('|') <= max_pipe_count]

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

def get_unique_first_elements(file_path):
    # Initialize a set to store the unique first element values
    unique_values = set()

    # Check if the file is a regular file
    if os.path.isfile(file_path):
        # Open the file
        with open(file_path, "r") as file:
            # Loop through every line in the file
            for line in file:
                # Ignore lines starting with #, UNINTEGRATED, and UNMAPPED
                if not line.startswith("#") and not line.startswith("UNINTEGRATED") and not line.startswith("UNMAPPED"):
                    # Extract the first element separated by : from other fields
                    first_element = line.split(":")[0]
                    # Add the first element to the set of unique values
                    unique_values.add(first_element)
    return unique_values



def process_data(taxa_file: str, pathway_file: str, predcitor_features_file: str) -> pd.DataFrame:
    try:
        taxa_input = read_filtered_file(taxa_file)
        unique_pathways = get_unique_first_elements(pathway_file)
        predictor_features_file = pd.read_csv(predcitor_features_file)
    except FileNotFoundError:
        print("File not found")
        return None
    
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

terms_features_file = "terms_features.csv"
        # Read the third file

parser = argparse.ArgumentParser(prog='Metagenomic Obesity Classifier',
                                 description='A classifier for predicting obesity based on metagenomic data.',
                                 epilog='The classifier uses taxa and pathway files as inputs, which are the outputs of Metaphlan and Humann pipelines, respectively. The outputs provide interpretable features showing the contribution of each predictor to the output features.',
                                 add_help=False)

# Create the command-line argument parser
parser = argparse.ArgumentParser(description='Process data files.')
parser.add_argument('-t', '--taxa', type=str, help='Path to the taxa file.')
parser.add_argument('-p', '--pathway', type=str, help='Path to the pathway file.')
parser.add_argument('-m', '--model', type=str, default='model.joblib', help='Path to the model file.')
parser.add_argument('-o', '--output_prob', type=str, default='output_prob.csv', help='Path to the output file.')
parser.add_argument('-a', '--output_binary', type=str, default='output_binary.csv', help='Path to the output file.')
parser.add_argument('-s', '--shap_output', type=str, default='SHAP_output.png', help='Path to the SHAP output image file.')
parser.add_argument('-f', '--shap_dataframe', type=str, default='SHAP_dataframe.csv', help='Path to the SHAP output DataFrame file.')

# Add the developer name
parser.add_argument('--developer', action='version', version='%(prog)s by Danesh Moradigaravand')

# Enable the default help option
args = parser.parse_args()

# Extract the file paths from the command-line arguments
taxa_file_path = args.taxa
pathway_file_path = args.pathway
model_file_path = args.model
output_file_path = args.output_prob
output_point_file_path = args.output_binary
shap_output_file_path = args.shap_output
shap_dataframe_file_path = args.shap_dataframe

model = joblib.load(model_file_path)

# Check if the required arguments are provided
if taxa_file_path is None or pathway_file_path is None:
    parser.error('Please provide both taxa and pathway file paths using the -t and -h parameters.')

# Call the process_data function with the provided file paths
output = process_data(taxa_file_path, pathway_file_path, terms_features_file )
output_df=pd.DataFrame(model.predict_proba(output), columns=["NonObese","Obese"])
output_df.to_csv(output_file_path, index=False)
print(f"Output probability saved to: {output_file_path}")

output_label="None-Obese"
if model.predict(output)[0]==0:
    output_label="Obese"
pd.DataFrame([output_label],columns=["Result"]).to_csv(output_point_file_path)
print(f"Output point saved to: {output_file_path}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(output)

local_plot = shap.force_plot(explainer.expected_value, 
                      shap_values[0], 
          features=output.loc[0],
          feature_names=output.columns,
          show=False, matplotlib=True)
plt.savefig(shap_output_file_path,dpi=500)
print(f"SHAP output saved to: {shap_output_file_path}")
 
shap_df=pd.DataFrame(list(np.squeeze(shap_values)), index=output.columns,columns=['SHAP'])
shap_df = shap_df.loc[shap_df['SHAP'] != 0]
shap_df.to_csv(shap_dataframe_file_path)
print(f"SHAP DataFrame saved to: {shap_dataframe_file_path}")