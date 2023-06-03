# Machine Learning Prediction of Obesity from Taxa and Functional Profiles in Human Gut Microbiota
#### Danesh Moradigaravand
#### Under the supervision of Prof Babak Hossein Khalaj 

The human gut harbors a diverse community of microorganisms collectively known as the gut microbiota, which plays a pivotal role in various aspects of human health, encompassing metabolism and immune function. Recent research has begun to unravel the potential links between alterations in the gut microbiota and the development of metabolic disorders, including obesity. This article delves into the intricate relationship between the gut microbiota and obesity, shedding light on the existing evidence and potential mechanisms involved. Emerging evidence highlights the association between changes in gut microbiota composition and function and obesity, thereby suggesting that the gut microbiota's composition could hold significant predictive information for the early diagnosis and prognosis of obesity. In this project, we harnessed the power of machine learning algorithms to predict obesity status using taxa and functional information derived from the gut microbiota of 247 human samples. Our findings demonstrated high predictive accuracy (AUC-ROC = 0.95) and allowed for the identification of key taxa and pathways implicated in obesity. To make our predictive model accessible to the research community, we developed a user-friendly data science solution featuring both a command line interface and a graphical interface.


# Command Line Obesity Classifier

The Obesity Classifier is a command-line tool that allows you to process data files related to obesity classification. By leveraging machine learning models, it predicts obesity status based on taxa and pathway information.

## Usage

To use the Obesity Classifier, follow the syntax below:

### Options

The Obesity Classifier supports the following options:

- `-h, --help`: Show the help message and exit.
- `-t TAXA, --taxa TAXA`: Path to the taxa file.
- `-p PATHWAY, --pathway PATHWAY`: Path to the pathway file.
- `-m MODEL, --model MODEL`: Path to the model file.
- `-o OUTPUT_PROB, --output_prob OUTPUT_PROB`: Path to the output probability file.
- `-a OUTPUT_BINARY, --output_binary OUTPUT_BINARY`: Path to the output binary file.
- `-s SHAP_OUTPUT, --shap_output SHAP_OUTPUT`: Path to the SHAP output image file.
- `-f SHAP_DATAFRAME, --shap_dataframe SHAP_DATAFRAME`: Path to the SHAP output DataFrame file.
- `--developer`: Show the program's version number and exit.

## Run the Command

To run the Obesity Classifier, open your command-line interface and navigate to the directory where the `ObesityClassifier.py` file is located. Then, execute the following command:

```shell
python ObesityClassifier.py [OPTIONS]
```

# Streamlit Application

In addition to the command-line tool, the Obesity Classifier also provides a Streamlit application for a more interactive experience. To run the Streamlit application, navigate to the `Streamlit` folder in the project repository. Open your command-line interface and execute the following command:

```shell
streamlit run ObesityClassifier.py
```

