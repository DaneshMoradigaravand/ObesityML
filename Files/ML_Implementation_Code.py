#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 3 23:13:51 2023

Author: Danesh Moradigaravand
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import shap

# Data reading and cleansing
df = pd.read_csv("ML_input_data.csv")
df = df.drop(["Isolates"], axis=1)

Y = df["label"]
drop_cols = ["label"]
X = df.loc[:, [x for x in df.columns if x not in drop_cols]]

# Transform Y and Split the data
lb = LabelBinarizer()
Y = lb.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, stratify=Y)

# Logistic Regression
param_grid = {'penalty': ['l2'], 'C': np.logspace(-4, 4, 20)}
logistic = LogisticRegression()
folds = 3
param_comb = 5
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
random_search_logistic = RandomizedSearchCV(logistic, param_distributions=param_grid, n_iter=param_comb,
                                            scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, Y_train),
                                            verbose=3, random_state=1001)
random_search_logistic.fit(X_train, Y_train)
pred_class_logistic = random_search_logistic.predict(X_test)
pred_probs_logistic = random_search_logistic.predict_proba(X_test)

# XGBoost
params = {'min_child_weight': [1, 5], 'gamma': [0.5, 1.5, 5], 'subsample': [0.8, 1.0],
          'colsample_bytree': [0.8, 1.0], 'max_depth': [3, 5]}
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
random_search_xgboost = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb,
                                           scoring='roc_auc', n_jobs=4, cv=skf.split(X_train, Y_train),
                                           verbose=3, random_state=1001)
random_search_xgboost.fit(X_train, Y_train)
pred_class_xgboost = random_search_xgboost.predict(X_test)
pred_probs_xgboost = random_search_xgboost.predict_proba(X_test)

# Plot ROC curve
fpr_xgboost, tpr_xgboost, thresholds_xgboost = roc_curve(Y_test, pred_probs_xgboost[:, 1])
fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(Y_test, pred_probs_logistic[:, 1])
plt.plot(fpr_xgboost, tpr_xgboost, label='Gradient Boosted Decision Trees (AUC = {:.2f})'.format(auc(fpr_xgboost, tpr_xgboost)), linewidth=2)
plt.plot(fpr_logistic, tpr_logistic, label='Logistic Regression (AUC = {:.2f})'.format(auc(fpr_logistic, tpr_logistic)), color='red', linewidth=2)
plt.plot([0, 1], [0, 1], color='orange', linestyle='--', linewidth=2)  # Add diagonal orange line
plt.title('Receiver Operating Characteristic', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion matrix evaluation
def confusion_matrix_eval(Y_truth, Y_pred):
    cm = confusion_matrix(Y_test, Y_pred)
    cr = classification_report(Y_test, Y_pred)
    print("-" * 90)
    print("[CLASS_REPORT] Printing classification report to console")
    print("-" * 90)
    print(cr)
    print("-" * 90)
    return [cm, cr]

cm = confusion_matrix_eval(Y_test, pred_class_logistic)
pd.DataFrame(cm)

# SHAP analysis
explainer = shap.Explainer(random_search_xgboost.best_estimator_)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
X_df = pd.DataFrame(X)
shap.summary_plot(shap_values, X_df, plot_type="bar")
pd.DataFrame(shap_values, columns=X_df.columns).to_csv("SHAP_report.csv")
