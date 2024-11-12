'''
Author: Sophia Wagner
Date: 8/9/2024
Description: using TPOT to find the best model for the full term deposit dataset
'''

# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from tpot.config import classifier_config_dict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, recall_score, precision_score
# models to compare for the full dataset model:
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#load in data
data = pd.read_csv('no-call-one-hot-term-deposit-data.csv')
data_full = pd.read_csv('one-hot-term-deposit-data.csv')

# quick look at the data, printing y value counts
#print(data['y'].value_counts()) #no: 37104, yes: 2896

# separate features and target variable
X = data.drop('y', axis=1)
y = data['y']

X_full = data_full.drop('y', axis=1)
y_full = data_full['y']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_tr_full, X_te_full, y_tr_full, y_te_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# SMOTE-ENN on full dataset training split
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_smtenn, y_train_smtenn = smote_enn.fit_resample(X_tr_full, y_tr_full)

# RUS on no-call dataset training split
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, replacement=False)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

X_tr_full_rus, y_tr_full_rus = rus.fit_resample(X_tr_full, y_tr_full)
'''
The models to compare for the full dataset model:
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
'''
##############################
# now getting into the TPOT pipeline

#creating a custom config dictionary
# custom_config = {
#     'sklearn.tree.DecisionTreeClassifier': classifier_config_dict['sklearn.tree.DecisionTreeClassifier'],
#     'sklearn.ensemble.GradientBoostingClassifier': classifier_config_dict['sklearn.ensemble.GradientBoostingClassifier']
# }

# creating classifier
tpot = TPOTClassifier(generations=5, population_size=50, 
                      verbosity=2, scoring='precision', #scoring='recall',
                      cv= 5, random_state=42, config_dict='TPOT light',
                      disable_update_check=True)

#fitting tpot model
tpot.fit(X_train_smtenn, y_train_smtenn)
# tpot.fit(X_tr_full_rus, y_tr_full_rus)

# tpot_recall_score = recall_score(y_test, tpot.predict(X_test))
# print(f'\n TPOT recall score: {tpot_recall_score}')
tpot_precision_score = precision_score(y_te_full, tpot.predict(X_te_full))
print(f'\n TPOT precision score: {tpot_precision_score}')

print('\n Best pipeline steps: ', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f'{idx}. {transform}')

# scoring the model
# print('TPOT no-call dataset training score: ', tpot.score(X_test, y_test))
print('TPOT full dataset testing score: ', tpot.score(X_te_full, y_te_full))

