'''
Author: Sophia Wagner
Date: 7/19/2024
Description: using the pycaret library to build a ML model for
             the Term Deposit marketing dataset
'''

# Importing the libraries
from pycaret.classification import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#load in data
### not used the orginal data, only the one-hot encoded datasets
one_hot_data = pd.read_csv('one-hot-term-deposit-data.csv')
no_call_data = pd.read_csv('no-call-one-hot-term-deposit-data.csv')

# quick look at the data
print(one_hot_data.head(5))     #size: 40000 x 35 (including target var)
print(no_call_data.head(5))     #size: 40000 x 28 (including target var)

''' 
DO NOT NEED THIS CODE BELOW FOR WHAT WE ARE DOING IN PYCARET
#splitting the data into features and target variable
X_one_hot = one_hot_data.drop('y', axis=1).values #complete dataset, one-hot encoded version
y_one_hot = one_hot_data['y'].values 
print(X_one_hot)
print(y_one_hot)
X_no_call = no_call_data.drop('y', axis=1).values #NO call related features, one-hot encoded version
y_no_call = no_call_data['y'].values
print(X_no_call)
print(y_no_call)
'''

# creating a train test split for each dataset
train_one_hot, test_one_hot = train_test_split(one_hot_data, test_size = 0.2, random_state=0, stratify=one_hot_data['y'])

train_no_call, test_no_call = train_test_split(no_call_data, test_size = 0.2, random_state=0, stratify=no_call_data['y'])

#count the number of 1 and 0 in 'y' in the training and testing sets
print("The number of 1s and 0s in the training and testing sets for the one-hot encoded data:")
print(f"training set: {train_one_hot['y'].value_counts()}")
print(f"testing set: {test_one_hot['y'].value_counts()}")

print("The number of 1s and 0s in the training and testing sets for the no call one-hot encoded data:")
print(f"training set: {train_no_call['y'].value_counts()}")
print(f"testing set: {test_no_call['y'].value_counts()}")

#set up pycaret environment
#### setup is already imported
# build a clasifier w the training data and generate a profile
clf_allfeats = setup(data=train_one_hot, target='y', experiment_name='term_deposit')
clf_nocall = setup(data=train_no_call, target='y', experiment_name='term_deposit_no_call')

#PyCaret will not train all models and list perormance metrics
models_allfeats = compare_models()
models_nocall = compare_models()

#picking the best model based on different metrics
best_allfeats_acc = automl(optimize = 'Accuracy')
best_allfeast_f1 = automl(optimize = 'F1')
best_allfeast_auc = automl(optimize = 'AUC')

best_nocall_acc = automl(optimize = 'Accuracy')
best_nocall_f1 = automl(optimize = 'F1')
best_nocall_auc = automl(optimize = 'AUC')

#printing these model choices
print(f"best model, all features, accuracy: {best_allfeats_acc}")
print(f"best model, all features, f1: {best_allfeast_f1}")
print(f"best model, all features, auc: {best_allfeast_auc}")

print(f"best model, no call features, accuracy: {best_nocall_acc}")
print(f"best model, no call features, f1: {best_nocall_f1}")
print(f"best model, no call features, auc: {best_nocall_auc}")

#fine-tuning the best model
tuned_allfeats_acc = tune_model(best_allfeats_acc)
tuned_allfeats_f1 = tune_model(best_allfeast_f1)
tuned_allfeats_auc = tune_model(best_allfeast_auc)

tuned_nocall_acc = tune_model(best_nocall_acc)
tuned_nocall_f1 = tune_model(best_nocall_f1)
tuned_nocall_auc = tune_model(best_nocall_auc)

#print the tuned models
print(f"tuned model, all features, accuracy: {tuned_allfeats_acc}")
print(f"tuned model, all features, f1: {tuned_allfeats_f1}")
print(f"tuned model, all features, auc: {tuned_allfeats_auc}")

print(f"tuned model, no call features, accuracy: {tuned_nocall_acc}")
print(f"tuned model, no call features, f1: {tuned_nocall_f1}")
print(f"tuned model, no call features, auc: {tuned_nocall_auc}")

#now making predictions and evaluating the model
pred_allfeats_accmodel = predict_model(tuned_allfeats_acc, data=test_one_hot)
pred_allfeats_f1model = predict_model(tuned_allfeats_f1, data=test_one_hot)
pred_allfeats_aucmodel = predict_model(tuned_allfeats_auc, data=test_one_hot)

pred_nocall_accmodel = predict_model(tuned_nocall_acc, data=test_no_call)
pred_nocall_f1model = predict_model(tuned_nocall_f1, data=test_no_call)
pred_nocall_aucmodel = predict_model(tuned_nocall_auc, data=test_no_call)

print(pred_allfeats_accmodel)

#print the evaluation metrics
print("Evaluation metrics for the all features models:")
print("starting with the accuracy model: \n")
print(f"accuracy: {accuracy_score(pred_allfeats_accmodel['y'], pred_allfeats_accmodel['prediction_label'])}")
print(f"f1: {f1_score(pred_allfeats_f1model['y'], pred_allfeats_f1model['prediction_label'])}")
print(f"roc_auc: {roc_auc_score(pred_allfeats_aucmodel['y'], pred_allfeats_aucmodel['prediction_label'])}")
print(classification_report(pred_allfeats_accmodel['y'], pred_allfeats_accmodel['prediction_label']))

print("now for the f1 model: \n")
print(f"accuracy: {accuracy_score(pred_allfeats_f1model['y'], pred_allfeats_f1model['prediction_label'])}")
print(f"f1: {f1_score(pred_allfeats_f1model['y'], pred_allfeats_f1model['prediction_label'])}")
print(f"roc_auc: {roc_auc_score(pred_allfeats_aucmodel['y'], pred_allfeats_aucmodel['prediction_label'])}")
print(classification_report(pred_allfeats_f1model['y'], pred_allfeats_f1model['prediction_label']))

print("finally for the auc model: \n")
print(f"accuracy: {accuracy_score(pred_allfeats_aucmodel['y'], pred_allfeats_aucmodel['prediction_label'])}")
print(f"f1: {f1_score(pred_allfeats_aucmodel['y'], pred_allfeats_aucmodel['prediction_label'])}")
print(f"roc_auc: {roc_auc_score(pred_allfeats_aucmodel['y'], pred_allfeats_aucmodel['prediction_label'])}")
print(classification_report(pred_allfeats_aucmodel['y'], pred_allfeats_aucmodel['prediction_label']))

print("Evaluation metrics for the no call features models:")
print("starting with the accuracy model: \n")
print(f"accuracy: {accuracy_score(pred_nocall_accmodel['y'], pred_nocall_accmodel['prediction_label'])}")
print(f"f1: {f1_score(pred_nocall_accmodel['y'], pred_nocall_accmodel['prediction_label'])}")
print(f"roc_auc: {roc_auc_score(pred_nocall_aucmodel['y'], pred_nocall_aucmodel['prediction_label'])}")
print(classification_report(pred_nocall_accmodel['y'], pred_nocall_accmodel['prediction_label']))

print("now for the f1 model: \n")
print(f"accuracy: {accuracy_score(pred_nocall_f1model['y'], pred_nocall_f1model['prediction_label'])}")
print(f"f1: {f1_score(pred_nocall_f1model['y'], pred_nocall_f1model['prediction_label'])}")
print(f"roc_auc: {roc_auc_score(pred_nocall_aucmodel['y'], pred_nocall_aucmodel['prediction_label'])}")
print(classification_report(pred_nocall_f1model['y'], pred_nocall_f1model['prediction_label']))

print("finally for the auc model: \n")
print(f"accuracy: {accuracy_score(pred_nocall_aucmodel['y'], pred_nocall_aucmodel['prediction_label'])}")
print(f"f1: {f1_score(pred_nocall_aucmodel['y'], pred_nocall_aucmodel['prediction_label'])}")
print(f"roc_auc: {roc_auc_score(pred_nocall_aucmodel['y'], pred_nocall_aucmodel['prediction_label'])}")
print(classification_report(pred_nocall_aucmodel['y'], pred_nocall_aucmodel['prediction_label']))

print('Done!')
