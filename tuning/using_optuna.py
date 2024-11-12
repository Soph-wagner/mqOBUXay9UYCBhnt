'''
Author: Sophia Wagner
Date: 8/12/2024
Description: Using Optuna to find the best hyperparameters for the No-Call data model
             Focus on Recall for Class 1

Date: 8/26/2024
Decription: using Optuna to find the best hyperparameters for a gradient boosting classifier
            or a Light GBM model; goal is to maximize Precision for class 1
'''

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import optuna
# models to be used 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 

import seaborn as sns


#load in data, using the one-hot encoded data
data = pd.read_csv('no-call-one-hot-term-deposit-data.csv')
full_data = pd.read_csv('one-hot-term-deposit-data.csv')

# quick look at the data
## print y counts
#print(data['y'].value_counts()) #no: 37104, yes: 2896

# separate features and target variable
X = data.drop('y', axis=1)
y = data['y']

X_full = full_data.drop('y', axis=1)
y_full = full_data['y']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_full_tr, X_full_te, y_full_tr, y_full_te = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# this is the place to impliment data resampling if needed
# using Random Under Sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, replacement=False)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
X_full_tr_rus, y_full_tr_rus = rus.fit_resample(X_full_tr, y_full_tr)

from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train_smtenn, y_train_smtenn = smote_enn.fit_resample(X_full_tr, y_full_tr)



# define the objective function
def objective(trial):

    print('running objective for layer 1 model')
    # define the parameters to search
    gbc_params = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 200),
        # 'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'loss': trial.suggest_categorical('loss', ['log_loss','exponential']),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        #'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.5),
        #'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 30),
        #'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        #'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        # n_iter_no_change=None,  
        #'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3)
        #'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.5)
   
    }

    '''
    dtc_params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'max_depth': trial.suggest_int('max_depth', 1, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    }
    '''

    # create the model
    model_gbc = GradientBoostingClassifier(**gbc_params)
    #model_dtc = DecisionTreeClassifier(**dtc_params)
    
    # fit the model
    model_gbc.fit(X_train_rus, y_train_rus)
    #model_dtc.fit(X_train, y_train)
    
    # predict the test set
    y_pred_gbc = model_gbc.predict(X_test)
    #y_pred_dtc = model_dtc.predict(X_test)
    
    # calculate the recall for class 1
    recall_gbc = recall_score(y_test, y_pred_gbc, pos_label=1)
    #recall_dtc = recall_score(y_test, y_pred_dtc, pos_label=1)
    
    '''
    # printing the model performance
    print("LAYER 1 NO CALL DATA MODEL")
    print(f'Gradient Boosting Classifier Recall: {recall_gbc}')
    #print(f'Decision Tree Classifier Recall: {recall_dtc}')

    # print the confusion matrix and classification report
    print('Gradient Boosting Classifier Confusion Matrix: \n', confusion_matrix(y_test, y_pred_gbc))
    print('Gradient Boosting Classifier Classification Report: \n', classification_report(y_test, y_pred_gbc))
    # print('Decision Tree Classifier Confusion Matrix: \n', confusion_matrix(y_test, y_pred_dtc))
    # print('Decision Tree Classifier Classification Report: \n', classification_report(y_test, y_pred_dtc))
    '''
    
    # return the recall for class 1
    return recall_gbc #, recall_dtc

def lay1_rfc_object(trial): 
    
    print('running RFC objective for layer 1 No-Call model') 

    rfc_params = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 200),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    }

    model_rfc = RandomForestClassifier(**rfc_params)

    model_rfc.fit(X_train_rus, y_train_rus)

    y_pred_rfc = model_rfc.predict(X_test)

    recall_rfc = recall_score(y_test, y_pred_rfc, pos_label=1)

    # printing the model performance
    print(f'Random Forest Classifier Precision: {recall_rfc}')

    # print the confusion matrix and classification report
    print('Random Forest Classifier Confusion Matrix: \n', confusion_matrix(y_test, y_pred_rfc))
    print('Random Forest Classifier Classification Report: \n', classification_report(y_test, y_pred_rfc))

    return recall_rfc

'''
#creating a pruner
pruner = optuna.pruners.MedianPruner() 
#prunes trials whose intermediate values are worse than the median of reported values for the same step

# create the study object
study = optuna.create_study(direction='maximize', pruner=pruner)

#optimize the objective function
study.optimize(lay1_rfc_object, n_trials=100)

print("LAYER 1 NO CALL DATA MODEL")
print("study best parameters: ", study.best_params)
print("study best value: ", study.best_value)
#print("study best trial: ", study.best_trial)
print("best trial parameters: ", study.best_trial.params)

'''

#########################################################################
# Redoing the optuna training process for the FUll Term Deposit Dataset 
# 
#
print('Full Term Deposit Dataset')

# defining objective function for GBC
def gbc_objective(trial):

    print('running gbc objective for layer 2 model')
    # define search parameters
    gbc_params = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 200),
        #'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'loss': trial.suggest_categorical('loss', ['log_loss','exponential']),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
        'max_depth': trial.suggest_int('max_depth', 1, 28),
        #'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.5),
        #'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 30),
        #'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        #'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5),
        # n_iter_no_change=None,  
        #'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'tol': trial.suggest_float('tol', 1e-5, 1e-3),
        #'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.5)
    }

    model_gbc = GradientBoostingClassifier(**gbc_params)
    
    model_gbc.fit(X_full_tr_rus, y_full_tr_rus)
    
    y_pred_gbc_full = model_gbc.predict(X_full_te)
    
    precs_gbc_full = precision_score(y_full_te, y_pred_gbc_full, pos_label=1)

    # printing the model performance
    print(f'Gradient Boosting Classifier Precision: {precs_gbc_full}')

    # print the confusion matrix and classification report
    print('Gradient Boosting Classifier Confusion Matrix: \n', confusion_matrix(y_full_te, y_pred_gbc_full))
    print('Gradient Boosting Classifier Classification Report: \n', classification_report(y_full_te, y_pred_gbc_full))
   
    return precs_gbc_full

def rfc_objective(trial):

    print('running rfc objective for layer 2 model') 

    rfc_params = {
        'n_estimators': trial.suggest_int('n_estimators', 2, 200),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    }

    model_rfc = RandomForestClassifier(**rfc_params)

    model_rfc.fit(X_train_smtenn, y_train_smtenn)

    y_pred_rfc_full = model_rfc.predict(X_full_te)

    precs_rfc_full = precision_score(y_full_te, y_pred_rfc_full, pos_label=1)

    # printing the model performance
    print(f'Random Forest Classifier Precision: {precs_rfc_full}')

    # print the confusion matrix and classification report
    print('Random Forest Classifier Confusion Matrix: \n', confusion_matrix(y_full_te, y_pred_rfc_full))
    print('Random Forest Classifier Classification Report: \n', classification_report(y_full_te, y_pred_rfc_full))

    return precs_rfc_full

def logreg_objective(trial):
    
    print('running LogReg objective for layer 2 model') 
    
    #solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    #penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    c = trial.suggest_float('C', 1e-6, 1e2, log=True)
    tol = trial.suggest_float('tol', 1e-5, 1e-3)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    max_iter = trial.suggest_int('max_iter', 50, 300) 

    model_logr = LogisticRegression(C=c, tol=tol, class_weight=class_weight, fit_intercept=fit_intercept, max_iter=max_iter)

    model_logr.fit(X_train_smtenn, y_train_smtenn)

    y_pred_logr_full = model_logr.predict(X_full_te)

    precs_logr_full = precision_score(y_full_te, y_pred_logr_full, pos_label=1)

    # printing the model performance
    print(f'Logistic Regression Precision: {precs_logr_full}')

    # print the confusion matrix and classification report
    print('Logistic Regression Confusion Matrix: \n', confusion_matrix(y_full_te, y_pred_logr_full))
    print('Logistic Regression Classification Report: \n', classification_report(y_full_te, y_pred_logr_full))

    return precs_logr_full

def dtc_objective(trial): 

    print('running DTC objective for layer 2 model')

    dtc_params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'max_depth': trial.suggest_int('max_depth', 1, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'max_leaf_nodes' : trial.suggest_int('max_leaf_nodes', 2, 100)
    }

    model_dtc = DecisionTreeClassifier(**dtc_params)

    model_dtc.fit(X_train_smtenn, y_train_smtenn)

    y_pred_dtc_full = model_dtc.predict(X_full_te)

    precs_dtc_full = precision_score(y_full_te, y_pred_dtc_full, pos_label=1)

    # printing the model performance
    print(f'Decision Tree Classifier Precision: {precs_dtc_full}')

    print('Decision Tree Classifier Confusion Matrix: \n', confusion_matrix(y_full_te, y_pred_dtc_full))
    print('Decision Tree Classifier Classification Report: \n', classification_report(y_full_te, y_pred_dtc_full))

#'''
#creating a pruner
pruner = optuna.pruners.MedianPruner()

# create the study object, want to maximize precision for class 1
study = optuna.create_study(direction='maximize', pruner=pruner)

#optimize the objective function
study.optimize(dtc_objective, n_trials=150)

print("Layer 2 Full Term Deposit Dataset Model")
print("study best parameters: ", study.best_params)
print("study best value: ", study.best_value)
#print("study best trial: ", study.best_trial)
print("best trial parameters: ", study.best_trial.params)
#'''


