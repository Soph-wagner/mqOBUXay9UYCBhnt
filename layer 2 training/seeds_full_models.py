'''
Author: Sophia Wagner
Date: 9/10/2024
Description: Hyperparameter tuning and optimizing 3 different models for the full term deposit dataset
             ensuring proper use of the random seed so that models are reproducible
'''

# import libraries
import random
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, precision_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

from pycaret.classification import setup, compare_models, automl, tune_model, predict_model

#hyperparameter tuning
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

# load in data, using the one-hot encoded data
full_data = pd.read_csv('data/one-hot-term-deposit-data.csv')

# separate features and target variable
X = full_data.drop('y', axis=1)
y = full_data['y']

# set random seed and print it
seed = random.randint(1000,9999)
print('seed: ', seed)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

test_data = pd.concat([X_test, y_test], axis=1)

# create RUS train data
rus = RandomUnderSampler(random_state=seed)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

rus_train_data = pd.concat([X_train_rus, y_train_rus], axis=1)

# create SMOTEENN train data
smtenn = SMOTEENN(random_state=seed)
X_train_smtenn, y_train_smtenn = smtenn.fit_resample(X_train, y_train)

smtenn_train_data = pd.concat([X_train_smtenn, y_train_smtenn], axis=1)

'''
# print shapes of datasets and the y counts
print("rus_train_data shape: ",rus_train_data.shape)
print(" smtenn_train_data shape: ",smtenn_train_data.shape)
print("X_train shape: ",X_train.shape)

print("y train value counts: ",y_train.value_counts())
print("y train RUS value counts: ",y_train_rus.value_counts())
print("y train SMOTEENN value counts: ",y_train_smtenn.value_counts())
'''


#pycaret environment setup and model selection
'''
##########################################################################
# set up PyCaret environment
clf1 = setup(data=rus_train_data, target='y')

# compare models
models = compare_models()

# pick winning model
best_model = automl(optimize='Accuracy')

# tune the best model
tuned_model = tune_model(best_model)

print("best model: ", best_model)
print("tuned best model: ", tuned_model)

# making predictions and evaluating the model
preds = predict_model(tuned_model, data=test_data)

print(preds.head())

print("predictions on test set using tuned model...")

#precision score
precision = precision_score(preds['y'], preds['prediction_label'])
print("Precision score: ", precision) 

#accuracy score
accuracy = accuracy_score(preds['y'], preds['prediction_label'])
print("Accuracy score: ", accuracy)

# confusion matrix and classification report
print(confusion_matrix(preds['y'], preds['prediction_label']))
print(classification_report(preds['y'], preds['prediction_label']))

# print seed    
print('seed: ', seed)

# set up smoteenn clf for PyCaret environment
clf2 = setup(data=smtenn_train_data, target='y')

# compare models
models2 = compare_models()

# pick winning model
best_model2 = automl(optimize='Accuracy')

# tune the best model
tuned_model2 = tune_model(best_model2)

print("best model: ", best_model2)
print("tuned best model: ", tuned_model2)

# making predictions and evaluating the model
preds2 = predict_model(tuned_model2, data=test_data)

print("predictions on test set using tuned model...")

#precision score
precision2 = precision_score(preds2['y'], preds2['prediction_label'])
print("Precision score: ", precision2)

#accuracy score
accuracy2 = accuracy_score(preds2['y'], preds2['prediction_label'])
print("Accuracy score: ", accuracy2)

# confusion matrix and classification report
print(confusion_matrix(preds2['y'], preds2['prediction_label']))
print(classification_report(preds2['y'], preds2['prediction_label']))
#######################################################################################
'''

# print seed
print('seed: ', seed)


################################################
# implimenting hyperopt for tuning the Gradient Boosting Classifier

# Objective Function for the Gradient Boosting Classifier 
# searching for hyperparameters

def gb_objective(params, X_train, y_train):
    clf = GradientBoostingClassifier(**params)
    precision = cross_val_score(clf, X_train, y_train, scoring=make_scorer(precision_score), cv=5).mean()
    return -precision #minimize the negative of the precision

# search space for the Gradient Boosting Classifier
gb_space = { 
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 250, 10)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.4),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 15, 1)),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'criterion': hp.choice('criterion', ['friedman_mse', 'squared_error'])
}

# objective function for random forest classifier
def rf_objective(params, X_train, y_train):
    clf = RandomForestClassifier(**params)
    precision = cross_val_score(clf, X_train, y_train, scoring=make_scorer(precision_score), cv=5).mean()
    return -precision

# search space for the Random Forest Classifier
rf_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 400, 10)),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 60, 1)),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
    'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None])
}

# sampling techniques
sampling_techniques = {
    'RUS': (X_train_rus, y_train_rus),
    'SMOTEENN': (X_train_smtenn, y_train_smtenn)
}

#hyperopt search for each classifier using each sampling technique
'''
best_params = []
sampl_name = []
best_params_dict = {}
for name, (X_train, y_train) in sampling_techniques.items():
     
    #print(f'Hyperopt search for Gradient Boosting Classifier using {name} sampling: ')
    print(f'Hyperopt search for Random Forest Classifier using {name} sampling: ')
    trials = Trials()

    # best = fmin(fn=lambda params: gb_objective(params, X_train, y_train),
    #             space=gb_space,
    #             algo=tpe.suggest,
    #             max_evals=15,
    #             trials=trials)
    
    best = fmin(fn=lambda params: rf_objective(params, X_train, y_train),
               space=rf_space,
               algo=tpe.suggest,
               max_evals=25,
               trials=trials)
    sampl_name.append(name)
    best_params.append(best)
    #print(f'Best Gradient Boosting Hyperparameters for {name}: ', best)
    print(f'Best Random Forest Hyperparameters for {name}: ', best)

# print the best hyperparameters for each sampling technique
for i in range(len(sampl_name)):
    best_params_dict[sampl_name[i]] = best_params[i]
print('printing the best parameter dictionary: ', best_params_dict)

#print('Best Gradient Boosting Hyperparameters for each sampling technique: ', best_params)
print('Best Random Forest Hyperparameters for each sampling technique: ', best_params)
##### sampling order is RUS, SMOTEENN
'''

#####################################################################
#####################################################################

# training and evaluating the models using the best hyperparameters
'''
# creating the Gradient Boosting Classifier model

gbc_rus = GradientBoostingClassifier(criterion='squared_error', learning_rate=0.28999124516900915,
                                     max_depth=2, max_features=None, n_estimators=130)

gbc_smoteenn = GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.31413822854233825,
                                          max_depth=12, max_features='log2', n_estimators=240)

scorer = make_scorer(precision_score)
cv_prec_scores = cross_val_score(gbc_rus, X_train_rus, y_train_rus, scoring=scorer, cv=5)
print('Gradient Boosting Cross-Val Precision Scores RUS: ', cv_prec_scores)
print('Gradient Boosting Cross-Val Precision Mean RUS: ', cv_prec_scores.mean())

cv_prec_scores2 = cross_val_score(gbc_smoteenn, X_train_smtenn, y_train_smtenn, scoring=scorer, cv=5)
print('Gradient Boosting Cross-Val Precision Scores SMOTEENN: ', cv_prec_scores2)
print('Gradient Boosting Cross-Val Precision Mean SMOTEENN: ', cv_prec_scores2.mean())

# fit the models
gbc_rus.fit(X_train_rus, y_train_rus)
gbc_smoteenn.fit(X_train_smtenn, y_train_smtenn)

# make predictions
y_pred_rus = gbc_rus.predict(X_test)
y_pred_smoteenn = gbc_smoteenn.predict(X_test)

# print the confusion matrix and classification report
print('GBC RUS Precision Score: ', precision_score(y_test, y_pred_rus))
print('Gradient Boosting Classifier RUS confusion matrix: ', confusion_matrix(y_test, y_pred_rus))
print('Gradient Boosting Classifier RUS classification report: ', classification_report(y_test, y_pred_rus))

print('GBC SMOTEENN Precision Score: ', precision_score(y_test, y_pred_smoteenn))
print('Gradient Boosting Classifier SMOTEENN confusion matrix: ', confusion_matrix(y_test, y_pred_smoteenn))
print('Gradient Boosting Classifier SMOTEENN classification report: ', classification_report(y_test, y_pred_smoteenn))
'''

# creating the Random Forest Classifier model

rf_rus = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=9, 
                                max_features=None, min_samples_leaf=4, min_samples_split=20, 
                                n_estimators=220)

rf_smtenn = RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=948,
                                      max_features='sqrt', min_samples_leaf=4, min_samples_split=8,
                                      n_estimators=390)

scorer = make_scorer(precision_score)
cv_prec_scores = cross_val_score(rf_rus, X_train_rus, y_train_rus, scoring=scorer, cv=5)
print('Random Forest Cross-Val Precision Scores RUS: ', cv_prec_scores)
print('Random Forest Cross-Val Precision Mean RUS: ', cv_prec_scores.mean())

cv_prec_scores2 = cross_val_score(rf_smtenn, X_train_smtenn, y_train_smtenn, scoring=scorer, cv=5)
print('Random Forest Cross-Val Precision Scores SMOTEENN: ', cv_prec_scores2)
print('Random Forest Cross-Val Precision Mean SMOTEENN: ', cv_prec_scores2.mean())

# fit the models
rf_rus.fit(X_train_rus, y_train_rus)
rf_smtenn.fit(X_train_smtenn, y_train_smtenn)

# make predictions
y_pred_rus = rf_rus.predict(X_test)
y_pred_smoteenn = rf_smtenn.predict(X_test)

# print the confusion matrix and classification report
print('RF RUS Precision Score: ', precision_score(y_test, y_pred_rus))
print('Random Forest Classifier RUS confusion matrix: ', confusion_matrix(y_test, y_pred_rus))
print('Random Forest Classifier RUS classification report: ', classification_report(y_test, y_pred_rus))

print('RF SMOTEENN Precision Score: ', precision_score(y_test, y_pred_smoteenn))
print('Random Forest Classifier SMOTEENN confusion matrix: ', confusion_matrix(y_test, y_pred_smoteenn))
print('Random Forest Classifier SMOTEENN classification report: ', classification_report(y_test, y_pred_smoteenn))





