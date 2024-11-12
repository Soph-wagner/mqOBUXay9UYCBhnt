'''
Author: Sophia Wagner
Date: 8/19/2024
Description: Hyperparameter optimizing and comparing 4 final sampling techniques on Gradient Boosting Classifier and 
             Decision Tree Classifier using the full term deposit dataset and striving for highest precision score
'''

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, precision_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

#hyperparameter tuning
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

# load in data, using the one-hot encoded data
full_data = pd.read_csv('one-hot-term-deposit-data.csv')

# make sure data is loaded in, checking y counts
print('full data y counts: ', full_data['y'].value_counts()) #no: 36548, yes: 445

# separate features and target variable
X_full = full_data.drop('y', axis=1)
y_full = full_data['y']

# split the data into training and testing sets
#want to make sure this dataset and the no_call dataset are split the same
import random 
seed = random.randint(1000,9999)
# seed = 1234 <- eventually change to set seed to train all models
print('seed: ', seed)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)


# for the full training data
## need to evaluate between no sampling, SMOTE, SMOTE-Tomek, and ADASYN
## use X/y_train_full for the No Sampling technique

# SMOTE sampling
smoteenn = SMOTEENN(random_state=seed)
X_train_smtenn, y_train_smtenn = smoteenn.fit_resample(X_train_full, y_train_full)

#random undersampling
rus = RandomUnderSampler(random_state=seed, replacement=False)
X_train_rus, y_train_rus = rus.fit_resample(X_train_full, y_train_full)

'''
# check the new y value counts for each sampling technique
for sampl_name, (X_train, y_train) in sampling_techniques.items():
    print(f'{sampl_name} resampled y value counts: ', pd.Series(y_train).value_counts())
'''

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

#hyperopt search for gradient boosting classifier used each sampling technique
'''
best_params = []
sampl_name = []
best_params_dict = {}
for name, (X_train, y_train) in sampling_techniques.items():
    #best_params.append(best)
    #    
    print(f'Hyperopt search for Gradient Boosting Classifier using {name} sampling: ')
    trials = Trials()

    best = fmin(fn=lambda params: gb_objective(params, X_train, y_train),
                space=gb_space,
                algo=tpe.suggest,
                max_evals=15,
                trials=trials)
    sampl_name.append(name)
    best_params.append(best)
    print(f'Best Gradient Boosting Hyperparameters for {name}: ', best)

# print the best hyperparameters for each sampling technique
for i in range(len(sampl_name)):
    best_params_dict[sampl_name[i]] = best_params[i]
print('printing the best parameter dictionary: ', best_params_dict)

print('Best Gradient Boosting Hyperparameters for each sampling technique: ', best_params)
##### sampling order is Smote, SMOTE-Tomek, ADASYN

'''

##############################################################
# creating the Gradient Boosting Classifier model
#'''

print('Evaluating Tuned Gradient Boosting Classifier model: ')
print('Seed: ', seed)

gbc1 = GradientBoostingClassifier(n_estimators=139, loss='exponential', learning_rate=0.25785434875111424, 
                                 criterion='friedman_mse', max_depth=4, min_samples_leaf=30,
                                min_samples_split=19, tol=0.0008591665140319732) 

gbc2 = GradientBoostingClassifier(n_estimators=190, loss='exponential', learning_rate=0.2365000529387143,
                                  criterion='friedman_mse', max_depth=4, min_samples_leaf=28, 
                                  min_samples_split=14, tol=0.0009435648331745362, random_state=seed)

scorer = make_scorer(precision_score, pos_label=1)
fulldata_prec_scores = cross_val_score(gbc2, X_train_full, y_train_full, scoring=scorer, cv=5)
# rus_prec_scores = cross_val_score(gbc2, X_train_rus, y_train_rus, scoring=scorer, cv=5)
# smtenn_prec_scores = cross_val_score(gbc2, X_train_smtenn, y_train_smtenn, scoring=scorer, cv=5)

print('Gradient Boosting Cross-Val Precision Scores: ')
print(f'no sampling prec scores: {fulldata_prec_scores}') # RUS prec scores: {rus_prec_scores} SMOTEENN prec scores: {smtenn_prec_scores}')
print('Gradient Boosting Cross-Val Precision Mean: ') 
print(f'no sampling prec mean: {fulldata_prec_scores.mean()} ') # RUS prec mean: {rus_prec_scores.mean()} SMOTEENN prec mean: {smtenn_prec_scores.mean()}')

gbc2.fit(X_train_full, y_train_full)
y_pred_gb = gbc2.predict(X_test_full)
test_prec_score = precision_score(y_test_full, y_pred_gb, pos_label=1)
print('GBC trained using non-sampled data')
print('Gradient Boosting Classifier test set Precision Score: ', test_prec_score)
print('Gradient Boosting Classifier confusion matrix: ', confusion_matrix(y_test_full, y_pred_gb))
print('Gradient Boosting Classifier classification report: ', classification_report(y_test_full, y_pred_gb))

# gbc2.fit(X_train_rus, y_train_rus)
# y_pred_gb = gbc2.predict(X_test_full)
# test_prec_score = precision_score(y_test_full, y_pred_gb, pos_label=1)
# print('GBC trained using RUS data')
# print('Gradient Boosting Classifier test set Precision Score: ', test_prec_score)
# print('Gradient Boosting Classifier confusion matrix: ', confusion_matrix(y_test_full, y_pred_gb))
# print('Gradient Boosting Classifier classification report: ', classification_report(y_test_full, y_pred_gb))

# gbc2.fit(X_train_smtenn, y_train_smtenn)
# y_pred_gb = gbc2.predict(X_test_full)
# test_prec_score = precision_score(y_test_full, y_pred_gb, pos_label=1)
# print('GBC trained using SMOTEENN data')
# print('Gradient Boosting Classifier test set Precision Score: ', test_prec_score)
# print('Gradient Boosting Classifier confusion matrix: ', confusion_matrix(y_test_full, y_pred_gb))
# print('Gradient Boosting Classifier classification report: ', classification_report(y_test_full, y_pred_gb))

