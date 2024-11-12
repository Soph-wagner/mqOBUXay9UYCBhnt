'''
Author: Sophia Wagner
Date: 8/14/2024
Description: Creating and tuning the final models for the term deposit dataset; using one Decision Tree Classifier and one Gradient Boosting Classifier
             Random Under Sampling used for the Decision Tree Classifier dataset and no sampling used for the Gradient Boosting Classifier dataset
             The Gradient Boosting Classifier will test on an input dataset that it determined by the results of the Decision Tree Classifier
'''

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, recall_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

#hyperparameter tuning
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

#load in data, using the one-hot encoded data
no_call_data = pd.read_csv('../data/no-call-one-hot-term-deposit-data.csv')

#making sure data is loaded in, checking y counts
#print('no call data y counts: ', no_call_data['y'].value_counts()) #no: 37104, yes: 2896        

# separate features and target variable
X_no_call = no_call_data.drop('y', axis=1)
y_no_call = no_call_data['y']

# import random int and set seed
import random 
# seed = random.randint(1000,9999)
seed = 9059
print('seed: ', seed)

# split the data into training and testing sets
#want to make sure both datasets are split the same
X_train_no_call, X_test_no_call, y_train_no_call, y_test_no_call = train_test_split(X_no_call, y_no_call, test_size=0.2, random_state=seed)

# sampling the no call training data
rus = RandomUnderSampler(random_state=seed, replacement=False)
X_tr_rus, y_tr_rus = rus.fit_resample(X_train_no_call, y_train_no_call)
## check new y value counts
#print('RUS resampled y value counts: ', pd.Series(y_tr_rus).value_counts())


################################################
# implimenting hyperopt for tuning the decision tree classifier

# define the objective function
def dt_objective(search_params):
    dt_clf = DecisionTreeClassifier(**search_params)
    recall = cross_val_score(dt_clf, X_tr_rus, y_tr_rus, scoring=make_scorer(recall_score), cv=5).mean()
    return -recall  ##minimize the negative of the recall

# decision tree search space
dt_space = {
    'class_weight': hp.choice('class_weight', ['balanced', None]),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'splitter': hp.choice('splitter', ['best', 'random']),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 32, 1)),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 32, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 32, 1))
}

# running hyperopt search for the Decision Tree Classifier
'''
trials = Trials()
best = fmin(fn=gb_objective,
               #space=dt_space,
               space=gb_space,
               algo=tpe.suggest,
               max_evals=100,
               trials=trials)

#print('Best Decision Tree Hyperparameters: ', best)
'''

#####################################################################
#####################################################################

# creating the Decision Tree Classifier model

# dtc = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', 
#                              splitter='random', max_depth=1, max_features='sqrt', 
#                              min_samples_leaf=19, min_samples_split=15, random_state=seed)

# gbc = GradientBoostingClassifier(n_estimators=82, loss='log_loss', learning_rate=0.15720724871814273, 
#                                  criterion='friedman_mse', max_depth=15, min_samples_leaf=2, 
#                                  min_samples_split=22, tol=0.0008952425309703175, random_state=seed)

rfc = RandomForestClassifier(n_estimators=177, criterion='entropy', max_depth=29,
                             min_samples_split=6, min_samples_leaf=1, max_features=None, random_state=seed)


scorer = make_scorer(recall_score)
cv_recall_scores = cross_val_score(rfc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)
print('Decision Tree Cross-Val Recall Scores: ', cv_recall_scores)
print('Decision Tree Cross-Val Recall Mean: ', cv_recall_scores.mean())

#clf.fit(X_tr_rus, y_tr_rus)
#gbc.fit(X_tr_rus, y_tr_rus)
rfc.fit(X_tr_rus, y_tr_rus)
#y_pred_dt = clf.predict(X_test_no_call)
#y_pred_gbc = gbc.predict(X_test_no_call)
y_pred_rfc = rfc.predict(X_test_no_call)
test_rec_score = recall_score(y_test_no_call, y_pred_rfc)
print(type(rfc))
print('Decision Tree Classifier test set Recall Score: ', test_rec_score)
print('Decision Tree Classifier confusion matrix: ', confusion_matrix(y_test_no_call, y_pred_rfc))
print('Decision Tree Classifier classification report: ', classification_report(y_test_no_call, y_pred_rfc))

print('Seed: ', seed)


# saving the final model
import joblib
joblib.dump(rfc, 'layer1_final_model.pkl')

print('No Call model saved') 
