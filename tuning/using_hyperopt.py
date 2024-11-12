'''
Author: Sophia Wagner
Date: 8/30/2024
Description: Using Hyeropt to optimize hyperparameters for the Decision Tree Classifier for the full dataset
'''

# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, precision_score, confusion_matrix, classification_report
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from imblearn.combine import SMOTEENN

# load in data
full_data = pd.read_csv('one-hot-term-deposit-data.csv')

# split train and test
X_full = full_data.drop('y', axis=1)
y_full = full_data['y']

X_tr_full, X_te_full, y_tr_full, y_te_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# SMOTE-ENN on full train set
smote_enn = SMOTEENN(random_state=42)
X_train_smtenn, y_train_smtenn = smote_enn.fit_resample(X_tr_full, y_tr_full)

# define the objective function
# define the objective function
def dt_objective(search_params):
    dt_clf = DecisionTreeClassifier(**search_params)
    recall = cross_val_score(dt_clf, X_train_smtenn, y_train_smtenn, scoring=make_scorer(precision_score), cv=5).mean()
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
best = fmin(fn=dt_objective,
               space=dt_space,
               algo=tpe.suggest,
               max_evals=200,
               trials=trials)

print('Best Decision Tree Hyperparameters: ', best)
'''

#####################################################################
#####################################################################

# creating the Decision Tree Classifier model

clf = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=26,
                                max_features=None, min_samples_leaf=6, min_samples_split=22,
                                splitter='best')

scorer = make_scorer(precision_score)
cv_recall_scores = cross_val_score(clf, X_train_smtenn, y_train_smtenn, scoring=scorer, cv=5)
print('Decision Tree Cross-Val Recall Scores: ', cv_recall_scores)
print('Decision Tree Cross-Val Recall Mean: ', cv_recall_scores.mean())

clf.fit(X_train_smtenn, y_train_smtenn)
y_pred_clf = clf.predict(X_te_full)
test_prec_score = precision_score(y_te_full, y_pred_clf)
print('Decision Tree Classifier test set Recall Score: ', test_prec_score)
print('Decision Tree Classifier confusion matrix: ', confusion_matrix(y_te_full, y_pred_clf))
print('Decision Tree Classifier classification report: ', classification_report(y_te_full, y_pred_clf))

# saving an adequate model
if test_prec_score > 0.75:
    # save the model
    import joblib
    joblib.dump(clf, 'layer2_dtc_model.pkl')

    print('Full dataset model saved') 
else:
    print('precision not high enough')



