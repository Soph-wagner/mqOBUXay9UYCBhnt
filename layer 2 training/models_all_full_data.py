'''
Author: Sophia Wagner
Date: 8/6/2024
Description: Creating and comparing 3 different model architectures for the full data term deposit dataset
             models used: Nearest Centroid, Decision Tree Classifier, Gradient Boosting Classifier
             first undersampling the majority class, then using a 5-fold cross validation
'''


# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
# resampling library
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN 
from imblearn.combine import SMOTEENN
# models to be used 
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#load in data, using the one-hot encoded data
data = pd.read_csv('one-hot-term-deposit-data.csv')

# quick look at the data
#print(data.head(5))     #size: 40000 x 28 (including target var)

# print y value counts
#print(data['y'].value_counts()) #no: 37104, yes: 2896

# separate features and target variable
X = data.drop('y', axis=1)
y = data['y']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Creating Random Undersampled training data: ")
# undersampling the majority class
rus = RandomUnderSampler(random_state=42, replacement=False)
X_tr_rus, y_tr_rus = rus.fit_resample(X_train, y_train)
## check new y value counts
print('resampled y value counts: ', pd.Series(y_tr_rus).value_counts())


print("Creating Random Oversampled training data: ")
# oversampling the minority class
ros = RandomOverSampler(random_state=42)
X_tr_ros, y_tr_ros = ros.fit_resample(X_train, y_train)
## check new y value counts
print('resampled y value counts: ', pd.Series(y_tr_ros).value_counts())

# create models 
nc_clf = NearestCentroid()
dt_clf = DecisionTreeClassifier(class_weight='balanced')
gb_clf = GradientBoostingClassifier()

models = []
models.append(('Nearest Centroid', nc_clf))
models.append(('Decision Tree Classifier', dt_clf))
models.append(('Gradient Boosting Classifier', gb_clf))

print("Training/Evaluating models using Random Undersampling: ")
# evaluating using 5-fold cross validation and the resampled training data
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    cv_scores = cross_val_score(model, X_tr_rus, y_tr_rus, cv=kfold, scoring='balanced_accuracy')
    results.append(cv_scores)
    names.append(name)
    print(f'{name}: {cv_scores.mean()} (std: {cv_scores.std()})')

# printing the confusion matrix and classification report for each model
# once again using a 5-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    #cross-val predictions for the training data
    y_pred_tr = cross_val_predict(model, X_tr_rus, y_tr_rus, cv=kfold)
    #train model on rus training data
    model.fit(X_tr_rus, y_tr_rus)
    #predict on the og test data
    y_pred_test = model.predict(X_test)
    #create confusion matrix using the test data preds
    cm = confusion_matrix(y_test, y_pred_test)
    print(f'{name} confusion matrix on test data: \n{confusion_matrix(y_test, y_pred_test)}')
    print(f'{name} classification report on test data: \n{classification_report(y_test, y_pred_test)}')
    print(f'Interpretation of {name} confusion matrix: \n True Positives (TP): {cm[1, 1]} \n True Negatives (TN): {cm[0, 0]} \n False Positives (FP): {cm[0, 1]} \n False Negatives (FN): {cm[1, 0]}')

##########################################
print("Training/Evaluating models using Random Oversampling: ")
# evaluate each model using 5-fold cross validation and the resampled training data
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_tr_ros, y_tr_ros, cv=kfold, scoring='balanced_accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# printing the confusion matrix and classification report for each model
# once again using a 5-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    #cross-val predictions for the training data
    y_pred_tr = cross_val_predict(model, X_tr_ros, y_tr_ros, cv=kfold)
    #train model on rus training data
    model.fit(X_tr_ros, y_tr_ros)
    #predict on the og test data
    y_pred_test = model.predict(X_test)
    #create confusion matrix using the test data preds
    cm = confusion_matrix(y_test, y_pred_test)
    print(f'{name} confusion matrix on test data: \n{confusion_matrix(y_test, y_pred_test)}')
    print(f'{name} classification report on test data: \n{classification_report(y_test, y_pred_test)}')
    print(f'Interpretation of {name} confusion matrix: \n True Positives (TP): {cm[1, 1]} \n True Negatives (TN): {cm[0, 0]} \n False Positives (FP): {cm[0, 1]} \n False Negatives (FN): {cm[1, 0]}')
  
