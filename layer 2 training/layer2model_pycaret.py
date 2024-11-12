'''
Author: Sophia Wagner
Date: 8/22/2024
Description: Using PyCaret and RandomUnderSampler on the full dataset
             to search for the best model with a good precision score
'''

# importing libraries
import pandas as pd
from pycaret.classification import setup, compare_models, automl, tune_model, predict_model
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score



# load in the full dataset
data = pd.read_csv('one-hot-term-deposit-data.csv')

# separate features and target variable
X = data.drop('y', axis=1)
y = data['y']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_data = pd.concat([X_test, y_test], axis=1)

# create RUS train data
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# turn X_train_rus and y_train_rus into one rus_train_data dataframe
rus_train_data = pd.concat([X_train_rus, y_train_rus], axis=1)

print("rus_train_data shape: ",rus_train_data.shape)
print("X_train shape: ",X_train.shape)
print("rus_train_data head: ",rus_train_data.head(5))

#print y counts of the original and RUS data
print("y train value counts: ",y_train.value_counts())
print("y train RUS value counts: ",y_train_rus.value_counts())
print("y test value counts: ",y_test.value_counts())


# set up PyCaret environment
clf1 = setup(data=rus_train_data, target='y')

# compare models
models = compare_models()

# pick winning model
best_model = automl(optimize='Precision')

# tune the best model
tuned_model = tune_model(best_model)

print("best model: ", best_model)
print("tuned best model: ", tuned_model)


####################
# making predictions and evaluating the model

preds = predict_model(tuned_model, data=test_data)

print(preds.head())

print("predictions on test set using tuned model...")

#precision score
precision = precision_score(preds['y'], preds['prediction_label'])
print("Precision score: ", precision)

# confusion matrix and classification report
cm = confusion_matrix(preds['y'], preds['prediction_label'])
cr = classification_report(preds['y'], preds['prediction_label'])

print(cm)
print(cr)



