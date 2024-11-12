'''
Author: Sophia Wagner
Date: 8/22/2024
Description: using the model returned from PyCaret to predict the entire term deposit dataset
             model is gradient boosting classifier and training using a 5-fold cross validation
'''

# importing libraries
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler

# load in the full dataset
data = pd.read_csv('one-hot-term-deposit-data.csv')

# separate features and target variable
X = data.drop('y', axis=1)
y = data['y']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create RUS train data
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

# create the model
gbc = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='log_loss', max_depth=3,   
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_samples_leaf=1,     
                           min_samples_split=2, min_weight_fraction_leaf=0.0, 
                           n_estimators=100, n_iter_no_change=None,
                           random_state=7815, subsample=1.0, tol=0.0001,      
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)

# cross validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
precision = cross_val_score(gbc, X_train_rus, y_train_rus, scoring='precision', cv=kf)

print("Precision: ", precision)
print("Mean Precision: ", precision.mean())

# fit the model
gbc.fit(X_train_rus, y_train_rus)

# evaluate the model
y_pred = gbc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Precision: ", precision)
print("Confusion Matrix: \n", cm)
print("Classification Report: \n", cr)

# save the model
joblib.dump(gbc, 'final2_gbc_model.pkl')

print("done!")

