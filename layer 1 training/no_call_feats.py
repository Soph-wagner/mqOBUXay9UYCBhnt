'''
Author: Sophia Wagner
Date: 7/13/2024
Description: creating a second dataset that does not contain any call realted features/data
'''

#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#import data
data = pd.read_csv('term-deposit-marketing-2020.csv')

#check that data loads
print(data.head(5))

# print columns
print(data.columns)

# drop columns that are call related
### those are cols 9, 10, 11, 12 
data_no_call = data.drop(columns=['contact', 'day', 'month', 'duration', 'campaign'])

print(data_no_call.head(5))
### new dataset has a size of 40000 rows x 9 columns

# Label Encoding 'y'
label_encoder = LabelEncoder()
data_no_call['y'] = label_encoder.fit_transform(data_no_call['y']) #label mapping: 'no' = 0, 'yes' = 1

# One Hot Encoding all other categorical columns
columns = ['job', 'marital', 'education', 'default', 'housing', 'loan']
encoder = OneHotEncoder(sparse_output=False) 
sk_one_hot_encoded = encoder.fit_transform(data_no_call[columns])
sk_one_hot_data = pd.DataFrame(sk_one_hot_encoded, columns=encoder.get_feature_names_out(columns))

#concat one hot dataframe with og dataframe
sk_encoded_data = pd.concat([data_no_call,sk_one_hot_data], axis=1)

# drop og categor cols
sk_one_hot_data = sk_encoded_data.drop(columns, axis=1)

# print final encoded dataframe
print(f"Sklearn Encoded term deposit data: \n{sk_one_hot_data}")

print(sk_one_hot_data.columns.tolist())
### dataset now has a size of 40000 rows x 28 columns

# save data_no_call to a new csv file
sk_one_hot_data.to_csv('no-call-one-hot-term-deposit-data.csv', index=False)