'''
Author: Sophia Wagner
Date: 7/13/2024
Description: investigating correlation among features
'''

# import libraries
import pandas as pd
import numpy as np
import matplotlib
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import OneHotEncoder

# load data
data = pd.read_csv('term-deposit-marketing-2020.csv')

#check that data loads
print(data.head(5))

#split data into features and target var
X = data.iloc[:, 0:12].values
y = data.iloc[:, -1].values

print(X)
print(y)

'''
Looking into Label Encoding 
 - it might be the best idea to use a combination of Label Encoding and One Hot Feature Encoding
 - this is becaue Label Encoding would help preserve 'relationships' between features and maintain dimensionality

 - Columns to Label Encode: 'y', 'month', 
'''
# label encoding 'month' and 'y' columns
from sklearn.preprocessing import LabelEncoder

# init label encoder
label_encoder = LabelEncoder()

# label encode 'month' and 'y' 
data['y'] = label_encoder.fit_transform(data['y']) #label mapping: 'no' = 0, 'yes' = 1
data['month'] = label_encoder.fit_transform(data['month']) 
# month label mapping: ['apr' 'aug' 'dec' 'feb' 'jan' 'jul' 'jun' 'mar' 'may' 'nov' 'oct']
#                      [ 0     1      2     3     4     5     6     7     8     9     10 ]

print(f"Label Encoded term deposit data: \n{data}")

#print the mapping of the label encoding?
print(f"Mapping of Label Encoding: \n{label_encoder.classes_}")

# using pandas to do One Hot Encoding 
##### choosing not to one hot encode the 'y' target variable
columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact']
label_cols = ['month', 'y']
pd_one_hot_data = pd.get_dummies(data, columns=columns)

print(f"Pandas Encoded term deposit data: \n{pd_one_hot_data}") 

#### dataset now has a size of 40000 rows x 35 columns

# using sklearn to do One Hot Encoding
###### choosing not the encode the 'y' target variable
# categor_cols = data.select_dtypes(include=['object']).columns.tolist()
# print(categor_cols)

# init One Hot Encoder
encoder = OneHotEncoder(sparse_output=False) 

#apply one-hot encoding to categorical columns
sk_one_hot_encoded = encoder.fit_transform(data[columns])

#create one hot encoded columns into a dataframe
##### also want to get the feature names of the encoded columns
sk_one_hot_data = pd.DataFrame(sk_one_hot_encoded, columns=encoder.get_feature_names_out(columns))

#concat one hot dataframe with og dataframe
sk_encoded_data = pd.concat([data,sk_one_hot_data], axis=1)

# drop og categor cols
sk_one_hot_data = sk_encoded_data.drop(columns, axis=1)

# print final encoded dataframe
print(f"Sklearn Encoded term deposit data: \n{sk_one_hot_data}")

##### dataset now has a size of 40000 rows x 35 columns

# save sk_one_hot_data to a new csv file
sk_one_hot_data.to_csv('one-hot-term-deposit-data.csv', index=False)

# now investigte correlation among features
one_hot_feats = sk_one_hot_data.drop('y', axis=1)
print(one_hot_feats.columns.tolist())

corr = one_hot_feats.corr()
print(f"printing the correlation: \n{corr}")
# visualize
import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', annot_kws={"size": 6})
# settin the font
sns.set_theme(font_scale=0.5)
# set labels
plt.title('Correlation Heatmap')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
# show plot
plt.show()

# taking a peek at VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF dataframe
vif_df = pd.DataFrame()
# calculating VIF for each feature
vif_df["feature"] = one_hot_feats.columns
vif_df["VIF"] = [variance_inflation_factor(one_hot_feats.values, i) for i in range(one_hot_feats.shape[1])]
print(f"VIF matrix: \n{vif_df}")
