'''
Author: Sophia Wagner
Date: 9/15/2024
Description: iterating through 40 random seeds to compare the precision scores of
             the final gradient boosting and random forest classifier models for layer 2
'''

# import libraries
import random 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, precision_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

#load in data, using the one-hot encoded data
full_dataset = pd.read_csv('data/one-hot-term-deposit-data.csv')

#making sure data is loaded in, checking y counts
#print('no call data y counts: ', no_call_data['y'].value_counts()) #no: 37104, yes: 2896        

# separate features and target variable
X_full = full_dataset.drop('y', axis=1)
y_full = full_dataset['y']

#####################################################################

'''
# empty lists to store the results per model
rfc_results = []
gbc_results = []

# iterate through 40 random seeds
for i in range(40):

    # import random int and set seed
    seed = random.randint(1000,9999)
    # seed = 1234 <- eventually change to set seed to train all models
    print('seed: ', seed)

    # split the data into training and testing sets
    #want to make sure both datasets are split the same
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)

    # # sampling the no call training data
    rus = RandomUnderSampler(random_state=seed, replacement=False)
    X_tr_rus, y_tr_rus = rus.fit_resample(X_train, y_train)

    # creating the Layer 1 models
    gbc = GradientBoostingClassifier(criterion='squared_error', learning_rate=0.28999124516900915,
                                     max_depth=2, max_features=None, n_estimators=130, random_state=seed)

    rfc = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=9, 
                                max_features=None, min_samples_leaf=4, min_samples_split=20, 
                                n_estimators=220, random_state=seed)

    scorer = make_scorer(precision_score)

    rfc_prec_cvscores = cross_val_score(rfc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)
    gbc_prec_cvscores = cross_val_score(gbc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)

    rfc.fit(X_tr_rus, y_tr_rus)
    gbc.fit(X_tr_rus, y_tr_rus)

    # rfc_rec_cvscores = cross_val_score(rfc, X_train_no_call, y_train_no_call, scoring=scorer, cv=5)
    # gbc_rec_cvscores = cross_val_score(gbc, X_train_no_call, y_train_no_call, scoring=scorer, cv=5)

    # rfc.fit(X_train_no_call, y_train_no_call)
    # gbc.fit(X_train_no_call, y_train_no_call)

    y_pred_rfc = rfc.predict(X_test)
    y_pred_gbc = gbc.predict(X_test)

    rfc_report = classification_report(y_test, y_pred_rfc, output_dict=True)
    gbc_report = classification_report(y_test, y_pred_gbc, output_dict=True)

    # rfc_recall_df = rfc_recall_df.append({'seed': seed, 'cv_mean': rfc_rec_cvscores.mean(), 
    #                                       'class_0_recall': rfc_report['0']['recall'], 
    #                                       'class_1_recall': rfc_report['1']['recall']}, ignore_index=True)
    # gbc_recall_df = gbc_recall_df.append({'seed': seed, 'cv_mean': gbc_rec_cvscores.mean(), 
    #                                       'class_0_recall': gbc_report['0']['recall'], 
    #                                       'class_1_recall': gbc_report['1']['recall']}, ignore_index=True)
    
    rfc_results.append({'seed': seed, 'cv_mean': rfc_prec_cvscores.mean(),
                        'class_0_precision': rfc_report['0']['precision'], 
                        'class_1_precision': rfc_report['1']['precision'],
                        'accuracy': rfc_report['accuracy']})
    gbc_results.append({'seed': seed, 'cv_mean': gbc_prec_cvscores.mean(),
                        'class_0_precision': gbc_report['0']['precision'], 
                        'class_1_precision': gbc_report['1']['precision'],
                        'accuracy': gbc_report['accuracy']})
    
    print(i)
    #print('checking rfc results list: ', rfc_results)

print('done!')

#convert list to dataframe
rfc_recall_df = pd.DataFrame(rfc_results)
gbc_recall_df = pd.DataFrame(gbc_results)

print('rfc dataframe: ', rfc_recall_df)
print('gbc dataframe: ', gbc_recall_df)

#save dataframes to csv
rfc_recall_df.to_csv('rfc_recall_seeds.csv', index=False)
gbc_recall_df.to_csv('gbc_recall_seeds.csv', index=False)

'''

# setting the highest performing seed
seed = 7748

print('seed: ', seed)

# split training data
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)

# rus on training 
rus = RandomUnderSampler(random_state=seed, replacement=False)
X_tr_rus, y_tr_rus = rus.fit_resample(X_train, y_train)

# reusing the RFC model
rfc = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=9, 
                            max_features=None, min_samples_leaf=4, min_samples_split=20, 
                            n_estimators=220, random_state=seed)

scorer = make_scorer(precision_score)

rfc_prec_cvscores = cross_val_score(rfc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)

rfc.fit(X_tr_rus, y_tr_rus)

y_pred_rfc = rfc.predict(X_test)

#printing the rfc model
print('evaluating trained rfc on the test set')
print('model: ', rfc)

print('rfc precision cv scores: ', rfc_prec_cvscores)
print('rfc precision cv mean: ', rfc_prec_cvscores.mean())
print('rfc confusion matrix: ', confusion_matrix(y_test, y_pred_rfc))
print('rfc classification report: ', classification_report(y_test, y_pred_rfc))

# now evaluating the model on the full dataset


# calculating hours, minutes, and seconds of a given seconds value
def convert_duration(seconds):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return hrs, mins, secs

total_secs_spent = full_dataset['duration'].sum()
total_secs_nosub = full_dataset[full_dataset['y'] == 0]['duration'].sum()
total_secs_yessub = full_dataset[full_dataset['y'] == 1]['duration'].sum()
total_hrs, total_mins, total_secs = convert_duration(total_secs_spent)
nosub_hrs, nosub_mins, nosub_secs = convert_duration(total_secs_nosub)
yessub_hrs, yessub_mins, yessub_secs = convert_duration(total_secs_yessub)

print(f"Total of {total_hrs} hours, {total_mins} minutes, and {total_secs} seconds spent calling all 40,000 customers.")
print(f"Total of {nosub_hrs} hours, {nosub_mins} minutes, and {nosub_secs} seconds spent calling customers who did not subscribe.")
print(f"Total of {yessub_hrs} hours, {yessub_mins} minutes, and {yessub_secs} seconds spent calling customers who subscribed.")

rfc_full_prec_score = cross_val_score(rfc, X_full, y_full, scoring=scorer, cv=5)

print('evaluating the rfc model on the full dataset')

print('rfc full dataset precision cv scores: ', rfc_full_prec_score)
print('rfc full dataset precision cv mean: ', rfc_full_prec_score.mean())

rfc = RandomForestClassifier(class_weight=None, criterion='entropy', max_depth=9, 
                            max_features=None, min_samples_leaf=4, min_samples_split=20, 
                            n_estimators=220, random_state=seed)

rfc_cv = cross_val_score(rfc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)
rfc.fit(X_tr_rus, y_tr_rus)
rfc_pred = rfc.predict(X_test)
rfc_report = classification_report(y_test, rfc_pred)
rfc_cm = confusion_matrix(y_test, rfc_pred)
print(" training data confusion matrix: ", rfc_cm)
print(" training data classification report: ", rfc_report )

print("full dataset model predictions")

rfc_pred = rfc.predict(X_full)
print("confusion matrix: ", confusion_matrix(y_full, rfc_pred))
print("classification report: ", classification_report(y_full, rfc_pred))

#save index of all false positive predictions
fp_indx = np.where((y_full == 0) & (rfc_pred == 1))[0]
# save index of all true positive predictions
tp_indx = np.where((y_full == 1) & (rfc_pred == 1))[0]
# save index of all false negative predictions
fn_indx = np.where((y_full == 1) & (rfc_pred == 0))[0]
# save index of all true negative predictions
tn_indx = np.where((y_full == 0) & (rfc_pred == 0))[0]

total_subs = len(tp_indx) + len(fn_indx)

# calculate 'duration' of all indx categories
tp_duration = convert_duration(full_dataset.iloc[tp_indx]['duration'].sum())
fn_duration = convert_duration(full_dataset.iloc[fn_indx]['duration'].sum())
fp_duration = convert_duration(full_dataset.iloc[fp_indx]['duration'].sum())
tn_duration = convert_duration(full_dataset.iloc[tn_indx]['duration'].sum())

print('Calculating time saved and % of subs lost')
print('seed: ', seed)

print(f"Total call time from model True Positives: {tp_duration[0]} hours, {tp_duration[1]} minutes, and {tp_duration[2]} seconds")
print(f"Total call time from model False Negatives: {fn_duration[0]} hours, {fn_duration[1]} minutes, and {fn_duration[2]} seconds")
print(f"Total call time from model False Positives: {fp_duration[0]} hours, {fp_duration[1]} minutes, and {fp_duration[2]} seconds")
print(f"Total call time from model True Negatives: {tn_duration[0]} hours, {tn_duration[1]} minutes, and {tn_duration[2]} seconds")

call_time_saved = full_dataset.iloc[fn_indx]['duration'].sum() + full_dataset.iloc[tn_indx]['duration'].sum()
convert_call_time_saved = convert_duration(call_time_saved)
print(f"According to rfc preds, total call effort saved: {convert_call_time_saved[0]} hours, {convert_call_time_saved[1]} minutes, and {convert_call_time_saved[2]} seconds")

pred_time_needed = full_dataset.iloc[tp_indx]['duration'].sum() + full_dataset.iloc[fp_indx]['duration'].sum()
convert_pred_time_needed = convert_duration(pred_time_needed)
print(f"According to rfc preds, total call effort worth using: {convert_pred_time_needed[0]} hours, {convert_pred_time_needed[1]} minutes, and {convert_pred_time_needed[2]} seconds")

print(f"from rfc preds, % of subs to be lost: {len(fn_indx) / total_subs * 100}%")

print(f"by using rfc preds, projected to save {call_time_saved / total_secs_spent * 100}% of total call time previously spent")

