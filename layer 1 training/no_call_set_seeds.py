'''
Author: Sophia Wagner
Date: 9/5/2024
Description:  iterating through 40 random seeds to compare the recall scores of the final models
'''

# import libraries
import random 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

#load in data, using the one-hot encoded data
no_call_data = pd.read_csv('no-call-one-hot-term-deposit-data.csv')
full_dataset = pd.read_csv('one-hot-term-deposit-data.csv')


#making sure data is loaded in, checking y counts
#print('no call data y counts: ', no_call_data['y'].value_counts()) #no: 37104, yes: 2896        

# separate features and target variable
X_no_call = no_call_data.drop('y', axis=1)
y_no_call = no_call_data['y']

X_full = full_dataset.drop('y', axis=1)
y_full = full_dataset['y']

'''
#####################################################################

# empty lists to store the results per model
rfc_results = []
gbc_results = []
dtc_results = []

# iterate through 40 random seeds
for i in range(20):

    # import random int and set seed
    seed = random.randint(1000,9999)
    # seed = 1234 <- eventually change to set seed to train all models
    print('seed: ', seed)

    # split the data into training and testing sets
    #want to make sure both datasets are split the same
    X_train_no_call, X_test_no_call, y_train_no_call, y_test_no_call = train_test_split(X_no_call, y_no_call, test_size=0.2, random_state=seed)

    # # sampling the no call training data
    # rus = RandomUnderSampler(random_state=seed, replacement=False)
    # X_tr_rus, y_tr_rus = rus.fit_resample(X_train_no_call, y_train_no_call)

    # creating the Layer 1 models
    dtc = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', 
                                splitter='random', max_depth=1, max_features='sqrt', 
                                min_samples_leaf=19, min_samples_split=15, random_state=seed)

    gbc = GradientBoostingClassifier(n_estimators=82, loss='log_loss', learning_rate=0.15720724871814273, 
                                    criterion='friedman_mse', max_depth=15, min_samples_leaf=2, 
                                    min_samples_split=22, tol=0.0008952425309703175, random_state=seed)

    rfc = RandomForestClassifier(n_estimators=177, criterion='entropy', max_depth=29,
                                min_samples_split=6, min_samples_leaf=1, max_features=None, random_state=seed)

    scorer = make_scorer(recall_score)

    # rfc_rec_cvscores = cross_val_score(rfc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)
    # gbc_rec_cvscores = cross_val_score(gbc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)
    # dtc_rec_cvscores = cross_val_score(dtc, X_tr_rus, y_tr_rus, scoring=scorer, cv=5)

    # rfc.fit(X_tr_rus, y_tr_rus)
    # gbc.fit(X_tr_rus, y_tr_rus)
    # dtc.fit(X_tr_rus, y_tr_rus)

    rfc_rec_cvscores = cross_val_score(rfc, X_train_no_call, y_train_no_call, scoring=scorer, cv=5)
    gbc_rec_cvscores = cross_val_score(gbc, X_train_no_call, y_train_no_call, scoring=scorer, cv=5)
    dtc_rec_cvscores = cross_val_score(dtc, X_train_no_call, y_train_no_call, scoring=scorer, cv=5)

    rfc.fit(X_train_no_call, y_train_no_call)
    gbc.fit(X_train_no_call, y_train_no_call)
    dtc.fit(X_train_no_call, y_train_no_call)

    y_pred_rfc = rfc.predict(X_test_no_call)
    y_pred_gbc = gbc.predict(X_test_no_call)
    y_pred_dtc = dtc.predict(X_test_no_call)

    rfc_report = classification_report(y_test_no_call, y_pred_rfc, output_dict=True)
    gbc_report = classification_report(y_test_no_call, y_pred_gbc, output_dict=True)
    dtc_report = classification_report(y_test_no_call, y_pred_dtc, output_dict=True)

    # rfc_recall_df = rfc_recall_df.append({'seed': seed, 'cv_mean': rfc_rec_cvscores.mean(), 
    #                                       'class_0_recall': rfc_report['0']['recall'], 
    #                                       'class_1_recall': rfc_report['1']['recall']}, ignore_index=True)
    # gbc_recall_df = gbc_recall_df.append({'seed': seed, 'cv_mean': gbc_rec_cvscores.mean(), 
    #                                       'class_0_recall': gbc_report['0']['recall'], 
    #                                       'class_1_recall': gbc_report['1']['recall']}, ignore_index=True)
    # dtc_recall_df = dtc_recall_df.append({'seed': seed, 'cv_mean': dtc_rec_cvscores.mean(),
    #                                         'class_0_recall': dtc_report['0']['recall'], 
    #                                         'class_1_recall': dtc_report['1']['recall']}, ignore_index=True)
    
    rfc_results.append({'seed': seed, 'cv_mean': rfc_rec_cvscores.mean(),
                        'class_0_recall': rfc_report['0']['recall'], 
                        'class_1_recall': rfc_report['1']['recall']})
    gbc_results.append({'seed': seed, 'cv_mean': gbc_rec_cvscores.mean(),
                        'class_0_recall': gbc_report['0']['recall'], 
                        'class_1_recall': gbc_report['1']['recall']})
    dtc_results.append({'seed': seed, 'cv_mean': dtc_rec_cvscores.mean(),
                        'class_0_recall': dtc_report['0']['recall'], 
                        'class_1_recall': dtc_report['1']['recall']})

    print(i)
    #print('checking rfc results list: ', rfc_results)

print('done!')

#convert list to dataframe
rfc_recall_df = pd.DataFrame(rfc_results)
gbc_recall_df = pd.DataFrame(gbc_results)
dtc_recall_df = pd.DataFrame(dtc_results)

print('rfc dataframe: ', rfc_recall_df)
print('gbc dataframe: ', gbc_recall_df)
print('dtc dataframe: ', dtc_recall_df)

#save dataframes to csv
rfc_recall_df.to_csv('rfc_recall_seeds.csv', index=False)
gbc_recall_df.to_csv('gbc_recall_seeds.csv', index=False)
dtc_recall_df.to_csv('dtc_recall_seeds.csv', index=False)

'''

# now creating the evaluations for each seed type
'''
one loop for one model at a time
TODO: array/list of seeds to use, gathered by model

TODO: for each seed train the associated model
TODO: print recall and eval matrixes

TODO: pass in trained model to evaluate on full dataset
TODO:  print % and time saved calcualtions for each model

'''

rfc_rus_seeds = [8603, 9059, 4197, 9698, 9814, 9346]
gbc_rus_seeds = [5790, 8603, 5620, 9059, 2295]
dtc_rus_seeds = [3048, 8549, 2343, 6399, 4727]

dtc_nosamp_seeds = [6467, 8609, 9916, 4958, 6059, 1905, 6855, 9800] 

# calculating hours, minutes, and seconds of a given seconds value
def convert_duration(seconds):
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    return hrs, mins, secs

# store all _seeds array into one iterable
all_seeds = [rfc_rus_seeds, gbc_rus_seeds, dtc_rus_seeds, dtc_nosamp_seeds]
names_seeds = ['rfc_rus_seeds', 'gbc_rus_seeds', 'dtc_rus_seeds', 'dtc_nosamp_seeds']

total_secs_spent = full_dataset['duration'].sum()
total_secs_nosub = full_dataset[full_dataset['y'] == 0]['duration'].sum()
total_secs_yessub = full_dataset[full_dataset['y'] == 1]['duration'].sum()
total_hrs, total_mins, total_secs = convert_duration(total_secs_spent)
nosub_hrs, nosub_mins, nosub_secs = convert_duration(total_secs_nosub)
yessub_hrs, yessub_mins, yessub_secs = convert_duration(total_secs_yessub)

print(f"Total of {total_hrs} hours, {total_mins} minutes, and {total_secs} seconds spent calling all 40,000 customers.")
print(f"Total of {nosub_hrs} hours, {nosub_mins} minutes, and {nosub_secs} seconds spent calling customers who did not subscribe.")
print(f"Total of {yessub_hrs} hours, {yessub_mins} minutes, and {yessub_secs} seconds spent calling customers who subscribed.")

scoring = make_scorer(recall_score)

for model in all_seeds:

    model_name = names_seeds[all_seeds.index(model)]

    for seed in model: 

        # split the data into training and testing sets
        X_train_no_call, X_test_no_call, y_train_no_call, y_test_no_call = train_test_split(X_no_call, y_no_call, test_size=0.2, random_state=seed)

        # sampling the no call training data
        rus = RandomUnderSampler(random_state=seed, replacement=False)
        X_tr_rus, y_tr_rus = rus.fit_resample(X_train_no_call, y_train_no_call)

        # creating the Layer 1 models
        print ('seed: ', seed)
        print ('model name: ', names_seeds[all_seeds.index(model)])

        if model_name == 'rfc_rus_seeds':
        
            print('training RFC model')
            rfc = RandomForestClassifier(n_estimators=177, criterion='entropy', max_depth=29,
                                        min_samples_split=6, min_samples_leaf=1, max_features=None, random_state=seed)
            rfc_cv = cross_val_score(rfc, X_tr_rus, y_tr_rus, scoring=make_scorer(recall_score), cv=5)
            rfc.fit(X_tr_rus, y_tr_rus)
            rfc_pred = rfc.predict(X_test_no_call)
            rfc_report = classification_report(y_test_no_call, rfc_pred, output_dict=True)
            rfc_cm = confusion_matrix(y_test_no_call, rfc_pred)
            print(f" given seed: {seed} \n for model: {names_seeds[all_seeds.index(model)]} \n  cross val recall: {rfc_cv} \n cross val recall mean: {rfc_cv.mean()}") 
            print(" training data confusion matrix: ", rfc_cm)
            print(" training data classification report: ", rfc_report )

            print("full no call dataset model predictions")

            rfc_pred = rfc.predict(X_no_call)
            print("confusion matrix: ", confusion_matrix(y_no_call, rfc_pred))
            print("classification report: ", classification_report(y_no_call, rfc_pred, output_dict=True))

            #save index of all false positive predictions
            fp_indx = np.where((y_no_call == 0) & (rfc_pred == 1))[0]
            # save index of all true positive predictions
            tp_indx = np.where((y_no_call == 1) & (rfc_pred == 1))[0]
            # save index of all false negative predictions
            fn_indx = np.where((y_no_call == 1) & (rfc_pred == 0))[0]
            # save index of all true negative predictions
            tn_indx = np.where((y_no_call == 0) & (rfc_pred == 0))[0]

            total_subs = len(tp_indx) + len(fn_indx)

            # calculate 'duration' of all indx categories
            tp_duration = convert_duration(full_dataset.iloc[tp_indx]['duration'].sum())
            fn_duration = convert_duration(full_dataset.iloc[fn_indx]['duration'].sum())
            fp_duration = convert_duration(full_dataset.iloc[fp_indx]['duration'].sum())
            tn_duration = convert_duration(full_dataset.iloc[tn_indx]['duration'].sum())

            print('Calculating time saved and % of subs lost')
            print('seed: ', seed)
            print('model name: ', names_seeds[all_seeds.index(model)])

            print(f"Total call time from model True Positives: {tp_duration[0]} hours, {tp_duration[1]} minutes, and {tp_duration[2]} seconds")
            print(f"Total call time from model False Negatives: {fn_duration[0]} hours, {fn_duration[1]} minutes, and {fn_duration[2]} seconds")
            print(f"Total call time from model False Positives: {fp_duration[0]} hours, {fp_duration[1]} minutes, and {fp_duration[2]} seconds")
            print(f"Total call time from model True Negatives: {tn_duration[0]} hours, {tn_duration[1]} minutes, and {tn_duration[2]} seconds")

            call_time_saved = full_dataset.iloc[fn_indx]['duration'].sum() + full_dataset.iloc[tn_indx]['duration'].sum()
            convert_call_time_saved = convert_duration(call_time_saved)
            print(f"According to {model_name} pred, total call effort saved: {convert_call_time_saved[0]} hours, {convert_call_time_saved[1]} minutes, and {convert_call_time_saved[2]} seconds")

            pred_time_needed = full_dataset.iloc[tp_indx]['duration'].sum() + full_dataset.iloc[fp_indx]['duration'].sum()
            convert_pred_time_needed = convert_duration(pred_time_needed)
            print(f"According to {model_name} pred, total call effort worth using: {convert_pred_time_needed[0]} hours, {convert_pred_time_needed[1]} minutes, and {convert_pred_time_needed[2]} seconds")

            print(f"from {model_name} pred, % of subs to be lost: {len(fn_indx) / total_subs * 100}%")

            print(f"by using {model_name} preds, projected to save {call_time_saved / total_secs_spent * 100}% of total call time previously spent")


        elif model_name == 'gbc_rus_seeds':

            print('training GBC model')
            gbc = GradientBoostingClassifier(n_estimators=82, loss='log_loss', learning_rate=0.15720724871814273, 
                                    criterion='friedman_mse', max_depth=15, min_samples_leaf=2, 
                                    min_samples_split=22, tol=0.0008952425309703175, random_state=seed)
            gbc_cv = cross_val_score(gbc, X_tr_rus, y_tr_rus, scoring=make_scorer(recall_score), cv=5)
            gbc.fit(X_tr_rus, y_tr_rus)
            gbc_pred = gbc.predict(X_test_no_call)
            gbc_report = classification_report(y_test_no_call, gbc_pred, output_dict=True)
            gbc_cm = confusion_matrix(y_test_no_call, gbc_pred)
            print(f" given seed: {seed} \n for model: {names_seeds[all_seeds.index(model)]} \n  cross val recall: {gbc_cv} \n cross val recall mean: {gbc_cv.mean()}") 
            print(" training set confusion matrix: ", gbc_cm)
            print(" training set classification report: ", gbc_report)
            
            print("full no call dataset model predictions")

            gbc_pred = gbc.predict(X_no_call)
            print("confusion matrix: ", confusion_matrix(y_no_call, gbc_pred))
            print("classification report: ", classification_report(y_no_call, gbc_pred, output_dict=True))

            #save index of all false positive predictions
            fp_indx = np.where((y_no_call == 0) & (gbc_pred == 1))[0]
            # save index of all true positive predictions
            tp_indx = np.where((y_no_call == 1) & (gbc_pred == 1))[0]
            # save index of all false negative predictions
            fn_indx = np.where((y_no_call == 1) & (gbc_pred == 0))[0]
            # save index of all true negative predictions
            tn_indx = np.where((y_no_call == 0) & (gbc_pred == 0))[0]

            total_subs = len(tp_indx) + len(fn_indx)
     
            # calculate 'duration' of all indx categories
            tp_duration = convert_duration(full_dataset.iloc[tp_indx]['duration'].sum())
            fn_duration = convert_duration(full_dataset.iloc[fn_indx]['duration'].sum())
            fp_duration = convert_duration(full_dataset.iloc[fp_indx]['duration'].sum())
            tn_duration = convert_duration(full_dataset.iloc[tn_indx]['duration'].sum())

            print('Calculating time saved and % of subs lost')
            print('seed: ', seed)
            print('model name: ', names_seeds[all_seeds.index(model)])

            print(f"Total call time from model True Positives: {tp_duration[0]} hours, {tp_duration[1]} minutes, and {tp_duration[2]} seconds")
            print(f"Total call time from model False Negatives: {fn_duration[0]} hours, {fn_duration[1]} minutes, and {fn_duration[2]} seconds")
            print(f"Total call time from model False Positives: {fp_duration[0]} hours, {fp_duration[1]} minutes, and {fp_duration[2]} seconds")
            print(f"Total call time from model True Negatives: {tn_duration[0]} hours, {tn_duration[1]} minutes, and {tn_duration[2]} seconds")

            call_time_saved = full_dataset.iloc[fn_indx]['duration'].sum() + full_dataset.iloc[tn_indx]['duration'].sum()
            convert_call_time_saved = convert_duration(call_time_saved)
            print(f"According to {model_name} pred, total call effort saved: {convert_call_time_saved[0]} hours, {convert_call_time_saved[1]} minutes, and {convert_call_time_saved[2]} seconds")

            pred_time_needed = full_dataset.iloc[tp_indx]['duration'].sum() + full_dataset.iloc[fp_indx]['duration'].sum()
            convert_pred_time_needed = convert_duration(pred_time_needed)
            print(f"According to {model_name} pred, total call effort worth using: {convert_pred_time_needed[0]} hours, {convert_pred_time_needed[1]} minutes, and {convert_pred_time_needed[2]} seconds")

            print(f"from {model_name} pred, % of subs to be lost: {len(fn_indx) / total_subs * 100}%")

            print(f"by using {model_name} preds, projected to save {call_time_saved / total_secs_spent * 100}% of total call time previously spent")


        elif model_name == 'dtc_rus_seeds':

            print('training DTC model')
            dtc = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', 
                                splitter='random', max_depth=1, max_features='sqrt', 
                                min_samples_leaf=19, min_samples_split=15, random_state=seed)
            dtc_cv = cross_val_score(dtc, X_tr_rus, y_tr_rus, scoring=make_scorer(recall_score), cv=5)
            dtc.fit(X_tr_rus, y_tr_rus)
            dtc_pred = dtc.predict(X_test_no_call)
            dtc_report = classification_report(y_test_no_call, dtc_pred, output_dict=True)
            dtc_cm = confusion_matrix(y_test_no_call, dtc_pred)
            print(f" given seed: {seed} \n for model: {names_seeds[all_seeds.index(model)]} \n  cross val recall: {dtc_cv} \n cross val recall mean: {dtc_cv.mean()}") 
            print(" training set confusion matrix: ", dtc_cm)
            print(" training set classification report: ", dtc_report)

            print("full no call dataset model predictions")

            dtc_pred = rfc.predict(X_no_call)
            print("confusion matrix: ", confusion_matrix(y_no_call, dtc_pred))
            print("classification report: ", classification_report(y_no_call, dtc_pred, output_dict=True))

            #save index of all false positive predictions
            fp_indx = np.where((y_no_call == 0) & (dtc_pred == 1))[0]
            # save index of all true positive predictions
            tp_indx = np.where((y_no_call == 1) & (dtc_pred == 1))[0]
            # save index of all false negative predictions
            fn_indx = np.where((y_no_call == 1) & (dtc_pred == 0))[0]
            # save index of all true negative predictions
            tn_indx = np.where((y_no_call == 0) & (dtc_pred == 0))[0]

            total_subs = len(tp_indx) + len(fn_indx)

            # calculate 'duration' of all indx categories
            tp_duration = convert_duration(full_dataset.iloc[tp_indx]['duration'].sum())
            fn_duration = convert_duration(full_dataset.iloc[fn_indx]['duration'].sum())
            fp_duration = convert_duration(full_dataset.iloc[fp_indx]['duration'].sum())
            tn_duration = convert_duration(full_dataset.iloc[tn_indx]['duration'].sum())

            print('Calculating time saved and % of subs lost')
            print('seed: ', seed)
            print('model name: ', names_seeds[all_seeds.index(model)])

            print(f"Total call time from model True Positives: {tp_duration[0]} hours, {tp_duration[1]} minutes, and {tp_duration[2]} seconds")
            print(f"Total call time from model False Negatives: {fn_duration[0]} hours, {fn_duration[1]} minutes, and {fn_duration[2]} seconds")
            print(f"Total call time from model False Positives: {fp_duration[0]} hours, {fp_duration[1]} minutes, and {fp_duration[2]} seconds")
            print(f"Total call time from model True Negatives: {tn_duration[0]} hours, {tn_duration[1]} minutes, and {tn_duration[2]} seconds")

            call_time_saved = full_dataset.iloc[fn_indx]['duration'].sum() + full_dataset.iloc[tn_indx]['duration'].sum()
            convert_call_time_saved = convert_duration(call_time_saved)
            print(f"According to {model_name} pred, total call effort saved: {convert_call_time_saved[0]} hours, {convert_call_time_saved[1]} minutes, and {convert_call_time_saved[2]} seconds")

            pred_time_needed = full_dataset.iloc[tp_indx]['duration'].sum() + full_dataset.iloc[fp_indx]['duration'].sum()
            convert_pred_time_needed = convert_duration(pred_time_needed)
            print(f"According to {model_name} pred, total call effort worth using: {convert_pred_time_needed[0]} hours, {convert_pred_time_needed[1]} minutes, and {convert_pred_time_needed[2]} seconds")

            print(f"from {model_name} pred, % of subs to be lost: {len(fn_indx) / total_subs * 100}%")

            print(f"by using {model_name} preds, projected to save {call_time_saved / total_secs_spent * 100}% of total call time previously spent")


        elif model_name == 'dtc_nosamp_seeds':

            print('training DTC model with no sampling')
            dtc = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', 
                                splitter='random', max_depth=1, max_features='sqrt', 
                                min_samples_leaf=19, min_samples_split=15, random_state=seed)
            dtc_cv = cross_val_score(dtc, X_train_no_call, y_train_no_call, scoring=make_scorer(recall_score), cv=5)
            dtc.fit(X_train_no_call, y_train_no_call)
            dtc_pred = dtc.predict(X_test_no_call)
            dtc_report = classification_report(y_test_no_call, dtc_pred, output_dict=True)
            dtc_cm = confusion_matrix(y_test_no_call, dtc_pred)
            print(f" given seed: {seed} \n for model: {names_seeds[all_seeds.index(model)]} \n  cross val recall: {dtc_cv} \n cross val recall mean: {dtc_cv.mean()}")
            print(" training set confusion matrix: ", dtc_cm)
            print(" training set classification report: ", dtc_report)

            print("full no call dataset model predictions")

            dtc_pred = dtc.predict(X_no_call)
            print("confusion matrix: ", confusion_matrix(y_no_call, rfc_pred))
            print("classification report: ", classification_report(y_no_call, rfc_pred, output_dict=True))

            #save index of all false positive predictions
            fp_indx = np.where((y_no_call == 0) & (dtc_pred == 1))[0]
            # save index of all true positive predictions
            tp_indx = np.where((y_no_call == 1) & (dtc_pred == 1))[0]
            # save index of all false negative predictions
            fn_indx = np.where((y_no_call == 1) & (dtc_pred == 0))[0]
            # save index of all true negative predictions
            tn_indx = np.where((y_no_call == 0) & (dtc_pred == 0))[0]

            total_subs = len(tp_indx) + len(fn_indx)

      # calculate 'duration' of all indx categories
            tp_duration = convert_duration(full_dataset.iloc[tp_indx]['duration'].sum())
            fn_duration = convert_duration(full_dataset.iloc[fn_indx]['duration'].sum())
            fp_duration = convert_duration(full_dataset.iloc[fp_indx]['duration'].sum())
            tn_duration = convert_duration(full_dataset.iloc[tn_indx]['duration'].sum())

            print('Calculating time saved and % of subs lost')
            print('seed: ', seed)
            print('model name: ', names_seeds[all_seeds.index(model)])

            print(f"Total call time from model True Positives: {tp_duration[0]} hours, {tp_duration[1]} minutes, and {tp_duration[2]} seconds")
            print(f"Total call time from model False Negatives: {fn_duration[0]} hours, {fn_duration[1]} minutes, and {fn_duration[2]} seconds")
            print(f"Total call time from model False Positives: {fp_duration[0]} hours, {fp_duration[1]} minutes, and {fp_duration[2]} seconds")
            print(f"Total call time from model True Negatives: {tn_duration[0]} hours, {tn_duration[1]} minutes, and {tn_duration[2]} seconds")

            call_time_saved = full_dataset.iloc[fn_indx]['duration'].sum() + full_dataset.iloc[tn_indx]['duration'].sum()
            convert_call_time_saved = convert_duration(call_time_saved)
            print(f"According to {model_name} pred, total call effort saved: {convert_call_time_saved[0]} hours, {convert_call_time_saved[1]} minutes, and {convert_call_time_saved[2]} seconds")

            pred_time_needed = full_dataset.iloc[tp_indx]['duration'].sum() + full_dataset.iloc[fp_indx]['duration'].sum()
            convert_pred_time_needed = convert_duration(pred_time_needed)
            print(f"According to {model_name} pred, total call effort worth using: {convert_pred_time_needed[0]} hours, {convert_pred_time_needed[1]} minutes, and {convert_pred_time_needed[2]} seconds")

            print(f"from {model_name} pred, % of subs to be lost: {len(fn_indx) / total_subs * 100}%")

            print(f"by using {model_name} preds, projected to save {call_time_saved / total_secs_spent * 100}% of total call time previously spent")

print('done!')















