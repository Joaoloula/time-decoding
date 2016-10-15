from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_texture
import time_decoding.decoding as de
import pandas as pd
import numpy as np

# Parameters
subject_list = range(7)
k = 10000
tr = 2.4

# stSVM parameters
model = 'spatiotemporal svm'
time_window = 3
delay = 1

scores, models, subjects = [], [], []
for subject in subject_list:
    subject_scores = []
    # Read data
    fmri, stimuli, onsets, conditions = read_data_texture(subject)
    separate_conditions = [[scan + 1 if conditions[session][scan] != '0' else 0
                            for scan in range(len(conditions[session]))]
                           for session in range(len(conditions))]

    # stimuli = stimuli.reshape(6, -1, 8)
    fmri_windows = de.apply_time_window(fmri, stimuli, time_window, delay)

    fmri_windows = [[fmri_windows[trial][cond]
                     for cond in range(len(conditions[trial]))
                     if conditions[trial][cond] != '0']
                    for trial in range(len(conditions))]
    session_id_onset = np.array([[session] * len(fmri_windows[session])
                                 for session in range(len(onsets))]).ravel()
    fmri_windows = np.vstack(fmri_windows)
    conditions = np.hstack(conditions)
    conditions_ = np.array([condition[:2] for condition in conditions
                            if condition != '0'])
    lplo = LeavePLabelOut(session_id_onset, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        fmri_windows_train, fmri_windows_test = (fmri_windows[train_index],
                                                 fmri_windows[test_index])
        conditions_train, conditions_test = (conditions_[train_index],
                                             conditions_[test_index])

        # Feature selection
        fmri_windows_train, fmri_windows_test = de.feature_selection(
            fmri_windows_train, fmri_windows_test, conditions_train, k=k)

        # Fit a logistic regression to score the model
        accuracy = de.svm_scoring(fmri_windows_train, fmri_windows_test,
                                  conditions_train, conditions_test)

        scores.append(accuracy)
        subjects.append(subject)
        models.append(model)
        print('finished one CV step')

    print('finished subject ' + str(subject))

dict = {}
dict['subject'] = subjects
dict['model'] = models
dict['accuracy'] = scores

data = pd.DataFrame(dict)
np.mean(data['accuracy'])
