from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_haxby
import time_decoding.decoding as de
import pandas as pd
import numpy as np

# Parameters
subject_list = np.arange(1, 7)
k = 10000
tr = 2.5
model = 'spatiotemporal SVM'

# SVM parameters
time_window = 10
delay = 0

scores, subjects, models = [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_haxby(subject)
    session_id_onset = [[session] * len(onsets[session])
                        for session in range(len(onsets))]
    fmri_windows = de.apply_time_window(fmri, stimuli, time_window, delay)

    fmri_windows = np.vstack(fmri_windows)
    session_id_onset = np.hstack(session_id_onset)
    conditions = np.hstack(conditions)

    lplo = LeavePLabelOut(session_id_onset, p=2)
    for train_index, test_index in lplo:
        # Split into train and test sets
        fmri_windows_train, fmri_windows_test = (fmri_windows[train_index],
                                                 fmri_windows[test_index])
        conditions_train, conditions_test = (conditions[train_index],
                                             conditions[test_index])

        # Feature selection
        fmri_windows_train, fmri_windows_test = de.feature_selection(
            fmri_windows_train, fmri_windows_test, conditions_train, k=k)

        # Fit a logistic regression to score the model
        accuracy = de.svm_scoring(fmri_windows_train, fmri_windows_test,
                                  conditions_train, conditions_test)

        scores.append(accuracy)
        subjects.append(subject)
        models.append(model)

    print('finished subject ' + str(subject))

dict = {}
dict['subject'] = subjects
dict['model'] = models
dict['accuracy'] = scores

data = pd.DataFrame(dict)
print(np.mean(data['accuracy']))
