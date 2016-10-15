from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_mrt
import time_decoding.decoding as de
import pandas as pd
import numpy as np

# Parameters
subject_list = np.arange(14)
k = 10000
tr = 2.
model = 'GLM'

# SVM parameters
time_window = 4
delay = 1

scores, subjects, models = [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_mrt(subject)
    session_id_onset = [[session] * len(onsets[session])
                        for session in range(len(onsets))]
    stimuli = np.array(stimuli)
    fmri_windows = de.apply_time_window(fmri, stimuli, time_window, delay)

    conditions = np.hstack([conditions[block][:len(fmri_windows[block])]
                            for block in range(len(fmri_windows))])
    fmri_windows = np.vstack(fmri_windows)
    session_id_onset = np.hstack(session_id_onset)

    junk_mask = np.where(conditions != 'ju')
    conditions = conditions[junk_mask]
    fmri_windows = fmri_windows[junk_mask]
    session_id_onset = session_id_onset[junk_mask]

    lplo = LeavePLabelOut(session_id_onset, p=1)
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
        subjects.append(subject + 1)
        models.append(model)

    print('finished subject ' + str(subject))

dict = {}
dict['subject'] = subjects
dict['model'] = models
dict['accuracy'] = scores

data = pd.DataFrame(dict)
print(np.mean(data['accuracy']))
