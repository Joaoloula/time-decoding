from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_mrt
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = np.arange(14)
tr = 2
k = 2000

# TDE parameters
time_window = 3
delay = 2

scores, subjects, models = [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_mrt(subject)
    session_id_fmri = [[session] * len(fmri[session])
                       for session in range(len(fmri))]

    # Stack the BOLD signals and the design matrices
    fmri = np.vstack(fmri)
    stimuli = np.vstack(stimuli)
    session_id_fmri = np.hstack(session_id_fmri)

    lplo = LeavePLabelOut(session_id_fmri, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        # Time window + feature selection
        fmri_train, fmri_test, stimuli_train, stimuli_test = de.apply_time_window(
            fmri_train, fmri_test, stimuli_train, stimuli_test, delay=delay,
            time_window=time_window, k=k)

        # Fit ridge regression
        prediction, score = de.fit_ridge(fmri_train, fmri_test, stimuli_train,
                                         stimuli_test)

        # Fit a logistic regression for deconvolution
        accuracy = de.ridge_scoring(prediction[:, 1:], stimuli_test[:, 1:])

        scores.append(accuracy)
        subjects.append(subject + 1)
        models.append('time-delayed embedding')

    print('finished subject ' + str(subject))
