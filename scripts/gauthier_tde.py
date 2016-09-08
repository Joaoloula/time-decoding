from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_gauthier
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = range(11)
tr = 1.5
k = 10000

# TDE parameters
time_window = 4
delay = 1

scores, subjects, models, isis = [], [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_fmri = [[session] * len(fmri[session])
                       for session in range(len(fmri))]
    session_id_fmri = [[19.2 / len(onsets[session])] * len(fmri[session])
                       for session in range(len(onsets))]

    # Stack the BOLD signals and the design matrices
    fmri = np.vstack(fmri)
    stimuli = np.vstack(stimuli)
    session_id_fmri = np.hstack(session_id_fmri)

    lplo = LeavePLabelOut(session_id_fmri, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        n_points = np.sum(stimuli_test[:, 1:])
        if n_points == 12 * 4:
            isi = 1.6
            time_window = 1
            delay = 3

        elif n_points == 6 * 4:
            isi = 3.2
            time_window = 2
            delay = 3

        elif n_points == 4 * 4:
            isi = 4.8
            time_window = 3
            delay = 3

        else:
            continue

        # Time window + feature selection
        fmri_train, fmri_test, stimuli_train, stimuli_test = de.apply_time_window(
            fmri_train, fmri_test, stimuli_train, stimuli_test, delay=delay,
            time_window=time_window, k=k)

        # Fit ridge regression
        prediction, score = de.fit_ridge(fmri_train, fmri_test, stimuli_train,
                                         stimuli_test)

        # Fit a logistic regression for deconvolution
        accuracy = de.ridge_scoring(prediction, stimuli_test)

        scores.append(accuracy)
        subjects.append(subject + 1)
        models.append('time-delayed embedding')
        isis.append(isi)

    print('finished subject ' + str(subject))
