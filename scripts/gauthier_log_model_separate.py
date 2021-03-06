from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_gauthier
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = range(11)
tr = 1.5
k = 10000

# GLM parameters
hrf_model = 'spm'
logistic_window = 2
delay = 3

scores, subjects, models, isis = [], [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_fmri = [[session] * len(fmri[session])
                       for session in range(len(fmri))]
    session_id_fmri = [[19.2 / len(onsets[session])] * len(fmri[session])
                       for session in range(len(onsets))]
    design = [de.design_matrix(len(fmri[session]), tr, onsets[session],
                               conditions[session], hrf_model=hrf_model,
                               drift_model='blank')
              for session in range(len(fmri))]

    # Stack the BOLD signals and the design matrices
    fmri = np.vstack(fmri)
    design = np.vstack(design)
    stimuli = np.vstack(stimuli)
    session_id_fmri = np.hstack(session_id_fmri)

    lplo = LeavePLabelOut(session_id_fmri, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        _, fmri_isi = fmri[train_index], fmri[test_index]
        _, design_isi = design[train_index], design[test_index]
        _, stimuli_isi = stimuli[train_index], stimuli[test_index]

        n_points = np.sum(stimuli_isi[:, 1:])
        if n_points == 12 * 4:
            isi = 1.6
            logistic_window = 4
            delay = 1

        elif n_points == 6 * 4:
            isi = 3.2
            logistic_window = 4
            delay = 1

        elif n_points == 4 * 4:
            isi = 4.8
            logistic_window = 4
            delay = 1

        else:
            continue

        labels = np.hstack([[session] * 20 for session in range(4)])
        lplo2 = LeavePLabelOut(labels, p=1)
        for train_id, test_id in lplo2:
            fmri_train, fmri_test = fmri_isi[train_id], fmri_isi[test_id]
            design_train, design_test = (design_isi[train_id],
                                         design_isi[test_id])
            stimuli_train, stimuli_test = (stimuli_isi[train_id],
                                           stimuli_isi[test_id])
            # Feature selection
            fmri_train, fmri_test = de.feature_selection(
                fmri_train, fmri_test, np.argmax(stimuli_train, axis=1))

            # Fit a ridge regression to predict the design matrix
            prediction_test, prediction_train, score = de.fit_ridge(
                fmri_train, fmri_test, design_train, design_test,
                double_prediction=True, extra=fmri_train)

            # Fit a logistic regression for deconvolution
            accuracy = de.logistic_deconvolution(
                prediction_train, prediction_test, stimuli_train,
                stimuli_test, logistic_window, delay=delay)

            scores.append(accuracy)
            subjects.append(subject + 1)
            models.append('logistic deconvolution')
            isis.append(isi)

    print('finished subject ' + str(subject))
