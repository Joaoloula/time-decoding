from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_mrt
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = np.arange(14)
tr = 2.
k = 10000

# GLM parameters
hrf_model = 'spm'
logistic_window = 3
delay = 3

scores, subjects, models = [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_mrt(subject)
    session_id_fmri = [[session] * len(fmri[session])
                       for session in range(len(fmri))]
    design = [de.design_matrix(len(fmri[session]), tr, onsets[session],
                               conditions[session], hrf_model=hrf_model,
                               drift_model='blank')
              for session in range(len(fmri))]

    # Stack the BOLD signals and the design matrices
    fmri = np.vstack(fmri)
    design = np.vstack(design)
    stimuli = np.vstack(stimuli)
    session_id_fmri = np.hstack(session_id_fmri)

    lplo = LeavePLabelOut(session_id_fmri, p=2)
    for train_index, test_index in lplo:
        # Split into train and test sets
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        design_train, design_test = design[train_index], design[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        # Feature selection
        fmri_train, fmri_test = de.feature_selection(
            fmri_train, fmri_test, np.argmax(stimuli_train, axis=1))

        # Fit a ridge regression to predict the design matrix
        prediction_test, prediction_train, score = de.fit_ridge(
            fmri_train, fmri_test, design_train, design_test,
            double_prediction=True, extra=fmri_train)

        # Fit a logistic regression for deconvolution
        accuracy = de.logistic_deconvolution(
            prediction_train, prediction_test, stimuli_train[:, 1:],
            stimuli_test[:, 1:], logistic_window, delay=delay)

        scores.append(accuracy)
        subjects.append(subject + 1)
        models.append('logistic deconvolution')

    print('finished subject ' + str(subject))
