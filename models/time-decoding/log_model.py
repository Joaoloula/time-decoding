from sklearn.cross_validation import LeavePLabelOut
from data_reading import read_data_gauthier
import decoding as de
import numpy as np

# Parameters
subject_list = range(11)
tr = 1.5
k = 10000

# GLM parameters
hrf_model = 'spm'
logistic_window = 4

all_scores = []
for subject in subject_list:
    subject_scores = []
    # Read data
    (fmri, stimuli, onsets, conditions, durations, session_id_fmri,
     session_id_onset) = read_data_gauthier(subject)
    design = [de.design_matrix(
        len(fmri[session]), tr, onsets[session], conditions[session],
        durations[session], hrf_model) for session in range(len(fmri))]

    # Stack the BOLD signals and the design matrices
    fmri = np.vstack(fmri)
    design = np.vstack(design)
    stimuli = np.vstack(stimuli)
    session_id_fmri = np.hstack(session_id_fmri)

    lplo = LeavePLabelOut(session_id_fmri, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        design_train, design_test = design[train_index], design[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        # Feature selection
        fmri_train, fmri_test = de.feature_selection(
            fmri_train, fmri_test, np.argmax(stimuli_train, axis=1))

        # Fit a ridge regression to predict the design matrix
        prediction_train, prediction_test, score = de.fit_ridge(
            fmri_train, fmri_test, stimuli_train, stimuli_test,
            double_prediction=True, extra=fmri_train)

        # Fit a logistic regression for deconvolution
        accuracy = de.logistic_deconvolution(
            prediction_train, prediction_test, stimuli_train[:, :4],
            stimuli_test[:, :4], logistic_window)

        subject_scores.append(accuracy)

    all_scores.append(subject_scores)

    print('finished subject ' + str(subject))
    print(subject_scores)
