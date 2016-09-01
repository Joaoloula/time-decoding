from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_mrt
from sklearn import linear_model
import time_decoding.decoding as de
import numpy as np


def logistic_deconvolution(estimation_train, estimation_test, stimuli_train,
                           stimuli_test, logistic_window):
    """ Learn a deconvolution filter for classification given a time window
    using logistic regression """
    log = linear_model.LogisticRegressionCV()
    cats_train = [
        estimation_train[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_train) - logistic_window + 1)]
    cats_test = [
        estimation_test[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_test) - logistic_window + 1)]

    train_mask = np.sum(
        stimuli_train[:len(cats_train), 1:], axis=1).astype(bool)
    test_mask = np.sum(
        stimuli_test[:len(cats_test), 1:], axis=1).astype(bool)

    stimuli_train, stimuli_test = (
        np.argmax(stimuli_train[:len(cats_train)][train_mask], axis=1),
        np.argmax(stimuli_test[:len(cats_test)][test_mask], axis=1))
    cats_train, cats_test = (
        np.array(cats_train)[train_mask], np.array(cats_test)[test_mask])

    log.fit(cats_train, stimuli_train)
    accuracy = log.score(cats_test, stimuli_test)
    probas = log.predict_proba(cats_test)

    return accuracy, probas


# Parameters
subject_list = [12]
tr = 2.
k = 10000

# GLM parameters
hrf_model = 'spm'
logistic_window = 4

all_scores = []
all_predictions = []
all_probas = []
for subject in subject_list:
    subject_scores = []
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
        prediction_test, prediction_train, score = de.fit_ridge(
            fmri_train, fmri_test, design_train, design_test,
            double_prediction=True, extra=fmri_train)

        all_predictions.append([design_test, prediction_test, score])

        # Fit a logistic regression for deconvolution
        accuracy, probas = logistic_deconvolution(
            prediction_train, prediction_test, stimuli_train[:, 1:],
            stimuli_test[:, 1:], logistic_window)

        subject_scores.append(accuracy)
        all_probas.append(probas)

        break

    all_scores.append(subject_scores)

    print('finished subject ' + str(subject))
    print(subject_scores)
