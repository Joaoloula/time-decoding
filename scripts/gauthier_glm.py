from sklearn.cross_validation import LeavePLabelOut
from time-decoding.data_reading import read_data_gauthier
import time-decoding.decoding as de
import numpy as np

# Parameters
subject_list = range(11)
k = 10000

# GLM parameters
hrf_model = 'spm'

all_scores = []
for subject in subject_list:
    subject_scores = []
    # Read data
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_onset = np.load('sessions_id_onset.npy')
    betas = de.glm(fmri, onsets, hrf_model=hrf_model, drift_model='blank')

    betas = np.vstack(betas)
    conditions = np.hstack(conditions)

    lplo = LeavePLabelOut(session_id_onset, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        betas_train, betas_test = betas[train_index], betas[test_index]
        conditions_train, conditions_test = (conditions[train_index],
                                             conditions[test_index])

        # Feature selection
        betas_train, betas_test = de.feature_selection(betas_train, betas_test,
                                                       conditions_train, k=k)

        # Fit a logistic regression to score the model
        accuracy = de.glm_scoring(betas_train, betas_test, conditions_train,
                                  conditions_test)

        subject_scores.append(accuracy)

    all_scores.append(subject_scores)

    print('finished subject ' + str(subject))
    print(subject_scores)
