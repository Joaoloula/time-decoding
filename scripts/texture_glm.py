from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_texture
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = range(1)
k = 10000

# GLM parameters
hrf_model = 'spm'

all_scores = []
for subject in subject_list:
    subject_scores = []
    # Read data
    fmri, stimuli, onsets, conditions = read_data_texture(subject)
    session_id_onset = np.array([[session] * len(onsets[session])
                                 for session in range(len(onsets))]).ravel()
    betas = de.glm(fmri, onsets, hrf_model=hrf_model)

    betas = np.vstack(betas)
    conditions = np.hstack(conditions)

    lplo = LeavePLabelOut(session_id_onset, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        betas_train, betas_test = betas[train_index], betas[test_index]
        conditions_train, conditions_test = (conditions[train_index],
                                             conditions[test_index])
        # Mask to remove '0' category
        train_mask = conditions_train != '0'
        test_mask = conditions_test != '0'
        betas_train, conditions_train = [betas_train[train_mask],
                                         conditions_train[train_mask]]
        betas_test, conditions_test = [betas_test[test_mask],
                                       conditions_test[test_mask]]

        # Feature selection
        betas_train, betas_test = de.feature_selection(betas_train, betas_test,
                                                       conditions_train, k=k)

        # Fit a logistic regression to score the model
        accuracy = de.glm_scoring(betas_train, betas_test, conditions_train,
                                  conditions_test)

        subject_scores.append(accuracy)
        print('finished one CV step')

    all_scores.append(subject_scores)

    print('finished subject ' + str(subject))
    print(subject_scores)
