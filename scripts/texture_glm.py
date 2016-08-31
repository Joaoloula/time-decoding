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
    betas, regressors = de.glm(fmri, onsets, conditions, hrf_model=hrf_model)
    session_id_onset = np.array([[session] * len(betas[0])
                                 for session in range(len(onsets))]).ravel()
    betas = np.vstack(betas)
    conditions = np.hstack(regressors)
    conditions_ = np.array([condition[:2] for condition in conditions])
    lplo = LeavePLabelOut(session_id_onset, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        betas_train, betas_test = betas[train_index], betas[test_index]
        conditions_train, conditions_test = (conditions_[train_index],
                                             conditions_[test_index])
        # Mask to remove '0' category
        train_mask = np.array([
            ct in ['01', '09', '12', '13', '14', '25']
            for ct in conditions_train])
        test_mask = np.array([
            ct in ['01', '09', '12', '13', '14', '25']
            for ct in conditions_test])
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
