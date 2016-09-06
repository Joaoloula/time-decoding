from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_texture
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = range(7)
k = 10000
tr = 2.4

# GLM parameters
hrf_model = 'spm'
model = 'GLMs'

scores, models, subjects = [], [], []
for subject in subject_list:
    subject_scores = []
    # Read data
    fmri, stimuli, onsets, conditions = read_data_texture(subject)
    betas, regressors = de.glm(fmri, tr, onsets, conditions,
                               hrf_model=hrf_model, model=model)
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

        scores.append(accuracy)
        subjects.append(subject)
        models.append(model)
        print('finished one CV step')

    print('finished subject ' + str(subject))
    print(subject_scores)
