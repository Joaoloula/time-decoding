from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_mrt
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = range(14)
k = 10000
tr = 2.
model = 'GLM'

# GLM parameters
hrf_model = 'spm'

scores, subjects, models = [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_mrt(subject)
    session_id_onset = [[session] * len(onsets[session])
                        for session in range(len(onsets))]
    betas, _ = de.glm(fmri, tr, onsets, hrf_model=hrf_model,
                      drift_model='blank', model=model)

    betas = np.vstack(betas)
    conditions = np.hstack(conditions)
    session_id_onset = np.hstack(session_id_onset)

    lplo = LeavePLabelOut(session_id_onset, p=2)
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

        scores.append(accuracy)
        subjects.append(subject + 1)
        models.append(model)

    print('finished subject ' + str(subject))
