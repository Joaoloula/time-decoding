from sklearn.cross_validation import LeavePLabelOut
from data_reading import read_data_gauthier
import decoding as de
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
    (fmri, stimuli, onsets, conditions, durations, session_id_fmri,
     session_id_onset) = read_data_gauthier(subject)
    betas = de.glm(fmri, onsets, durations, hrf_model)

    # Mask and stack the activation maps and the conditions
    mask = np.where(np.logical_or(
        conditions == 'face', conditions == 'house'))
    conditions = conditions[mask]
    betas = betas[mask]
    masked_sessions = np.array(session_id_onset)[mask]

    lplo = LeavePLabelOut(masked_sessions, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        betas_train, betas_test = betas[train_index], betas[test_index]
        conditions_train, conditions_test = (conditions[train_index],
                                             conditions[test_index])

        # Feature selection
        betas_train, betas_test = de.feature_selection(betas_train, betas_test,
                                                       conditions_train)

        # Fit a logistic regression to score the model
        accuracy = de.glm_scoring(betas_train, betas_test, conditions_train,
                                  conditions_test)

        subject_scores.append(accuracy)

    all_scores.append(subject_scores)

    print('finished subject ' + str(subject))
    print(subject_scores)
