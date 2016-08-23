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
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_onset = np.load('sessions_id_onset.npy')
    glm_onsets = np.load('glm_onsets.npy')
    betas = de.glm(fmri, glm_onsets, hrf_model=hrf_model)

    fmri = np.vstack(fmri)
    betas = np.vstack(betas)

    lplo = LeavePLabelOut(session_id_onset, p=2)
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
