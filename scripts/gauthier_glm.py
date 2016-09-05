from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_gauthier
import time_decoding as de
import numpy as np

# Parameters
subject_list = range(11)
k = 10000
tr = 1.5
model = 'GLMs'

# GLM parameters
hrf_model = 'spm'

scores = []
subjects = []
models = []
isis = []
for subject in subject_list:
    subject_scores = []
    # Read data
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_onset = np.load('sessions_id_onset.npy')
    betas, _ = de.glm(fmri, tr, onsets, hrf_model=hrf_model,
                      drift_model='blank', model=model)

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

        n_points = len(conditions_test)
        if n_points == 12:
            isi = 1.6

        elif n_points == 6:
            isi = 3.2

        if n_points == 4:
            isi = 4.8

        scores.append(accuracy)
        subjects.append(subject + 1)
        models.append(model)
        isis.append(isi)

    print('finished subject ' + str(subject))
