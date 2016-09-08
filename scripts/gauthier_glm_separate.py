from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_gauthier
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = range(11)
tr = 1.5
k = 10000
model = 'GLM'

# GLM parameters
hrf_model = 'spm'

scores, subjects, models, isis = [], [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_onset = np.load('sessions_id_onset.npy')
    session_id_onset = [[19.2 / len(onsets[session])] * len(onsets[session])
                        for session in range(len(onsets))]
    betas, reg = de.glm(fmri, tr, onsets, hrf_model=hrf_model,
                        drift_model='blank', model=model)

    betas = np.vstack(betas)
    conditions = np.hstack(conditions)
    session_id_onset = np.hstack(session_id_onset)

    lplo = LeavePLabelOut(session_id_onset, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        betas_isi = betas[test_index]
        conditions_isi = conditions[test_index]

        n_points = len(conditions_isi)
        if n_points == 12 * 4:
            isi = 1.6

        elif n_points == 6 * 4:
            isi = 3.2

        elif n_points == 4 * 4:
            isi = 4.8

        else:
            continue

        labels = np.hstack([[session] * int(19.2/isi) for session in range(4)])
        lplo2 = LeavePLabelOut(labels, p=2)
        for train_id, test_id in lplo2:
            betas_train, betas_test = betas_isi[train_id], betas_isi[test_id]
            conditions_train, conditions_test = (conditions_isi[train_id],
                                                 conditions_isi[test_id])

            # Feature selection
            betas_train, betas_test = de.feature_selection(
                betas_train, betas_test, conditions_train, k=k)

            # Fit a logistic regression to score the model
            accuracy = de.glm_scoring(betas_train, betas_test, conditions_train,
                                      conditions_test)

            scores.append(accuracy)
            subjects.append(subject + 1)
            models.append(model)
            isis.append(isi)

    print('finished subject ' + str(subject))
