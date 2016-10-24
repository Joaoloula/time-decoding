from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_mrt
import time_decoding.decoding as de
import numpy as np

# Parameters
subject_list = [12]
tr = 2.
model = 'GLM'

# GLM parameters
hrf_model = 'spm'

for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_mrt(subject)
    session_id_onset = [[session] * len(onsets[session])
                        for session in range(len(onsets))]
    betas, reg = de.glm(fmri, tr, onsets, hrf_model=hrf_model, model=model)

    break
    betas = np.vstack(betas)
    session_id_onset = np.hstack(session_id_onset)
    conditions = np.hstack(conditions)

    junk_mask = np.where(conditions != 'ju')
    conditions = conditions[junk_mask]
    betas = betas[junk_mask]
    session_id_onset = session_id_onset[junk_mask]

    lplo = LeavePLabelOut(session_id_onset, p=1)
    for train_index, test_index in lplo:
        # Split into train and test sets
        betas_train, betas_test = betas[train_index], betas[test_index]
        conditions_train, conditions_test = (conditions[train_index],
                                             conditions[test_index])

        # Fit a logistic regression to score the model
        accuracy = de.glm_scoring(betas_train, betas_test, conditions_train,
                                  conditions_test)

    print('finished subject ' + str(subject))
