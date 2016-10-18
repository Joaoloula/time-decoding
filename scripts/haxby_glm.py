from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_haxby
import time_decoding.decoding as de
import pandas as pd
import numpy as np

# Parameters
subject_list = np.arange(1, 7)
k = 10000
tr = 2.5
model = 'GLMs'

# GLM parameters
hrf_model = 'spm'

scores, subjects, models = [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_haxby(subject)
    session_id_onset = [[session] * len(onsets[session])
                        for session in range(len(onsets))]
    durations = [[24] * 8] * len(fmri)
    betas, reg = de.glm(fmri, tr, onsets, durations=durations,
                        hrf_model=hrf_model, model=model)

    betas = np.vstack(betas)
    session_id_onset = np.hstack(session_id_onset)
    conditions = np.hstack(conditions)

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
        subjects.append(subject)
        models.append(model)

    print('finished subject ' + str(subject))

dict = {}
dict['subject'] = subjects
dict['model'] = models
dict['accuracy'] = scores

data = pd.DataFrame(dict)
print(np.mean(data['accuracy']))
