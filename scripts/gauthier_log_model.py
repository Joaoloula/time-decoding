from sklearn.cross_validation import LeavePLabelOut
from time_decoding.data_reading import read_data_gauthier
import time_decoding.decoding as de
import pandas as pd
import numpy as np

# Parameters
subject_list = range(11)
tr = 1.5
k = 10000
n_tests = 20

# GLM parameters
hrf_model = 'spm'
logistic_window = 1
delay = 4

scores, subjects, models, isis = [], [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_fmri = [[session] * len(fmri[session])
                       for session in range(len(fmri))]
    session_id_onset = np.load('sessions_id_onset.npy')
    """
    session_id_fmri = [[19.2 / len(onsets[session])] * len(fmri[session])
                       for session in range(len(onsets))]
    """
    durations = [[19.2 / len(onsets[session])] * len(onsets[session])
                 for session in range(len(onsets))]
    design = [de.design_matrix(len(fmri[session]), tr, onsets[session],
                               conditions[session],
                               durations=None,
                               hrf_model=hrf_model, drift_model='blank')
              for session in range(len(fmri))]

    # Stack the BOLD signals and the design matrices
    fmri = np.vstack(fmri)
    design = np.vstack(design)
    stimuli = np.vstack(stimuli)
    session_id_fmri = np.hstack(session_id_fmri)

    blocks = [0, 0]
    lplo = LeavePLabelOut(session_id_fmri, p=2)
    for train_index, test_index in lplo:
        if blocks[1] < 11:
            blocks[1] += 1
        else:
            blocks[0] += 1
            blocks[1] = blocks[0] + 1

        # Split into train and test sets
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        design_train, design_test = design[train_index], design[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        n_points = np.sum(stimuli_test[:, 1:])
        if n_points == 12 * 2:
            isi = 1.6

        elif n_points == 6 * 2:
            isi = 3.2

        elif n_points == 4 * 2:
            isi = 4.8

        else:
            continue

        # Feature selection
        fmri_train, fmri_test = de.feature_selection(
            fmri_train, fmri_test, np.argmax(stimuli_train, axis=1), k=k)

        # Fit a ridge regression to predict the design matrix
        prediction_test, prediction_train, score = de.fit_ridge(
            fmri_train, fmri_test, design_train, design_test,
            double_prediction=True, extra=fmri_train)

        # Fit a logistic regression for deconvolution
        accuracy = de.logistic_deconvolution(
            prediction_train, prediction_test, stimuli_train,
            stimuli_test, logistic_window, delay=delay, balance=True,
            n_tests=n_tests, block=blocks, session_id_onset=session_id_onset)

        scores.append(accuracy)
        subjects.append(subject + 1)
        models.append('logistic deconvolution')
        isis.append(isi)

        print('Score for isi of {isi}: {score}'.format(isi=isi,
                                                       score=accuracy))

    print('finished subject ' + str(subject))

dict = {}
dict['subject'] = subjects
dict['model'] = models
dict['accuracy'] = scores
dict['isi'] = isis

data = pd.DataFrame(dict)
print(np.mean(data.loc[data['isi'] == 1.6]['accuracy']))
print(np.mean(data.loc[data['isi'] == 3.2]['accuracy']))
print(np.mean(data.loc[data['isi'] == 4.8]['accuracy']))
