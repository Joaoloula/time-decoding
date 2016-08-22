from sklearn.cross_validation import LeavePLabelOut
from data_reading import read_data_gauthier
import helper_functions as hf
import numpy as np

# Parameters
subject_list = range(11)
time_window = 1
delay = 0
k = 10000
plot = True

# GLM parameters
basis = '3hrf'
mode = 'glm'
logistic_window = 4

all_scores = []
for subject in subject_list:
    subject_scores = []
    # Read data
    fmri, stimuli, runs = read_data_gauthier(subject)
    _, glm_stimuli, glm_runs = read_data_gauthier(subject, glm=True)

    # Mask the sessions uniformly and reshape the data
    fmri = hf.uniform_masking(fmri, high_pass=0.01)

    # Feature selection

    fmri = [hf.feature_selection(fmri[session], np.zeros_like(fmri[session]),
                                 stimuli[session],
                                 np.zeros_like(stimuli[session]), k=k)
            for session in range(len(fmri))]

    # Run the GLM
    beta = [hf.glm(
        fmri[session], glm_stimuli[session][2], glm_stimuli[session][0])
        for session in range(len(fmri))]

    fmri, stimuli, beta = np.vstack(fmri), np.vstack(stimuli), np.vstack(beta)
    onsets, durations, labels = np.vstack(glm_stimuli)

    # Convolve the events
    design = hf.convolve_events(labels, onsets, len(fmri), basis=basis)

    big_lplo = LeavePLabelOut(runs, p=1)
    for train, test in big_lplo:
        fmri_train, fmri_test = fmri[train], fmri[test]
        design_train, design_test = design[train], design[test]
        stimuli_train, stimuli_test = stimuli[train], stimuli[test]

        prediction_test, prediction_train, scores = hf.fit_ridge(
            fmri_train, fmri_test, design_train, design_test,
            time_window=time_window, double_prediction=True, extra=fmri_train)

        accuracy = hf.logistic_deconvolution(
            prediction_train, prediction_test, stimuli_train, stimuli_test,
            logistic_window)

        subject_scores.append(accuracy)

    all_scores.append(subject_scores)

    print('finished subject ' + str(subject))
    print(subject_scores)
