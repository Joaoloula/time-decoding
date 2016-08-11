from sklearn.cross_validation import LeavePLabelOut
import helper_functions as hf
import numpy as np

# Parameters
subject_list = [11]
time_window = 1
delay = 0
k = 10000
two_classes = True
plot = True

# GLM parameters
basis = 'hrf'
mode = 'glm'
logistic_window = 3

subject_scores = []
for subject in subject_list:
    # Read data
    fmri, stimuli, runs = hf.read_data(subject, two_classes=two_classes)
    _, glm_stimuli, glm_runs = hf.read_data(subject, glm=True)

    # Mask the sessions uniformly and reshape the data
    fmri = hf.uniform_masking(fmri, high_pass=0.01)
    fmri, stimuli = np.vstack(fmri), np.vstack(stimuli)

    # Feature extraction

    fmri, _, stimuli, __ = hf.apply_time_window(
        fmri, np.zeros_like(fmri), stimuli, np.zeros_like(stimuli), delay=delay,
        time_window=time_window, k=k)
    fmri = np.array(fmri)

    # Run the GLM
    hrf, beta, labels, onsets = hf.glm(fmri, glm_stimuli, glm_runs, basis=basis,
                                       mode=mode)

    # Convolve the events
    design = hf.convolve_events(labels, onsets, len(fmri), basis=basis)

    lplo = LeavePLabelOut(runs, p=1)
    runs = np.array(runs)
    first_train = np.where(np.logical_or(runs == 0, runs == 1))[0]
    second_train = np.where(np.logical_or(runs == 2, runs == 3))[0]
    second_test = np.where(np.logical_or(runs == 4, runs == 5))[0]

    fmri_train, fmri_test = fmri[first_train], fmri[second_train]
    fmri_second_test = fmri[second_test]
    design_train, design_test = design[first_train], design[second_train]
    stimuli_train, stimuli_test = stimuli[second_train], stimuli[second_test]

    prediction_train, prediction_test, scores = hf.fit_ridge(
        fmri_train, fmri_test, design_train, design_test,
        time_window=time_window, double_prediction=True, extra=fmri_second_test)

    accuracy = hf.logistic_deconvolution(
        prediction_train, prediction_test, stimuli_train, stimuli_test,
        logistic_window)

    subject_scores.append(accuracy)

    print('finished subject ' + str(subject))
