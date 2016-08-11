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
deconvolution = 'rich'

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
    for train_index, test_index in lplo:
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        design_train, design_test = design[train_index], design[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        prediction, scores = hf.fit_ridge(fmri_train, fmri_test, design_train,
                                          design_test, time_window=time_window)

        if deconvolution == 'poor':
            prediction, stimuli_test = prediction[3:], stimuli_test[: -3]

        elif deconvolution == 'rich':
            prediction = hf.deconvolution(prediction)

        accuracy = hf.classification_score(prediction, stimuli_test, mode='glm')
        subject_scores.append(accuracy)

        if plot:
            hf.plot(prediction[:, 1:], design_test[:, 1:], scores[1:], accuracy,
                    delay=delay, time_window=time_window,
                    two_classes=two_classes)

        print('finished subject ' + str(subject))
        break  # Only one CV run per subject for fast prototyping
