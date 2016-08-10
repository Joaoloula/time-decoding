from sklearn.cross_validation import LeavePLabelOut
import numpy as np
import helper_functions as hf

# Parameters
subject_list = [11]
time_window = 1
delay = 0
k = 3000
plot = False
two_classes = True

# Kernel parameters
kernel = None
penalty = 1

subject_scores = []
for subject in subject_list:
    # Read data
    fmri, stimuli, runs = hf.read_data(subject, two_classes=two_classes)

    # Mask the sessions uniformly
    fmri_masked = hf.uniform_masking(fmri, high_pass=0.01)

    # Separate on train and test
    fmri, stimuli = np.vstack(fmri_masked), np.vstack(stimuli)
    lplo = LeavePLabelOut(runs, p=1)
    for train_index, test_index in lplo:
        fmri_train, fmri_test = fmri[train_index], fmri[test_index]
        stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

        # Apply time window
        fmri_train, fmri_test, stimuli_train, stimuli_test = hf.apply_time_window(
            fmri_train, fmri_test, stimuli_train, stimuli_test,
            time_window=time_window, delay=delay, k=k)

        prediction, scores = hf.fit_ridge(fmri_train, fmri_test, stimuli_train,
                                          stimuli_test, kernel=kernel,
                                          penalty=penalty,
                                          time_window=time_window)
        accuracy = hf.classification_score(prediction, stimuli_test)
        subject_scores.append(accuracy)

        if plot:
            hf.plot(prediction, stimuli_test[:, 1:], scores, accuracy,
                    delay=delay, time_window=time_window,
                    two_classes=two_classes, kernel=kernel,
                    penalty=penalty)

        print('finished subject ' + str(subject))
        break  # Only one CV run per subject for fast prototyping
