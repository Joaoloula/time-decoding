import numpy as np
import helper_functions as hf

# Parameters
subject_list = [12]
time_window = 3
delay = 2
k = 3000
plot = True
two_classes = True

# Kernel parameters
kernel = 'time_smoothing'
penalty = 2


for subject in subject_list:
    # Read data
    fmri, stimuli = hf.read_data(subject, two_classes=two_classes)

    # Mask the sessions uniformly
    fmri_masked = hf.uniform_masking(fmri, high_pass=0.01)

    # Separate on train and test
    fmri_train, fmri_test = np.vstack(fmri_masked[:-1]), fmri_masked[-1]
    stimuli_train, stimuli_test = np.vstack(stimuli[:-1]), stimuli[-1]

    # Apply time window
    fmri_train, fmri_test, stimuli_train, stimuli_test = hf.apply_time_window(
        fmri_train, fmri_test, stimuli_train, stimuli_test,
        time_window=time_window, delay=delay, k=k)

    prediction, scores = hf.fit_ridge(fmri_train, fmri_test, stimuli_train,
                                      stimuli_test, kernel=kernel,
                                      penalty=penalty, time_window=time_window)
    accuracy = hf.classification_score(prediction, stimuli_test)

    if plot:
        hf.plot(prediction, stimuli_test[:, 1:], scores, accuracy, delay=delay,
                time_window=time_window, two_classes=two_classes, kernel=kernel,
                penalty=penalty)
