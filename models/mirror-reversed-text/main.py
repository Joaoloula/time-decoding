import numpy as np
import helper_functions as hf

# Parameters
n_subjects = 1
time_window = 2
plot = True
two_classes = True

for subject in range(n_subjects):
    # Read data
    fmri, stimuli = hf.read_data(subject, two_classes=two_classes)

    # Mask the sessions uniformly
    fmri_masked = hf.uniform_masking(fmri, high_pass=0.01)

    # Separate on train and test
    fmri_train, fmri_test = np.vstack(fmri_masked[:-1]), fmri_masked[-1]
    stimuli_train, stimuli_test = np.vstack(stimuli[:-1]), stimuli[-1]

    # Apply time window
    fmri_train, stimuli_train = hf.apply_time_window(fmri_train, stimuli_train,
                                                     time_window=time_window)
    fmri_test, stimuli_test = hf.apply_time_window(fmri_test, stimuli_test,
                                                   time_window=time_window)

    prediction, scores = hf.fit_logistic_regression(fmri_train,
        fmri_test, stimuli_train, stimuli_test, k=10000)
    print(scores)

    if plot:
        hf.plot(prediction, stimuli_test[:, 1:], scores,
                time_window=time_window, two_classes=two_classes)
