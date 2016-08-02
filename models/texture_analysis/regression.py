import helper_functions as hf
import numpy as np

subject_names = ['pf120155']
n_subjects = 1
time_window = 3
delay = 0
k = 3000

# Kernel parameters
kernel = 'voxel_weighing'
penalty = 1.

for subject in range(n_subjects):
    fmri, stimuli = hf.read_data(subject)
    stimuli = np.vstack(stimuli)
    # Load masked fmri from file for faster testing
    fmri = np.load(subject_names[subject] + '_fmri_masked.npy')

    # Split data into train and test
    split = (fmri.shape[0]/12) * 10
    fmri_train, fmri_test = fmri[: split], fmri[split:]
    stimuli_train, stimuli_test = stimuli[: split], stimuli[split:]

    # Apply time window
    fmri_train, fmri_test, stimuli_train, stimuli_test = hf.apply_time_window(
        fmri_train, fmri_test, stimuli_train, stimuli_test, delay=delay, k=k,
        time_window=time_window)

    prediction, score = hf.fit_ridge(fmri_train, fmri_test, stimuli_train,
                                     stimuli_test, kernel=kernel,
                                     penalty=penalty)

    accuracy = hf.classification_score(prediction, stimuli_test)

    hf.plot(prediction, stimuli_test, score, accuracy, delay=delay,
            time_window=time_window)
