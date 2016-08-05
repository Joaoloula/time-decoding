import numpy as np
import helper_functions as hf

# Parameters
subject_list = [12]
time_window = 1
delay = 0
k = 10000
two_classes = True

# GLM parameters
basis = 'hrf'
mode = 'glm'

for subject in subject_list:
    # Read data
    fmri, stimuli = hf.read_data(subject, two_classes=two_classes)
    _, glm_stimuli = hf.read_data(subject, glm=True)

    # Mask the sessions uniformly
    fmri_masked = hf.uniform_masking(fmri, high_pass=0.01)

    # Separate on train and test
    fmri_train, fmri_test = np.vstack(fmri_masked[:-1]), fmri_masked[-1]
    stimuli_train, stimuli_test = np.vstack(stimuli[:-1]), stimuli[-1]
    glm_stimuli_train, glm_stimuli_test = glm_stimuli[:-1], glm_stimuli[-1]

    # Apply time window
    fmri_train, fmri_test, stimuli_train, stimuli_test = hf.apply_time_window(
        fmri_train, fmri_test, stimuli_train, stimuli_test,
        time_window=time_window, delay=delay, k=k)

    hrf, betas = hf.glm(fmri_train, glm_stimuli_train, basis=basis, mode=mode)
