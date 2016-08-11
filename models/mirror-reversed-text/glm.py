from sklearn.cross_validation import LeavePLabelOut
import helper_functions as hf
import numpy as np

# Parameters
subject_list = [11]
time_window = 1
delay = 0
k = 10000
two_classes = True

# GLM parameters
basis = 'hrf'
mode = 'glm'

scores = []
for subject in subject_list:
    # Read data
    fmri, stimuli, _ = hf.read_data(subject, two_classes=two_classes)
    _, glm_stimuli, runs = hf.read_data(subject, glm=True)

    # Mask the sessions uniformly and reshape the data
    fmri = hf.uniform_masking(fmri, high_pass=0.01)
    fmri, stimuli = np.vstack(fmri), np.vstack(stimuli)

    # Feature extraction

    fmri, _, stimuli, __ = hf.apply_time_window(
        fmri, np.zeros_like(fmri), stimuli, np.zeros_like(stimuli), delay=delay,
        time_window=time_window, k=k)

    # Run the GLM
    hrf, beta, labels, onsets = hf.glm(fmri, glm_stimuli, runs, basis=basis,
                                       mode=mode)

    # Separate on train and test (leave one session out, no 'junk' class)
    lplo = LeavePLabelOut(runs, p=1)
    for train_index, test_index in lplo:
        train_mask = np.intersect1d(train_index, np.where(labels != 'j'))
        test_mask = np.intersect1d(test_index, np.where(labels != 'j'))
        beta_train, beta_test = beta[train_mask], beta[test_mask]
        labels_train, labels_test = labels[train_mask], labels[test_mask]

        # Fit and score logistic regression
        scores.append(
            hf.glm_scoring(beta_train, beta_test, labels_train, labels_test))

        print('finished subject ' + str(subject))

        break  # Only do one CV run for fast prototyping
