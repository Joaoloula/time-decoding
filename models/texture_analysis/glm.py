from sklearn.cross_validation import LeavePLabelOut
import helper_functions as hf
import numpy as np

subject_names = ['pf120155']
n_subjects = 1
time_window = 1
delay = 0
k = 10000

# GLM Parameters
basis = 'hrf'
mode = 'glm'

scores = []
for subject in range(n_subjects):
    _, anova_stimuli = hf.read_data(subject)
    _, glm_stimuli = hf.read_data(subject, glm=True)
    fmri = np.load(subject_names[subject] + '_fmri_masked.npy')

    # Only get first run (TODO change this)
    anova_stimuli = anova_stimuli[0]
    glm_stimuli = glm_stimuli[0]
    fmri = fmri[:fmri.shape[0] / 2]

    # Apply time window
    fmri, _, anova_stimuli, __ = hf.apply_time_window(
        fmri, np.zeros_like(fmri), anova_stimuli, np.zeros_like(anova_stimuli),
        delay=delay, k=k, time_window=time_window)

    # Fit GLM
    hrf, beta, labels, onsets = hf.glm(fmri, glm_stimuli, basis=basis,
                                       mode=mode)

    # Split data into train and test
    runs = np.repeat(range(6), 108)
    lplo = LeavePLabelOut(runs, p=1)
    for train_index, test_index in lplo:
        train_mask = np.intersect1d(train_index, np.where(glm_stimuli != '0'))
        test_mask = np.intersect1d(test_index, np.where(glm_stimuli != '0'))
        labels_train, labels_test = labels[train_mask], labels[test_mask]
        beta_train, beta_test = beta[train_mask], beta[test_mask]

        scores.append(
            hf.glm_scoring(beta_train, beta_test, labels_train, labels_test))

        print('finished subject ' + str(subject))
