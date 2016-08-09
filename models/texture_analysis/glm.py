from sklearn.cross_validation import LeavePLabelOut
import helper_functions as hf
import numpy as np

subject_names = ['pf120155']
n_subjects = 1
time_window = 1
delay = 1
k = 10000

# GLM Parameters
basis = 'hrf'
mode = 'glm'

for subject in range(n_subjects):
    _, anova_stimuli = hf.read_data(subject)
    _, glm_stimuli = hf.read_data(subject, glm=True)
    fmri = np.load(subject_names[subject] + '_fmri_masked.npy')

    # Only get first run
    anova_stimuli = anova_stimuli[0]
    glm_stimuli = glm_stimuli[0].ravel()
    new_glm_stimuli = []
    for n, element in enumerate(glm_stimuli):
        new_glm_stimuli.append(element)
        if n % 2 == 0:
            new_glm_stimuli.append('0')
    glm_stimuli = new_glm_stimuli
    fmri = fmri[:fmri.shape[0] / 2]

    # Apply time window
    fmri, _, anova_stimuli, __ = hf.apply_time_window(
        fmri, np.zeros_like(fmri), anova_stimuli, np.zeros_like(anova_stimuli),
        delay=delay, k=k, time_window=time_window)

    # Fit GLM
    hrf, betas = hf.glm(fmri, glm_stimuli, basis=basis, mode=mode)

    # Split data into train and test
    lplo = LeavePLabelOut(runs, p=1)
    for train_index, test_index in lplo:
        train_mask = np.intersect1d(train_index, np.where())
        test_mask = np.intersect1d(test_index, np.where())

    anova_split, glm_split = (
        np.array([fmri.shape[0], glm_stimuli.shape[0]]) * (5. / 6))
    fmri_train, fmri_test = fmri[: anova_split], fmri[anova_split:]
    anova_stimuli_train, anova_stimuli_test = (
        anova_stimuli[: anova_split], anova_stimuli[anova_split:])
    glm_stimuli_train, glm_stimuli_test = (
        glm_stimuli[: glm_split], glm_stimuli[glm_split:])

    # Get model prediction
    prediction = np.dot(fmri_test, np.multiply(hrf[0], betas).T)
    score = hf.classification_score(prediction, anova_stimuli_test[1:])
    print(score)
