from sklearn import neighbors
import helper_functions as hf
import numpy as np

n_subjects = 1
time_window = 3
two_classes = True

for subject in range(n_subjects):
    # Read data
    fmri, stimuli = hf.read_data(subject, two_classes=two_classes)

    # Mask the sessions uniformely
    fmri_masked = hf.uniform_masking(fmri)
    fmri_train, fmri_test = np.vstack(fmri_masked[: -1]), fmri_masked[-1]
    stimuli_train, stimuli_test = np.vstack(stimuli[: -1]), stimuli[-1]

    # Apply time window
    fmri_train, stimuli_train = hf.apply_time_window(fmri_train, stimuli_train,
                                                     time_window=time_window)
    fmri_test, stimuli_test = hf.apply_time_window(fmri_test, stimuli_test,
                                                   time_window=time_window)

    # Restrain analysis to points where a stimulus is present
    stimuli_train_mask = np.sum(stimuli_train, axis=1).astype(bool)
    stimuli_test_mask = np.sum(stimuli_test, axis=1).astype(bool)
    fmri_train, stimuli_train = (np.array(fmri_train)[stimuli_train_mask],
                                 np.array(stimuli_train)[stimuli_train_mask])
    fmri_test, stimuli_test = (np.array(fmri_test)[stimuli_test_mask],
                               np.array(stimuli_test)[stimuli_test_mask])

    score = hf.fit_ridge_classifier(fmri_train, fmri_test, stimuli_train,
                                    stimuli_test)

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(fmri_train, stimuli_train)
    score_knn = knn.score(fmri_test, stimuli_test)

    print(score)
    print(score_knn)

    hf.embed(fmri_train, stimuli_train)
