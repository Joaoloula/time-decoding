from sklearn.cluster import MiniBatchKMeans
import helper_functions as hf
import numpy as np

# Parameters
n_subjects = 1
time_window = 5
two_classes = True

for subject in range(n_subjects):
    # Read data
    fmri, stimuli = hf.read_data(subject, two_classes=two_classes)

    # Mask the sessions uniformly
    fmri_masked = hf.uniform_masking(fmri, high_pass=0.01)

    # Separate on train and test
    fmri_train, fmri_test, = np.vstack(fmri_masked[:-1]), fmri_masked[-1]
    stimuli_train, stimuli_test, = np.vstack(stimuli[:-1]), stimuli[-1]

    # Apply time window
    fmri_train, stimuli_train = hf.apply_time_window(fmri_train, stimuli_train,
                                                     time_window=time_window)
    fmri_test, stimuli_test = hf.apply_time_window(fmri_test, stimuli_test,
                                                   time_window=time_window)

    # Reshape to go from time vector to time matrices
    fmri_train, fmri_test = np.array(fmri_train), np.array(fmri_test)
    fmri_train = fmri_train.reshape(-1, time_window)

    fmri_test = fmri_test.reshape(-1, time_window)

    kmeans = MiniBatchKMeans(n_clusters=100)
    kmeans.fit(fmri_train)
    centroids = kmeans.cluster_centers
