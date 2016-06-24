# Performs multinomial logistic regression on activation data created from the
# Haxby dataset, using a custom time window
# Accuracy: 0.89 with 8 categories
from hrf_estimation.savitzky_golay import savgol_filter
from hrf_estimation import rank_one
from sklearn.cross_validation import LeavePLabelOut
from sklearn import metrics
from nilearn import datasets
from helper_functions import read_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PARAMETERS
n_scans = 1452
n_sessions = 12
n_c = 5  # number of Cs to use in logistic regression CV
n_jobs = 2  # number of jobs to use in logistic regression CV
n_subjects = 6
n_basis = 5  # Number of points to use in the hrf estimation basis
plot_subject = 9  # ID of the subject to plot
time_window = 5

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# MODEL

# Initialize mean score and score counter
categories_r2_scores = np.zeros(8)

sns.set_style('whitegrid')
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = read_data(subject, haxby_dataset,
                                                      n_scans)
    # Apply a time window of 'time_window'
    fmri_window = np.asarray([fmri[scan: scan + time_window]
                              for scan in range(len(fmri) - time_window)])
    fmri_window = fmri_window.reshape((n_scans - time_window),
                                      time_window * np.shape(fmri)[1])
    series = series[: -time_window]
    sessions_id = sessions_id[: -time_window]

    # Convert 'series' to one-hot encoding
    one_hot_series = np.zeros((n_scans - time_window, len(categories)))
    for scan in range(n_scans - time_window):
        one_hot_series[scan][series[scan]] = 1
    one_hot_series = one_hot_series[:, 1:]  # Eliminate 'rest' from events

    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        one_hot_series_train = one_hot_series[train_index]
        one_hot_series_test = one_hot_series[test_index]
        series_train = series[train_index]
        series_test = series[test_index]
        fmri_train = fmri[train_index]
        fmri_test = fmri[test_index]
        fmri_window_train = fmri_window[train_index]
        fmri_window_test = fmri_window[test_index]

    # Predict HRF and activation signals
    # Introduce a stop point st. the number of voxels is divisible by n_basis
    stop = len(fmri_train[0]) - len(fmri_train[0]) % n_basis
    hrf, beta = rank_one(fmri_train[:, : stop], one_hot_series_train,
                         n_basis=n_basis, basis='fir')
    hrf_beta = [np.kron(beta[:, k], hrf[:, k].T) for k in range(8)]
    hrf_beta = np.asarray(hrf_beta).T
    prediction = np.dot(fmri_test[:, : stop], hrf_beta)
    prediction = [savgol_filter(prediction[:, k], 7, 3) for k in range(8)]
    prediction = np.asarray(prediction).T

    # SCORE
    categories_r2_scores += metrics.r2_score(one_hot_series_test, prediction,
                                             multioutput='raw_values')

categories_r2_scores /= n_subjects

plt.bar(range(8), categories_r2_scores, tick_label=categories[1:])
plt.title('R2 scores for r1glm model, n_basis = ' + str(n_basis))

plt.show()
