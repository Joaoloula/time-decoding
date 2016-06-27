import numpy as np
from sklearn.cross_validation import LeavePLabelOut
from helper_functions import read_data
from hrf_estimation import rank_one
from nilearn import datasets
from sklearn import metrics

# Parametes
n_basis = 5
n_scans = 1452
subject = 0
basis = 'fir'  # Type of basis to use for rank one estimation

# Fetch dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=1)

# Direct model
X, series, sessions_id, categories = read_data(subject, haxby_dataset, n_scans)
stop = len(X[0]) - len(X[0]) % n_basis
real_hrf = np.random.rand(n_basis, len(X))
real_beta = np.random.rand((len(X[0]) / n_basis), len(X))
real_hrf_beta = [np.kron(real_beta[:, k], real_hrf[:, k].T) for k in range(8)]
real_hrf_beta = np.asarray(real_hrf_beta).T
y = np.dot(X[:, :stop], real_hrf_beta)

# Train and test
lplo = LeavePLabelOut(sessions_id, p=2)
for train_index, test_index in lplo:
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    # Prediction
    hrf, beta = rank_one(X_train[:, : stop], y_train, n_basis=n_basis,
                         basis=basis)
    pred_hrf_beta = [np.kron(beta[:, k], hrf[:, k].T) for k in range(8)]
    pred_hrf_beta = np.asarray(pred_hrf_beta).T
    y_pred = np.dot(X_test[:, :stop], pred_hrf_beta)

    score = metrics.r2_score(y_test, y_pred)
    print(score)

    break  # Only run once for fast prototyping
