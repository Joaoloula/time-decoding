from nistats.design_matrix import make_design_matrix
from nilearn.input_data import NiftiMasker
from sklearn import linear_model
from sklearn import metrics
import pandas as pd
import numpy as np


def read_data(subject, haxby_dataset):
    """Generates data for a given haxby dataset subject.

    Parameters
    ----------

    subject: int from 0 to 5
        id of the subject whose data should be retrieved

    haxby_dataset: dictionary,
        nilearn-generated dictionary containing filepaths for the dataset

    Returns
    -------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    series: numpy array of shape [n_scans]
        time series of the stimuli, coded as ints from 0 to 8

    sessions_id: numpy array of shape [n_scans]
        identifications of the sessions each scan belongs to (12 in total)

    categories: numpy array of shape [n_categories]
        list of all the categories, in the order they are coded in series
    """
    # Read labels
    labels = np.recfromcsv(
        haxby_dataset.session_target[subject], delimiter=" ")
    sessions_id = labels['chunks']
    target = labels['labels']
    categories = np.unique(target)
    # Make 'rest' be the first category in the list
    categories = np.roll(
        categories, len(categories) - np.where(categories == 'rest')[0])

    # Initialize series array
    n_scans = len(sessions_id)
    series = np.zeros(n_scans)
    for c, category in enumerate(categories):
        series[target == category] = c

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, sessions=sessions_id)
    func_filename = haxby_dataset.func[subject]
    fmri = nifti_masker.fit_transform(func_filename)

    return fmri, series, sessions_id, categories


def create_paradigm(series, categories, tr):
    """
    Generates the experimental paradigm for a given set of events.

    Parameters
    ----------

    series: numpy array of shape [n_scans]
        time series of the stimuli, coded as ints from 0 to 8

    categories: string list of length 9
        list of all the categories, in the order they are coded in series

    tr: float
        temporal resolution of the acquisition

    Returns
    -------

    paradigm: pandas DataFrame
        experimental paradigm, contains the following arrays of shape [n_events]

            onset: contains the time of the onset of each event
            name: contains the name of the condition of each event
            duration: contains the duration of each event
    """
    onsets = []
    con_id = []
    n_scans = len(series)
    for scan in range(1, n_scans):
        for category in range(1, len(categories)):  # exclude 'rest'
            if series[scan] == category and series[scan - 1] != category:
                onsets.append(scan * tr)
                con_id.append(categories[category])

    paradigm = pd.DataFrame({'onset': onsets, 'name': con_id,
                             'duration': np.repeat(9 * tr, len(onsets))})

    return paradigm


def apply_time_window(fmri, series, sessions_id, time_window=8, delay=3):
    """
    Applies a time window to a given experiment, extending the fmri to contain
    voxel activations of all scans in that window at each time step, and
    adjusting the time series and session identifiers accordingly. Also adjusts
    for the delay in the Haxby dataset.

    Parameters
    ----------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    series: numpy array of shape [n_scans]
        time series of the stimuli, coded as ints from 0 to 8

    sessions_id: numpy array of shape [n_scans]
        identifications of the sessions each scan belongs to (12 in total)

    time_window: int
        length of the time window to be applied (defaults to 8)

    delay: int
        length of the delay on the dataset to be corrected (defaults to 3)

    Returns
    -------

    fmri_window: numpy array of shape [n_scans - time_window, n_voxels]
        data from the fmri sessions corrected for the time window and the delay

    series_window: numpy array of shape [n_scans - time_window]
        time series of the stimuli, coded as ints from 0 to 8, corrected for the
        time window and the delay

    sessions_id_window: numpy array of shape [n_scans - time_window]
        identifications of the sessions each scan belongs to (12 in total),
        corrected for the time window and the delay
    """
    n_scans = np.shape(fmri)[0]
    n_voxels = np.shape(fmri)[1]

    if time_window - delay == 0:
        series_window = series[delay:]
        sessions_id_window = sessions_id[delay:]

    else:
        series_window = series[delay: -(time_window - delay)]
        sessions_id_window = sessions_id[delay: -(time_window - delay)]

    fmri_window = np.asarray([fmri[scan: scan + time_window]
                              for scan in range(n_scans - time_window)])
    # Reshape to create a vector for each time window, keeping the same voxels
    # at different scans close together, ie. the time dimension varies faster
    # than the spatial one in the vector
    fmri_window = fmri_window.reshape((n_scans - time_window,
                                       time_window * n_voxels), order='F')

    return fmri_window, series_window, sessions_id_window


def to_one_hot(series):
    """ Converts a time series to one-hot encoding """
    n_cats = len(np.unique(series))
    one_hot = np.zeros((n_cats, len(series)))
    for scan in range(len(series)):
        one_hot[series[scan]][scan] = 1
    return one_hot.T


def fit_log(fmri_train, fmri_test, series_train, series_test, n_c, n_jobs=2):
    """ Fits a multinomial logistic regression on the data. """
    # Fit and predict
    log = linear_model.LogisticRegressionCV(Cs=n_c, n_jobs=n_jobs)
    log.fit(fmri_train, series_train)
    prediction = log.predict(fmri_test)
    prediction_proba = log.predict_proba(fmri_test)
    # Score
    accuracy = log.score(fmri_test, series_test)

    return prediction, prediction_proba, accuracy


def _create_time_smoothing_kernel(length, penalty=10., time_window=3):
    """ Creates a kernel matrix and its inverse for RKHS """

    if time_window == 3:
        sample_matrix = np.array([[1, 1, 1], [-1, 2, -1], [-1, 0, 1]])
        k_block = sample_matrix * [[1./np.sqrt(3)],
                                   [penalty * 1./np.sqrt(6)],
                                   [1./np.sqrt(2)]]

    elif time_window == 5:
        sample_matrix = np.array([[1, 1, 1, 1, 1],
                                  [-1, 1, 0, 0, 0],
                                  [0, -1, 1, 0, 0],
                                  [0, 0, -1, 1, 0],
                                  [0, 0, 0, -1, 1]])
        q, _ = np.linalg.qr(sample_matrix.T)
        k_block = q.T * [[1. / (penalty ** 4)],
                         [penalty],
                         [penalty],
                         [penalty],
                         [penalty]]

    k = np.kron(np.eye(length), k_block)

    inv_k_block = np.linalg.pinv(k_block)
    inv_k = np.kron(np.eye(length), inv_k_block)

    return k, inv_k


def _create_voxel_weighing_kernel(betas, time_window):
    """ Takes the beta coefficients learned by ridge regression and returns a
    diagonal kernel with entries equal to the inverse of the each voxel's beta's
    norm """
    betas_norm = betas.reshape(-1, time_window, order='F')
    betas_norm = np.repeat(np.linalg.norm(betas_norm, axis=1), time_window)
    # Perform a prior normalization first to make sure no entries go to zero or
    # infinity, and then normalize based on the product so that the kernel's
    # determinant is equal to 1
    betas_norm /= np.mean(betas_norm)  # Some prior form of normalization
    betas_norm /= np.prod(betas_norm) ** (1./len(betas_norm))
    kernel = np.diag(1. / betas_norm)
    print(np.linalg.det(kernel))
    inv_kernel = np.diag(betas_norm)

    return kernel, inv_kernel


def fit_ridge(fmri_train, fmri_test, one_hot_train, one_hot_test,
              paradigm=None, cutoff=0, n_alpha=5, kernel=None,
              penalty=10, time_window=8, n_iterations=1):
    """
    Fits a Ridge regression on the data, using cross validation to choose the
    value of alpha. Also applies a low-pass filter using a Discrete Cosine
    Transform and regresses out confounds.

    Parameters
    ----------

    fmri_train: numpy array of shape [n_scans_train, n_voxels]
        train data from the fmri sessions

    fmri_test: numpy array of shape [n_scans_train, n_voxels]
        test data from the fmri sessions

    one_hot_train: numpy array of shape [n_scans, n_categories]
        time series of the train stimuli with one-hot encoding

    one_hot_test: numpy array of shape [n_scans, n_categories]
        time series of the test stimuli with one-hot encoding

    paradigm: pandas DataFrame
        experimental paradigm, as implemented in nistats. See 'create_paradigm'

    cutoff: float
        period (in seconds) of the cutoff for the low-pass filter.
        Defaults to 0 (no filtering).

    n_alpha: int
        number of alphas to test (logarithmically distributed around 1).
        Defaults to 5.

    kernel: string or None
        type of kernel to use: options are 'time_smoothing' and 'voxel_weighing'
        Defaults to None (identity matrix).

    penalty: float
        the ratio to be used for penalization of the difference to the median
        in relation to the average

    time_window: int
        the time window applied to the fmri data

    n_iterations: int
        number of regression iterations to perform for the 'voxel_weighing'
        kernel

    Returns
    -------

    prediction: numpy array of size [n_categories, n_test_scans]
        model prediction for the test fmri data

    score: numpy array of size [n_categories]
        prediction r2 score for each category
    """
    # Create drifts for each session separately, then stack to obtain drifts
    # for training and testing
    if cutoff != 0:
        session_drift = 1 * (make_design_matrix(2.5 * np.arange(0, 121),
                                                hrf_model='fir',
                                                drift_model='cosine',
                                                period_cut=cutoff,
                                                paradigm=paradigm
                                                ).as_matrix()).tolist()

        train_drift = np.asarray((session_drift * 10)[:len(fmri_train)])

        # Correct fmri signal and one-hot matrices using the drifts
        DDT_inv = np.linalg.pinv(np.dot(train_drift, train_drift.T))
        correction = np.dot(np.dot(DDT_inv, train_drift), train_drift.T)

        fmri_train = fmri_train - np.dot(correction, fmri_train)
        one_hot_train = one_hot_train - np.dot(correction, one_hot_train)

    # Create alphas and initialize ridge estimator
    alphas = np.logspace(- n_alpha / 2, n_alpha - (n_alpha / 2), num=n_alpha)
    ridge = linear_model.RidgeCV(alphas=alphas)

    if kernel is None:
        # Fit and predict
        ridge.fit(fmri_train, one_hot_train)
        prediction = ridge.predict(fmri_test)

    if kernel == 'time_smoothing':
        # Fit time-smoothing RKHS model
        n_voxels = len(fmri_train[0])/time_window
        kernel, inv_kernel = _create_time_smoothing_kernel(
            n_voxels, penalty=penalty, time_window=time_window)
        fmri_train = np.dot(fmri_train, inv_kernel)
        fmri_test = np.dot(fmri_test, inv_kernel)
        # Fit and predict
        ridge.fit(fmri_train, one_hot_train)
        prediction = ridge.predict(fmri_test)

    elif kernel == 'voxel_weighing':
        ridge.fit(fmri_train, one_hot_train)
        prediction = ridge.predict(fmri_test)
        betas = ridge.coef_.T
        for iteration in range(n_iterations):
            new_betas = np.zeros_like(betas)
            new_prediction = np.zeros_like(prediction)
            # Perform a ridge regression to obtain the beta maps
            for category in range(betas.shape[1]):
                cat_betas = betas[:, category]
                # Fit voxel-weighing RHKS model
                kernel, inv_kernel = _create_voxel_weighing_kernel(
                    cat_betas, time_window=time_window)
                new_fmri_train = np.dot(fmri_train, inv_kernel)
                new_fmri_test = np.dot(fmri_test, inv_kernel)
                ridge.fit(new_fmri_train, one_hot_train[:, category])
                new_prediction[:, category] = ridge.predict(new_fmri_test)
                new_betas[:, category] = ridge.coef_.T

            betas = new_betas
            prediction = new_prediction

    # Score
    score = metrics.r2_score(
        one_hot_test, prediction, multioutput='raw_values')
    return prediction, score


def create_embedding(fmri, series, categories, n_sessions=12):
    """
    Creates an embedding of the points in the stimuli blocks using the
    experiment data as input.

    Parameters
    ----------

    fmri: numpy array of shape [n_scans_train, n_voxels]
        data from the fmri sessions

    series: numpy array of shape [n_scans]
        time series of the train stimuli

    categories: numpy array of shape[n_categories]
        list of all the categories, in the order they are coded in series

    Returns
    -------

    embedding: numpy array of shape [duration * n_sessions * n_categories,
    n_voxels]
        embedding of all the data in the stimuli blocks

    labels: numpy array of shape[duration * n_sessions * n_categories]
        labels identifying the categories of the stimuli
        (int in range 0, len(categories) - 1, as the category 'rest is
        eliminated')
    """
    embedding = np.zeros(
        (len(categories) - 1, 9 * n_sessions, np.shape(fmri)[1]))
    labels = []
    for category in range(len(categories) - 1):
        embed_times = [time for time in range(1, len(fmri))
                       if series[time] == category + 1]
        embedding[category] = fmri[embed_times]
        labels.append([category] * n_sessions * 9)

    embedding = np.vstack(embedding)
    labels = np.asarray(labels).ravel()
    return embedding, labels
