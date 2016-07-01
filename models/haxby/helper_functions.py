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

    categories: string list of length 9
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
    mask_filename = haxby_dataset.mask_vt[subject]
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

    fmri_window: numpy array of shape [n_scans - (time_window + delay),
    n_voxels]
        data from the fmri sessions corrected for the time window and the delay

    series_window: numpy array of shape [n_scans - (time_window + delay)]
        time series of the stimuli, coded as ints from 0 to 8, corrected for the
        time window and the delay

    sessions_id_window: numpy array of shape [n_scans - (time_window + delay)]
        identifications of the sessions each scan belongs to (12 in total),
        corrected for the time window and the delay
    """
    series_window = series[delay: -(time_window - delay)]
    sessions_id_window = sessions_id[delay: -(time_window - delay)]
    fmri_window = np.asarray([fmri[scan: scan + time_window]
                              for scan in range(len(fmri) - time_window)])
    fmri_window = fmri_window.reshape((np.shape(fmri_window)[0],
                                       time_window * np.shape(fmri)[1]))

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


def fit_ridge(fmri_train, fmri_test, one_hot_train, one_hot_test,
              paradigm=None, cutoff=0, n_alpha=5):
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

    one_hot_train: numpy array of shape [n_scans]
        time series of the train stimuli with one-hot encoding

    one_hot_test: numpy array of shape [n_scans]
        time series of the test stimuli with one-hot encoding

    paradigm: pandas DataFrame
        experimental paradigm, as implemented in nistats. See 'create_paradigm'

    cutoff: float
        period (in seconds) of the cutoff for the low-pass filter.
        Defaults to 0 (no filtering).

    n_alpha: int
        number of alphas to test (logarithmically distributed around 1).
        Defaults to 5.

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

    # Create alphas
    alphas = np.logspace(n_alpha / 2, n_alpha - (n_alpha / 2), num=n_alpha)

    # Fit and predict
    ridge = linear_model.RidgeCV(alphas=alphas)
    ridge.fit(fmri_train, one_hot_train)
    prediction = ridge.predict(fmri_test)

    # Score
    score = metrics.r2_score(
        one_hot_test, prediction, multioutput='raw_values')
    return prediction, score


def create_embedding(fmri, series, categories, n_sessions=12):
    """
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
