from nistats.design_matrix import make_design_matrix
from nilearn.input_data import NiftiMasker
from sklearn import linear_model
from sklearn import metrics
import numpy as np


def read_data(subject, haxby_dataset):
    """
    Generates data for a given haxby dataset subject.

    Parameters
    ----------

    subject: int from 0 to 5
        id of the subject whose data should be retrieved

    haxby_dataset: dictionary
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
    labels = np.recfromcsv(haxby_dataset.session_target[subject], delimiter=" ")
    sessions_id = labels['chunks']
    target = labels['labels']
    categories = np.unique(target)
    # Make 'rest' be the first category in the list
    categories = np.roll(categories,
                         len(categories) - np.where(categories == 'rest')[0])

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


def conditions_onsets(series, categories, tr):
    """
    Generates conditions and onsets for a given experiment.

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
    con_id: numpy array of shape [n_events]
        identifier for the conditions of each onset

    onsets: numpy array of shape [n_events]
        time of the onset of each event

    """
    onsets = []
    con_id = []
    n_scans = len(series)
    for scan in range(1, n_scans):
        for category in range(len(categories)):
            if series[scan] == category and series[scan - 1] != category:
                onsets.append(scan * tr)
                con_id.append(categories[category])
    return con_id, onsets


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


def fit_ridge(fmri_train, fmri_test, one_hot_train, one_hot_test, n_alpha,
              cutoff=24):
    """
    Fits a Ridge regression on the data, using cross validation to choose the
    value of alpha. Also applies a low-pass filter using a Discrete Cosine
    Transform.

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

    n_alpha: int
        number of alphas to test (logarithmically distributed around 1)

    cutoff: float
        period (in seconds) of the cutoff for the low-pass filter

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
    # Create drifts for each session separately, then stack to obtain drifts
    # for training and testing
    session_drift = 100 * (make_design_matrix(2.5 * np.arange(0, 121),
                                              drift_model='cosine',
                                              period_cut=cutoff
                                              ).as_matrix()).tolist()

    train_drift = (session_drift * 10)[:len(fmri_train)]
    test_drift = (session_drift * 2)[:len(fmri_test)]
    # Create alphas
    alphas = np.logspace(n_alpha/2, n_alpha - (n_alpha/2), num=n_alpha)

    # Fit and predict
    ridge = linear_model.RidgeCV(alphas=alphas)
    ridge.fit(np.hstack((fmri_train, train_drift)), one_hot_train)
    predict = ridge.predict(np.hstack((fmri_test, np.zeros_like(test_drift))))

    # Score
    score = metrics.r2_score(one_hot_test, predict, multioutput='raw_values')

    return predict, score
