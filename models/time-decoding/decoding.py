from sklearn.feature_selection import SelectKBest, f_classif
from nistats.design_matrix import make_design_matrix
from sklearn import linear_model, metrics
import pandas as pd
import numpy as np


def feature_selection(fmri_train, fmri_test, stimuli_train, k=10000):
    """ Applies anova feature selection to fmri data using classification
    accuracy on stimuli data as measure of performance """

    # Fit the anova feature selection
    anova = SelectKBest(f_classif, k=k)
    fmri_train = anova.fit_transform(fmri_train, stimuli_train)

    # Transform the given test set
    fmri_test = anova.transform(fmri_test)

    return fmri_train, fmri_test


def apply_time_window(fmri_train, fmri_test, stimuli_train, stimuli_test,
                      delay=3, time_window=8, k=10000):
    """
    Applies a time window to a given experiment, extending the fmri to contain
    voxel activations of all scans in that window at each time step, and
    adjusting the time series and session identifiers accordingly. Also adjusts
    for the delay in the Haxby dataset.

    Parameters
    ----------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    stimuli: numpy array of shape [n_scans]
        labels for the stimuli

    time_window: int
        length of the time window to be applied (defaults to 8)

    Returns
    -------

    fmri_window: numpy array of shape [n_scans - time_window, n_voxels]
        data from the fmri sessions corrected for the time window

    stimuli_window: numpy array of shape [n_scans - time_window]
        labels for the stimuli, corrected for the time window

    """
    # Apply the delay
    if delay != 0:
        fmri_train, fmri_test = fmri_train[delay:], fmri_test[delay:]
        stimuli_train, stimuli_test = (stimuli_train[:-delay],
                                       stimuli_test[:-delay])

    # Fit the anova feature selection
    stimuli_train_1d = np.argmax(stimuli_train, axis=1)
    anova = SelectKBest(f_classif, k=k)
    fmri_train = anova.fit_transform(fmri_train, stimuli_train_1d)
    fmri_test = anova.transform(fmri_test)

    n_scans_train, n_scans_test = fmri_train.shape[0], fmri_test.shape[0]

    fmri_train = [fmri_train[scan: scan + time_window].ravel()
                  for scan in xrange(n_scans_train - time_window + 1)]
    fmri_test = [fmri_test[scan: scan + time_window].ravel()
                 for scan in xrange(n_scans_test - time_window + 1)]

    if time_window != 1:
        stimuli_train, stimuli_test = (stimuli_train[: -(time_window - 1)],
                                       stimuli_test[: -(time_window - 1)])

    return fmri_train, fmri_test, stimuli_train, stimuli_test


def fit_ridge(fmri_train, fmri_test, one_hot_train, one_hot_test,
              n_alpha=5, time_window=3, double_prediction=False, extra=None):
    """
    Fits a Ridge regression on the data, using cross validation to choose the
    value of alpha. Also applies a low-pass filter using a Discrete Cosine
    Transform and regresses out confounds.

    Parameters
    ----------

    fmri_train: numpy array of shape [n_scans_train, n_voxels]
        train data from the fmri sessions

    fmri_test: numpy array of shape [n_scans_test, n_voxels]
        test data from the fmri sessions

    one_hot_train: numpy array of shape [n_scans_train, n_categories]
        time series of the train stimuli with one-hot encoding

    one_hot_test: numpy array of shape [n_scans_test, n_categories]
        time series of the test stimuli with one-hot encoding

    n_alpha: int
        number of alphas to test (logarithmically distributed around 1).
        Defaults to 5.

    time_window: int
        the time window applied to the fmri data

    double_prediction: bool
        whether to make a prediction for an extra input as well

    extra: numpy array of shape [n_scans_test, n_voxels]
        extra input to be predicted if double prediction is True

    Returns
    -------

    prediction: numpy array of size [n_categories, n_test_scans]
        model prediction for the test fmri data

    score: numpy array of size [n_categories]
        prediction r2 score for each category
    """

    # Create alphas and initialize ridge estimator
    alphas = np.logspace(- n_alpha / 2, n_alpha - (n_alpha / 2), num=n_alpha)
    ridge = linear_model.RidgeCV(alphas=alphas)

    # Fit and predict
    ridge.fit(fmri_train, one_hot_train)
    prediction = ridge.predict(fmri_test)

    score = metrics.r2_score(
        one_hot_test, prediction, multioutput='raw_values')

    if double_prediction:
        extra_prediction = ridge.predict(extra)

        return prediction, extra_prediction, score

    return prediction, score


def classification_score(prediction, stimuli, mode='regression'):
    """ Returns a classification score from a regressor by doing a softmax """
    # Restrain analysis to scans with stimuli (i.e. no 'rest' category)
    if mode == 'regression':
        mask = np.sum(stimuli[:, 1: -1], axis=1).astype(bool)
        prediction, stimuli = np.array((prediction[mask][:, 1: -1],
                                        stimuli[mask][:, 1: -1]))

    elif mode == 'glm':
        mask = np.sum(stimuli[:, 1: -1], axis=1).astype(bool)
        # Flip prediction to correspond to stimuli
        prediction, stimuli = np.array((np.fliplr(prediction[mask][:, 1:]),
                                       stimuli[mask][:, 1: -1]))

    classifier = np.array([[1, 0]
                           if prediction[scan][0] > prediction[scan][1]
                           else [0, 1]
                           for scan in range(prediction.shape[0])])

    score = metrics.accuracy_score(stimuli, classifier)

    return score


def logistic_deconvolution(estimation_train, estimation_test, stimuli_train,
                           stimuli_test, logistic_window):
    """ Learn a deconvolution filter for classification given a time window
    using logistic regression """
    log = linear_model.LogisticRegressionCV()
    cats_train = [
        estimation_train[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_train) - logistic_window + 1)]
    cats_test = [
        estimation_test[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_test) - logistic_window + 1)]

    train_mask = np.sum(
        stimuli_train[:len(cats_train), 1: -1], axis=1).astype(bool)
    test_mask = np.sum(
        stimuli_test[:len(cats_test), 1: -1], axis=1).astype(bool)

    stimuli_train, stimuli_test = (
        np.argmax(stimuli_train[:len(cats_train)][train_mask], axis=1),
        np.argmax(stimuli_test[:len(cats_test)][test_mask], axis=1))
    cats_train, cats_test = (
        np.array(cats_train)[train_mask], np.array(cats_test)[test_mask])

    log.fit(cats_train, stimuli_train)
    accuracy = log.score(cats_test, stimuli_test)

    return accuracy


def design_matrix(n_scans, tr, onsets, conditions, durations=None,
                  hrf_model='spm'):
    """ """
    frame_times = np.arange(n_scans) * tr
    paradigm = {}
    paradigm['onset'] = onsets
    paradigm['name'] = conditions
    if durations is not None:
        paradigm['duration'] = durations
    paradigm = pd.DataFrame(paradigm)

    X = make_design_matrix(frame_times, paradigm, hrf_model=hrf_model)

    return X


def glm(fmri, onsets, durations=None, hrf_model='spm'):
    """ Fit a GLM for comparison with time decoding model """
    tr = 1.5
    betas = []
    for session in range(len(fmri)):
        n_scans = len(fmri[session])
        separate_conditions = xrange(len(onsets[session]))
        X = design_matrix(n_scans, tr, onsets[session], separate_conditions)
        session_betas = np.dot(np.linalg.pinv(X), fmri[session])
        betas.append(session_betas)
    betas = np.array(betas)

    return betas


def glm_scoring(betas_train, betas_test, labels_train, labels_test):
    """ Fits a logistic regression and scores it for a glm estimation """
    log = linear_model.LogisticRegression()
    log.fit(betas_train, labels_train)
    score = log.score(betas_test, labels_test)

    return score
