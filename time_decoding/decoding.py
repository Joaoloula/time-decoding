from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from nistats.design_matrix import make_design_matrix
from sklearn import linear_model, metrics
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import itertools


def feature_selection(fmri_train, fmri_test, stimuli_train, k=10000,
                      selector=False):
    """ Applies anova feature selection to fmri data using classification
    accuracy on stimuli data as measure of performance """

    # Fit the anova feature selection
    anova = SelectKBest(f_classif, k=k)
    fmri_train = anova.fit_transform(fmri_train, stimuli_train)

    # Transform the given test set
    fmri_test = anova.transform(fmri_test)

    if selector == True:
        return fmri_train, fmri_test, anova

    return fmri_train, fmri_test


def apply_time_window(fmri_list, stimuli_list, time_window, delay):

    if len(stimuli_list.shape) != 3:
        stimuli_list = stimuli_list.reshape(12, -1, 3)

    mask = [np.sum(stimuli[:, 1:], axis=1) for stimuli in stimuli_list]

    if delay != 0:
        fmri_list = [fmri[delay:] for fmri in fmri_list]
        stimuli_list = [stimuli[:-delay] for stimuli in stimuli_list]

    fmri_window = [[fmri_list[block][scan: scan + time_window].ravel()
                    for scan in xrange(len(fmri_list[block]) - time_window + 1)
                    if mask[block][scan] == 1]
                   for block in range(len(fmri_list))]

    return fmri_window


def apply_time_window2(fmri_train, fmri_test, stimuli_train, stimuli_test,
                      delay=3, time_window=8, k=10000):
    """
    Applies a time window to a given experiment, extending the fmri to contain
    voxel activations of all scans in that window at each time step, and
    adjusting the time series and session identifiers accordingly. Also adjusts
    for a given delay in the embedding, and performs feature selection.

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

    n_scans_train, n_scans_test = fmri_train.shape[0], fmri_test.shape[0]

    fmri_train = [fmri_train[scan: scan + time_window].ravel()
                  for scan in xrange(n_scans_train - time_window + 1)]
    fmri_test = [fmri_test[scan: scan + time_window].ravel()
                 for scan in xrange(n_scans_test - time_window + 1)]

    if time_window != 1:
        stimuli_train, stimuli_test = (stimuli_train[: -(time_window - 1)],
                                       stimuli_test[: -(time_window - 1)])

    # Fit the anova feature selection
    stimuli_train_1d = np.argmax(stimuli_train, axis=1)
    anova = SelectKBest(f_classif, k=k)
    fmri_train = anova.fit_transform(fmri_train, stimuli_train_1d)
    fmri_test = anova.transform(fmri_test)

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


def ridge_scoring(prediction, stimuli):
    """ Returns a classification score from a regressor by doing a softmax """
    # Restrain analysis to scans with stimuli (i.e. no 'rest' category)
    mask = np.sum(stimuli[:, 1:], axis=1).astype(bool)
    prediction, stimuli = np.array((prediction[mask][:, 1:],
                                    stimuli[mask][:, 1:]))

    classifier = np.zeros_like(stimuli)
    for stim in range(len(classifier)):
        predicted_class = np.argmax(prediction[stim])
        classifier[stim][predicted_class] = 1

    score = metrics.accuracy_score(stimuli, classifier)

    return score


def logistic_deconvolution(estimation_train, estimation_test, stimuli_train,
                           stimuli_test, logistic_window, delay=0,
                           balance=False, n_tests=20, block=None,
                           session_id_onset=None):
    """ Learn a deconvolution filter for classification given a time window
    using logistic regression """
    log = linear_model.LogisticRegressionCV()

    if delay != 0:
        estimation_train, estimation_test = (estimation_train[delay:],
                                             estimation_test[delay:])
        stimuli_train, stimuli_test = (stimuli_train[:-delay],
                                       stimuli_test[:-delay])

    cats_train = [
        estimation_train[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_train) - logistic_window + 1)]
    cats_test = [
        estimation_test[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_test) - logistic_window + 1)]

    train_mask = np.sum(
        stimuli_train[:len(cats_train), 1:], axis=1).astype(bool)
    test_mask = np.sum(
        stimuli_test[:len(cats_test), 1:], axis=1).astype(bool)

    stimuli_train, stimuli_test = (
        np.argmax(stimuli_train[:len(cats_train)][train_mask], axis=1),
        np.argmax(stimuli_test[:len(cats_test)][test_mask], axis=1))
    cats_train, cats_test = (
        np.array(cats_train)[train_mask], np.array(cats_test)[test_mask])

    if balance:
        accuracy = 0
        # Balance classes in train set
        isi_id = [round(19.2 / len(np.where(session_id_onset == trial)[0]), 2)
                  for trial in range(12)]
        isi_id = np.delete(isi_id, block)
        onsets = np.delete(session_id_onset,
                           np.union1d(np.where(session_id_onset == block[0])[0],
                                      np.where(session_id_onset == block[1])[0])
                           )
        combinations_face = np.asarray([
            [item for item in
             itertools.combinations(np.where(onsets == trial)[0][::2], 2)]
            for trial in range(12) if (trial not in block)])
        combinations_house = np.asarray([
            [item for item in
             itertools.combinations(np.where(onsets == trial)[0][1::2], 2)]
            for trial in range(12) if (trial not in block)])

        for iteration in range(n_tests):
            """
            balanced_trials = np.union1d(
                np.random.choice(np.where(isi_id == 1.6)[0], 2, False),
                np.union1d(
                    np.random.choice(np.where(isi_id == 3.2)[0], 2, False),
                    np.random.choice(np.where(isi_id == 4.8)[0], 2, False)))
            """
            balanced_trials = np.where(isi_id == isi_id[block[0]])
            balanced_combinations_face = combinations_face[balanced_trials]
            balanced_combinations_house = combinations_house[balanced_trials]
            balance_index = np.hstack([np.union1d(
                balanced_combinations_face[trial][np.random.randint(len(
                    balanced_combinations_face[trial]))],
                balanced_combinations_house[trial][np.random.randint(len(
                    balanced_combinations_house[trial]))])
                for trial in range(len(balanced_combinations_face))])
            selected_cats_train = cats_train[balance_index]
            selected_stimuli_train = stimuli_train[balance_index]
            log.fit(selected_cats_train, selected_stimuli_train)
            accuracy += log.score(cats_test, stimuli_test)
        accuracy /= n_tests

    else:
        log.fit(cats_train, stimuli_train)
        accuracy = log.score(cats_test, stimuli_test)

    return accuracy


def forest_deconvolution(estimation_train, estimation_test, stimuli_train,
                         stimuli_test, logistic_window, delay=0):
    """ Learn a deconvolution filter for classification given a time window
    using logistic regression """
    forest = RandomForestClassifier()

    if delay != 0:
        estimation_train, estimation_test = (estimation_train[delay:],
                                             estimation_test[delay:])
        stimuli_train, stimuli_test = (stimuli_train[:-delay],
                                       stimuli_test[:-delay])

    cats_train = [
        estimation_train[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_train) - logistic_window + 1)]
    cats_test = [
        estimation_test[scan: scan + logistic_window].ravel()
        for scan in xrange(len(estimation_test) - logistic_window + 1)]

    train_mask = np.sum(
        stimuli_train[:len(cats_train), 1:], axis=1).astype(bool)
    test_mask = np.sum(
        stimuli_test[:len(cats_test), 1:], axis=1).astype(bool)

    stimuli_train, stimuli_test = (
        np.argmax(stimuli_train[:len(cats_train)][train_mask], axis=1),
        np.argmax(stimuli_test[:len(cats_test)][test_mask], axis=1))
    cats_train, cats_test = (
        np.array(cats_train)[train_mask], np.array(cats_test)[test_mask])

    forest.fit(cats_train, stimuli_train)
    accuracy = forest.score(cats_test, stimuli_test)

    return accuracy


def design_matrix(n_scans, tr, onsets, conditions, durations=None,
                  hrf_model='spm', drift_model='cosine'):
    """ """
    frame_times = np.arange(n_scans) * tr
    paradigm = {}
    paradigm['onset'] = onsets
    paradigm['name'] = conditions
    if durations is not None:
        paradigm['duration'] = durations
    paradigm = pd.DataFrame(paradigm)

    X = make_design_matrix(frame_times, paradigm, hrf_model=hrf_model,
                           drift_model=drift_model)
    return X


def glm(fmri, tr, onsets, conditions=None, durations=None, hrf_model='spm',
        drift_model='cosine', model='GLM'):
    """ Fit a GLM for comparison with time decoding model """
    betas = []
    regressors = []
    for session in range(len(fmri)):
        n_scans = len(fmri[session])
        if conditions is not None:
            separate_conditions = conditions[session]
        else:
            separate_conditions = np.arange(len(onsets[session]))
        if durations is not None:
            X = design_matrix(n_scans, tr, onsets[session], separate_conditions,
                              durations=durations[session],
                              drift_model=drift_model)
        else:
            X = design_matrix(n_scans, tr, onsets[session], separate_conditions,
                              drift_model=drift_model)
        if model == 'GLMs':
            session_betas = []
            design_sum = np.sum(X, axis=1)
            for condition in np.unique(separate_conditions):
                separate_X = np.array([X[condition],
                                      design_sum - X[condition]]).T
                separate_beta = np.dot(np.linalg.pinv(separate_X),
                                       fmri[session])[0]
                session_betas.append(separate_beta)

        elif model == '2GLMs':
            session_betas = []
            face_id = np.arange(len(separate_conditions))[::2]
            house_id = np.arange(len(separate_conditions))[1::2]
            face_sum = np.sum(X[face_id], axis=1)
            house_sum = np.sum(X[house_id], axis=1)
            for condition in np.unique(separate_conditions):
                if condition % 2 == 0:
                    separate_X = np.array([X[condition],
                                           face_sum,
                                           house_sum - X[condition]]).T
                else:
                    separate_X = np.array([X[condition],
                                           face_sum - X[condition],
                                           house_sum]).T

                separate_beta = np.dot(np.linalg.pinv(separate_X),
                                       fmri[session])[0]
                session_betas.append(separate_beta)

        else:
            session_betas = np.dot(np.linalg.pinv(X), fmri[session])

        betas.append(session_betas[:len(np.unique(separate_conditions))])
        regressors.append(X.columns[:len(np.unique(separate_conditions))])

    betas = np.array(betas)
    regressors = np.array(regressors)
    return betas, regressors


def glm_scoring(betas_train, betas_test, labels_train, labels_test):
    """ Fits a logistic regression and scores it for a glm estimation """
    log = linear_model.LogisticRegression()
    log.fit(betas_train, labels_train)
    score = log.score(betas_test, labels_test)
    return score

def svm_scoring(betas_train, betas_test, labels_train, labels_test):
    """ Fits a logistic regression and scores it for a glm estimation """
    svc = SVC(kernel='linear')
    svc.fit(betas_train, labels_train)
    score = svc.score(betas_test, labels_test)
    return score
