from nistats.design_matrix import make_design_matrix
from nilearn.image import load_img
from nilearn import input_data
from sklearn import linear_model, metrics, manifold, decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import hrf_estimation as he
import pandas as pd
import numpy as np
import glob
import os


def _read_fmri(sub, run, path):
    """ Reads BOLD signal data """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    fmri_path = (('{path}/ses-test/func/r{sub}_ses-test_{task}'
                  '_{run}_bold.nii')
                 .format(path=path, sub=sub, task=task, run=run))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimulus(sub, run, path, n_scans, tr, two_classes, glm=False):
    """ Reads stimulus data """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    onsets_path = (('{path}/ses-test/func/{sub}_ses-test_{task}'
                    '_{run}_events.tsv')
                   .format(path=path, sub=sub, task=task, run=run))

    paradigm = pd.read_csv(onsets_path, sep='\t')
    onsets, durations, conditions = (paradigm['onset'], paradigm['duration'],
                                     paradigm['trial_type'])
    if glm:
        return onsets, conditions

    cats = np.array(['rest', 'junk', 'pl_ns', 'pl_sw', 'mr_ns', 'mr_sw'])
    stimuli = np.zeros((n_scans, len(cats)))
    for index, condition in enumerate(conditions):
        stim_onset = int(onsets.loc[index])/tr
        stim_duration = 1 + int(durations.loc[index])/tr
        (stimuli[stim_onset: stim_onset + stim_duration,
                 np.where(cats == condition)[0][0]]) = 1
    # Fill the rest with 'rest'
    stimuli[np.logical_not(np.sum(stimuli, axis=1).astype(bool)), 0] = 1

    if two_classes:
        stimuli = np.array([stimuli[:, 0],
                            stimuli[:, 2] + stimuli[:, 3],
                            stimuli[:, 4] + stimuli[:, 5],
                            stimuli[:, 1]]).T

    return stimuli


def read_data(subject, n_runs=6, tr=2, n_scans=205, two_classes=False,
    path='/home/loula/Programming/python/neurospin/ds006_R2.0.0/results/',
    glm=False):
    """
    Reads data from the Jimura dataset.

    Parameters
    ----------

    subject: int from 0 to 13
        subject from which to read the data

    n_runs: int from 1 to 6
        number of runs to read from the subject, defaults to 6

    tr: float
        repetition time for the task (defaults to 2)

    n_scans: int
        number of scans per run, defaults to 205

    two_classes: bool
        whether to restrict stimuli to two classes (text vs. reversed text), or
        to keep all the four original categories (which also include whether a
        given stimulus is the same as the one that precedes it)

    Returns
    -------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    stimuli: numpy array of shape [n_scans, n_classes]
        labels for the stimuli in one-hot encoding

    """
    if subject < 9:
        sub = 'sub-0' + str(subject + 1)
    else:
        sub = 'sub-' + str(subject + 1)
    path += sub

    os.chdir(path + "/ses-test/func/")
    runs = []
    for run in range(n_runs):
        if glob.glob("*run-0" + str(run + 1) + "*"):
            runs.append(run)

    stimuli = [_read_stimulus(sub, run, path, n_scans, tr, two_classes, glm)
               for run in runs]

    fmri = [_read_fmri(sub, run, path) for run in runs]

    labels = []
    for run_n in range(len(runs)):
        if glm:
            labels += [runs[run_n]] * len(stimuli[run_n][0])
        else:
            labels += [runs[run_n]] * n_scans

    return fmri, stimuli, labels


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


def uniform_masking(fmri_list, high_pass=0.01, smoothing=5):
    """ Mask all the sessions uniformly, doing standardization, linear
    detrending, DCT high_pas filtering and gaussian smoothing.

    Parameters
    ----------

    fmri_list: array-like
        array containing multiple BOLD data from different sessions

    high_pass: float
        frequency at which to apply the high pass filter, defaults to 0.01

    smoothing: float
        spatial scale of the gaussian smoothing filter in mm, defaults to 5

    Returns
    -------

    fmri_list_masked: array-like
        array containing the masked data

    """
    masker = input_data.MultiNiftiMasker(mask_strategy='epi', standardize=True,
                                         detrend=True, high_pass=0.01, t_r=2,
                                         smoothing_fwhm=smoothing)
    fmri_list_masked = masker.fit_transform(fmri_list)

    return fmri_list_masked


def _create_time_smoothing_kernel(length, penalty=10., time_window=3):
    """ Creates a kernel matrix and its inverse for RKHS """

    if time_window not in [3, 5]:
        raise NotImplementedError

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
    betas_norm /= np.prod(betas_norm ** (1./len(betas_norm)))
    kernel = 1. / betas_norm
    inv_kernel = betas_norm

    return kernel, inv_kernel


def fit_ridge(fmri_train, fmri_test, one_hot_train, one_hot_test,
              n_alpha=5, kernel=None, penalty=10, time_window=3,
              n_iterations=1, classify=False, k=10000, double_prediction=False,
              extra=None):
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

    if kernel not in [None, 'time_smoothing', 'voxel_weighing']:
        raise NotImplementedError

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
                new_fmri_train = np.multiply(fmri_train, inv_kernel)
                new_fmri_test = np.multiply(fmri_test, inv_kernel)
                ridge.fit(new_fmri_train, one_hot_train[:, category])
                new_prediction[:, category] = ridge.predict(new_fmri_test)
                new_betas[:, category] = ridge.coef_.T

            betas = new_betas
            prediction = new_prediction

    if classify:
        mask = np.logical_not(one_hot_test[:, 0])
        class_regression, class_one_hot = prediction[mask], one_hot_test[mask]
        class_prediction = np.zeros_like(class_regression)
        for scan in range(class_regression.shape[0]):
            class_prediction[scan][np.argmax(class_regression[scan])] = 1
        class_score = metrics.accuracy_score(class_one_hot, class_prediction)
        return prediction, class_score

    score = metrics.r2_score(
        one_hot_test, prediction, multioutput='raw_values')

    if double_prediction:
        extra_prediction = ridge.predict(extra)

        return prediction, extra_prediction, score

    return prediction, score


def fit_logistic_regression(fmri_train, fmri_test, stimuli_train, stimuli_test,
                            k=10000):
    """ Fits a logistic regression to the data.

    Parameters
    ----------

    fmri_train: numpy array of shape [n_scans_train, n_voxels]
        train data from the fmri sessions

    fmri_test: numpy array of shape [n_scans_test, n_voxels]
        test data from the fmri sessions

    stimuli_train: numpy array of shape [n_scans_train, n_categories]
        train labels for the stimuli

    stimuli_test: numpy array of shape [n_scans_test, n_categories]
        test labels for the stimuli

    k: int
        number of features to select on anova

    Returns
    -------

    prediction: array of shape [n_scans_test]
        prediction for the classes

    score: array of shape [n_categories]
        accuracy scores for the prediction

    """
    anova = SelectKBest(f_classif, k=k)
    log = linear_model.LogisticRegression()
    anova_log = Pipeline([('anova', anova), ('log', log)])

    # Transform one-hot to int encoding
    new_stimuli_train = np.zeros(stimuli_train.shape[0])
    for scan in range(len(new_stimuli_train)):
        new_stimuli_train[scan] = np.argmax(stimuli_train[scan])

    new_stimuli_test = np.zeros(stimuli_test.shape[0])
    for scan in range(len(new_stimuli_test)):
        new_stimuli_test[scan] = np.argmax(stimuli_test[scan])

    anova_log.fit(fmri_train, new_stimuli_train)
    prediction = anova_log.predict(fmri_test)
    probas = anova_log.predict_proba(fmri_test)
    score = anova_log.score(fmri_test, new_stimuli_test)

    return probas, score


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


def hrf_line(onset_scan, n_scans, hrf_length=32.):
    """ Create a line for the convolution matrix used in the deconvolution
    function"""
    hrf = he.hrf.spmt(np.linspace(0, hrf_length, (hrf_length // 2.)))
    # hrf = glover_hrf(tr=2., oversampling=1, onset=0, time_length=hrf_length)
    hrf_size = len(hrf)
    padding = n_scans - onset_scan - hrf_size
    if padding >= 0:
        line = np.concatenate((np.zeros(onset_scan), hrf, np.zeros(padding)))

    else:
        line = np.concatenate((np.zeros(onset_scan), hrf[: padding]))

    return line


def convolve_events(conditions, onsets, n_scans, basis='hrf'):
    """ Creates a design matrix with the events convolved with an hrf specified
    by the 'basis' argument """
    X, _ = he.create_design_matrix(conditions, onsets, TR=2.,
                                   n_scans=n_scans, basis=basis,
                                   oversample=1, hrf_length=32)

    return X


def deconvolution(reg_estimation, hrf_model='glover'):
    """ Deconvolve an estimation obtained by regression by solving a Ridge
    regularization problem with a convolution matrix created by stacking time-
    lagged HRFs """
    n_scans, n_classes = reg_estimation.shape
    if hrf_model == 'glover':
        conv_matrix = [hrf_line(scan, n_scans) for scan in range(n_scans)]
    ridge = linear_model.RidgeCV()
    ridge.fit(np.array(conv_matrix).T, reg_estimation)
    deconvolved_estimation = ridge.coef_

    return deconvolved_estimation


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


def glm(fmri, glm_stimuli, labels, basis='hrf', mode='glm'):
    """ Fit a GLM for comparison with time decoding model """
    onsets = np.empty(len(labels))
    conditions = np.empty(len(labels), dtype='str')
    start = 0
    for run, stim in enumerate(glm_stimuli):
        onsets[start: start + len(stim[0])] = stim[0] + (run * 410)
        conditions[start: start + len(stim[0])] = stim[1]
        start += len(stim[0])

    # Correction for problematic onsets
    if onsets[-1] > (len(fmri) - 1) * 2:
        onsets[-1] = (len(fmri) - 1) * 2

    tr = 2.
    frame_times = np.arange(len(fmri)) * tr
    separate_conditions = xrange(len(conditions))
    paradigm = {}
    paradigm['onset'] = onsets
    paradigm['name'] = separate_conditions
    paradigm = pd.DataFrame(paradigm)

    X = make_design_matrix(frame_times, paradigm, hrf_model='spm')
    """
    X, hrf = he.create_design_matrix(
            separate_conditions, onsets, tr, len(fmri), basis=basis,
            oversample=1, hrf_length=20)
    """
    if mode == 'glm':
        betas = np.dot(np.linalg.pinv(X), fmri)

    return None, betas, conditions, onsets


def glm_scoring(betas_train, betas_test, labels_train, labels_test):
    """ Fits a logistic regression and scores it for a glm estimation """
    log = linear_model.LogisticRegression()
    log.fit(betas_train, labels_train)
    score = log.score(betas_test, labels_test)

    return score


def plot(prediction, stimuli, scores, accuracy, delay=3, time_window=8,
         two_classes=False, kernel=None, penalty=1):
    """ Plots predictions and ground truths for each of the classes, as well
    as their r2 scores. """
    plt.style.use('ggplot')
    if two_classes:
        fig, axes = plt.subplots(2)

        title = ('Ridge predictions for \'plain\' vs. \'reversed\', time window'
                 'of {tw}, delay of {delay}. Accuracy: {acc:.2f}').format(
                 tw=time_window, delay=delay, acc=accuracy)

        if kernel == 'time_smoothing':
            title += ' Kernel: time smoothing, penalty = {}'.format(penalty)

        elif kernel == 'voxel_weighing':
            title += ' Kernel: voxel weighing'

        fig.suptitle(title, fontsize=20)
        axes[0].plot(stimuli[:, 0])
        axes[0].plot(prediction[:, 0])
        axes[0].set_title(('Predictions for category \'plain\', r2 score of '
                           '{score:.2f}').format(score=scores[0]), fontsize=18)
        axes[1].plot(stimuli[:, 1])
        axes[1].plot(prediction[:, 1])
        axes[1].set_title(('Predictions for category \'reversed\', r2 score of '
                           '{score:.2f}').format(score=scores[1]), fontsize=18)
    else:
        cats = np.array(['junk', 'pl_ns', 'pl_sw', 'mr_ns', 'mr_sw'])
        fig, axes = plt.subplots(5)
        fig.suptitle('Ridge predictions for all classes, time window of {tw}'
                     .format(tw=time_window), fontsize=20)
        for cat in range(len(cats)):
            axes[cat].plot(stimuli[:, cat])
            axes[cat].plot(prediction[:, cat])
            axes[cat].set_title(('Prediction for category {cat}, R2 score of '
                                 '{score:.2f}').format(cat=cats[cat],
                                                       score=scores[cat]),
                                fontsize=18)

    plt.show()


def embed(fmri_data, stimuli, decomposer='tsne'):
    """ Creates an embedded visualization of the fmri_data.
    IMPORTANT: both fmri_data and stimuli must already be restricted to the
    times when a stimulus is present """
    if decomposer == 'tsne':
        pca = decomposition.PCA(n_components=50)
        first_embedding = pca.fit_transform(fmri_data)
        tsne = manifold.TSNE()
        embedding = tsne.fit_transform(first_embedding)

    elif decomposer == 'pca':
        pca = decomposition.PCA(n_components=2)
        embedding = pca.fit_transform(fmri_data)

    elif decomposer == 'pca':
        tsvd = decomposition.TruncatedSVD()
        embedding = tsvd.fit_transform(fmri_data)

    plt.style.use('ggplot')
    plt.title('Embedding of the data for {n_classes} classes using t-SNE'
              .format(n_classes=len(stimuli[0])))
    colors = ['b', 'g', 'r', 'y']
    for label in range(len(stimuli[0])):
        plt.scatter(embedding[stimuli[:, label].astype(bool)][:, 0],
                    embedding[stimuli[:, label].astype(bool)][:, 1],
                    color=colors[label])

    plt.show()

    return embedding


def score_barplot(score_list, model_list):
    """ """
    n_subjects = len(score_list[0])
    n_models = len(model_list)

    scores = np.hstack(score_list)
    models = np.hstack([[model] * n_subjects for model in model_list])
    subjects = range(1, n_subjects + 1) * n_models
    dict = {}
    dict['accuracy'] = scores
    dict['model'] = models
    dict['subjects'] = subjects
    data = pd.DataFrame(dict)

    plt.style.use('ggplot')
    sns.set_context('talk', font_scale=1.5)
    ax = sns.boxplot(x='model', y='accuracy', data=data)
    ax.set_title('Classification accuracies for GLM and time-domain decoding')
    ax.set_ylim(0.5, 1)

    plt.show()
