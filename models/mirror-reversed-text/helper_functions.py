from nilearn.image import load_img
from nilearn import input_data
from sklearn import linear_model, metrics, manifold, decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import hrf_estimation as he
import pandas as pd
import numpy as np


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
        stimuli = np.array([stimuli[:, 0] + stimuli[:, 1],
                            stimuli[:, 2] + stimuli[:, 3],
                            stimuli[:, 4] + stimuli[:, 5]]).T

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

    stimuli = [_read_stimulus(sub, run, path, n_scans, tr, two_classes, glm)
               for run in range(n_runs)]

    fmri = [_read_fmri(sub, run, path) for run in range(n_runs)]

    return fmri, stimuli


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

    fmri_train_window = [fmri_train[scan: scan + time_window].ravel()
                         for scan in xrange(n_scans_train - time_window)]
    fmri_test_window = [fmri_test[scan: scan + time_window].ravel()
                        for scan in xrange(n_scans_test - time_window)]

    stimuli_train_window, stimuli_test_window = (stimuli_train[: -time_window],
                                                 stimuli_test[: -time_window])

    return (fmri_train_window, fmri_test_window, stimuli_train_window,
            stimuli_test_window)


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
              n_iterations=1, classify=False, k=10000):
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


def classification_score(prediction, stimuli):
    """ Returns a classification score from a regressor by doing a softmax """
    # Restrain analysis to scans with stimuli (i.e. no 'rest' category)
    mask = np.sum(stimuli[:, 1:], axis=1).astype(bool)
    prediction, stimuli = np.array((prediction[mask], stimuli[mask]))
    classifier = np.array([[0, 1, 0]
                           if prediction[scan][1] > prediction[scan][2]
                           else [0, 0, 1]
                           for scan in range(prediction.shape[0])])
    score = metrics.accuracy_score(stimuli, classifier)

    return score


def glm(fmri, glm_stimuli, basis='hrf', mode='glm'):
    """ Fit a GLM for comparison with time decoding model """
    onsets = np.empty(64 * len(glm_stimuli))
    conditions = np.empty(64 * len(glm_stimuli))
    for run, stim in enumerate(glm_stimuli):
        conditions[64 * run: 64 * (run + 1)] = stim[0] + (run * 410)
        onsets[64 * run: 64 * (run + 1)] = stim[1]

    tr = 2.
    hrfs, betas = he.glm(conditions, onsets, tr, fmri, basis=basis, mode=mode)

    return hrfs, betas


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
