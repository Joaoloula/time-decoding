from nilearn.image import load_img
from nilearn import input_data
from sklearn import linear_model, metrics, manifold, decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _read_fmri(sub, run, path):
    """ Reads BOLD signal data """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    fmri_path = (('{path}/ses-test/func/ra{sub}_ses-test_{task}'
                  '_{run}_bold.nii')
                 .format(path=path, sub=sub, task=task, run=run))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimulus(sub, run, path, n_scans, tr, two_classes):
    """ Reads stimulus data """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    onsets_path = (('{path}/ses-test/func/{sub}_ses-test_{task}'
                    '_{run}_events.tsv')
                   .format(path=path, sub=sub, task=task, run=run))

    paradigm = pd.read_csv(onsets_path, sep='\t')
    onsets, durations, conditions = (paradigm['onset'], paradigm['duration'],
                                     paradigm['trial_type'])

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
    path='/home/loula/Programming/python/neurospin/ds006_R2.0.0/results/'):
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

    stimuli = [_read_stimulus(sub, run, path, n_scans, tr, two_classes)
               for run in range(n_runs)]
    fmri = [_read_fmri(sub, run, path) for run in range(n_runs)]

    return fmri, stimuli


def apply_time_window(fmri, stimuli, time_window=8):
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
    n_scans, n_voxels = np.shape(fmri)

    fmri_window = [fmri[scan: scan + time_window].ravel()
                   for scan in xrange(n_scans - time_window)]

    stimuli_window = stimuli[: -time_window]

    return (fmri_window, stimuli_window)


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


def _create_kernel(length, time_window=3):
    """ Creates a kernel matrix and its inverse for RKHS """

    k_block = [[1./3, 1./3, 1./3], [-1, 1, 0], [0, 1, -1]]
    k = np.kron(np.eye(length), k_block)

    inv_k_block = np.linalg.pinv(k_block)
    inv_k = np.kron(np.eye(length), inv_k_block)

    return k, inv_k


def fit_ridge_regression(fmri_train, fmri_test, stimuli_train, stimuli_test,
                         alphas=[0.01, 0.1, 1, 10, 100], k=10000, kernel=False):
    """ Fits a ridge classification to the data.

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

    alphas: array
        alpha coefficients to test for hyperparameter selection

    k: int
        number of features to select on anova

    kernel: bool
        whether or not to use an RKHS regression

    Returns
    -------

    prediction: array of shape [n_scans_test]
        prediction for the regression

    scores: array of shape [n_categories]
        r2 scores for the prediction

    """

    anova = SelectKBest(f_classif, k=k)
    ridge = linear_model.RidgeCV(alphas=alphas)

    # Transform one-hot to int encoding
    new_stimuli_train = np.zeros(stimuli_train.shape[0])
    for scan in range(len(new_stimuli_train)):
        new_stimuli_train[scan] = np.argmax(stimuli_train[scan])

    new_stimuli_test = np.zeros(stimuli_test.shape[0])
    for scan in range(len(new_stimuli_test)):
        new_stimuli_test[scan] = np.argmax(stimuli_test[scan])

    if kernel:
        # Fit RKHS model
        train_scans, test_scans = len(fmri_train), len(fmri_test)
        train_kernel, inv_train_kernel = _create_kernel(train_scans)
        test_kernel, inv_test_kernel = _create_kernel(test_scans)
        fmri_train = np.array(fmri_train).reshape(train_scans * 3, -1,
                                                  order='F')
        fmri_test = np.array(fmri_test).reshape(test_scans * 3, -1, order='F')
        fmri_train = (np.dot(fmri_train.T, inv_train_kernel)
                      .T.reshape(train_scans, -1))
        fmri_test = (np.dot(fmri_test.T, inv_test_kernel)
                     .T.reshape(test_scans, -1))

    new_fmri_train = anova.fit_transform(fmri_train, new_stimuli_train)
    new_fmri_test = anova.transform(fmri_test)

    ridge.fit(new_fmri_train, stimuli_train[:, 1:])
    prediction = ridge.predict(new_fmri_test)
    scores = metrics.r2_score(stimuli_test[:, 1:], prediction,
                              multioutput='raw_values')

    return prediction, scores


def fit_ridge_classification(fmri_train, fmri_test, stimuli_train, stimuli_test,
                             alphas=[0.01, 0.1, 1, 10, 100], k=10000):
    """ Fits a ridge classification to the data.

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

    alphas: array
        alpha coefficients to test for hyperparameter selection

    k: int
        number of features to select on anova

    Returns
    -------

    prediction: array of shape [n_scans_test]
        prediction for the regression

    score: array of shape [n_categories]
        r2 scores for the prediction

    norest_score: array of shape [n_categories - 1]
        r2 scores for the prediction without the category 'rest'

    """
    anova = SelectKBest(f_classif, k=k)
    ridge = linear_model.RidgeClassifierCV(alphas=alphas)
    anova_ridge = Pipeline([('anova', anova), ('ridge', ridge)])

    # Transform one-hot to int encoding
    new_stimuli_train = np.zeros(stimuli_train.shape[0])
    for scan in range(len(new_stimuli_train)):
        new_stimuli_train[scan] = np.argmax(stimuli_train[scan])

    new_stimuli_test = np.zeros(stimuli_test.shape[0])
    for scan in range(len(new_stimuli_test)):
        new_stimuli_test[scan] = np.argmax(stimuli_test[scan])

    anova_ridge.fit(fmri_train, new_stimuli_train)
    prediction = anova_ridge.predict(fmri_test)
    probas = anova_ridge.decision_function(fmri_test)
    score = anova_ridge.score(fmri_test, new_stimuli_test)

    # Calculate score without class 'rest'
    mask_test = stimuli_test[:, 0] != 1
    new_stimuli_test = new_stimuli_test[mask_test]
    prediction_no_rest = np.add(np.argmax(probas[:, 1:], axis=1), 1)[mask_test]
    norest_score = metrics.accuracy_score(new_stimuli_test, prediction_no_rest)

    return prediction, score, norest_score


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


def classification_score(ridge_prediction, stimuli):
    """ Returns a classification score from a regressor by doing a softmax """
    # Restrain analysis to scans with stimuli (i.e. no 'rest' category)
    mask = np.sum(stimuli, axis=1).astype(bool)
    ridge_prediction, stimuli = ridge_prediction[mask], stimuli[mask]
    ridge_classifier = np.zeros_like(ridge_prediction)
    for scan in range(ridge_prediction.shape[0]):
        ridge_classifier[scan][np.argmax(ridge_prediction[scan])] = 1
    score = metrics.accuracy_score(stimuli, ridge_classifier)

    return score


def plot(prediction, stimuli, scores, time_window=8, two_classes=False):
    """ Plots predictions and ground truths for each of the classes, as well
    as their r2 scores. """
    plt.style.use('ggplot')
    if two_classes:
        fig, axes = plt.subplots(2)
        fig.suptitle(('Ridge predictions for \'plain\' vs. \'reversed\','
                      'time window of {tw}')
                     .format(tw=time_window), fontsize=20)
        axes[0].plot(stimuli[:, 0])
        axes[0].plot(prediction[:, 0])
        axes[0].set_title(('Predictions for category \'plain\', r2 score of'
                           '{score}').format(score=scores[0]), fontsize=18)
        axes[1].plot(stimuli[:, 1])
        axes[1].plot(prediction[:, 1])
        axes[1].set_title(('Predictions for category \'reversed\', r2 score of'
                           '{score}').format(score=scores[1]), fontsize=18)
    else:
        cats = np.array(['junk', 'pl_ns', 'pl_sw', 'mr_ns', 'mr_sw'])
        fig, axes = plt.subplots(5)
        fig.suptitle('Ridge predictions for all classes, time window of {tw}'
                     .format(tw=time_window), fontsize=20)
        for cat in range(len(cats)):
            axes[cat].plot(stimuli[:, cat])
            axes[cat].plot(prediction[:, cat])
            axes[cat].set_title(('Prediction for category {cat}, R2 score of '
                                 '{score}').format(cat=cats[cat],
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
