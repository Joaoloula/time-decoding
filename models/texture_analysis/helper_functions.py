from nilearn.image import load_img
from nilearn import input_data
from sklearn import linear_model, metrics, manifold, decomposition
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _read_fmri(sub, run, path, task):
    """ Reads BOLD signal data """
    run = sub + '_' + str(run + 1)
    fmri_path = (('{path}{run}/fmri/cra{run}_td{task}.nii')
                 .format(path=path, task=task+1, run=run))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimuli(stimuli_path, n_tasks=6, tr=2.4):
    """ Reads stimuli for the texture dataset and returns a numpy array of
    shape [n_scans, n_classes] """
    session_stimuli_zero = np.load(stimuli_path + '0.npy')
    session_stimuli_one = np.load(stimuli_path + '1.npy')

    classes = np.array(['rest', '01', '09', '12', '13', '14', '25'])
    n_scans = 184 * n_tasks
    stimuli = np.zeros((n_scans, len(classes)))
    for task in range(n_tasks):
        for stimulus in range(len(session_stimuli_zero[task])):
            zero_class = np.where(
                classes == session_stimuli_zero[task][stimulus][:2])[0][0]
            one_class = np.where(
                classes == session_stimuli_one[task][stimulus][:2])[0][0]
            zero_index = int(round(((36*12*task) + (12*stimulus)) / tr))
            one_index = int(round(((36*12*task) + (12*stimulus) + 4) / tr))

            # Add one_hot vector to respective locations
            stimuli[zero_index][zero_class] = 1
            stimuli[one_index][one_class] = 1

    # Fill the rest with category 'rest' 
    rest_scans = np.where(np.sum(stimuli, axis=1) == 0)
    stimuli[rest_scans, 0] = 1
   
    return stimuli


def read_data(subject, n_runs=2, n_tasks=6, tr=2.4, n_scans=205,
    path='/home/parietal/eickenbe/workspace/data/TextureAnalysis/'):
    """
    Reads data from the Texture dataset.

    Parameters
    ----------

    subject: int from 0 to 3
        subject from which to read the data

    n_runs: int from 1 to 3
        number of runs to read from the subject, defaults to 2

    n_tasks: int from 1 to 6
        number of tasks to read from the subject, defaults to 6

    tr: float
        repetition time for the task (defaults to 2.4)

    n_scans: int
        number of scans per run, defaults to 184

    Returns
    -------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    stimuli: numpy array of shape [n_scans, n_classes]
        labels for the stimuli in one-hot encoding

    """
    stimuli = [_read_stimuli(path+'stimuli/stim_seq')
               for session in range(n_runs)]

    subject_list = ['pf120155', 'ns110383', 'ap100009', 'pb120360']
    sub = subject_list[subject]
    path += sub + '/'
    fmri = [_read_fmri(sub, run, path, task) for run in range(n_runs)
            for task in range(n_tasks)]

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


def uniform_masking(fmri_list, subject=0, high_pass=0.01, smoothing=5, n_runs=3, n_tasks=6):
    """ Mask all the sessions uniformly, doing standardization, linear
    detrending, DCT high_pas filtering and gaussian smoothing.

    Parameters
    ----------

    fmri_list: array-like
        array containing multiple BOLD data from different sessions

    subject: int from 0 to 3
        subject from which to read fmri data in case of opting to open pickle
        file. Defaults to 0

    high_pass: float
        frequency at which to apply the high pass filter, defaults to 0.01

    smoothing: float
        spatial scale of the gaussian smoothing filter in mm, defaults to 5

    Returns
    -------

    fmri_list_masked: array-like
        array containing the masked data

    """
    # Initialize the masker
    masker = input_data.MultiNiftiMasker(mask_strategy='epi', standardize=True,
                                         detrend=True, high_pass=0.01, t_r=2.4,
                                         smoothing_fwhm=smoothing)

    # Fit the masker for each run independently
    masker.fit(fmri_list[: n_tasks])
    fmri_list_masked = np.vstack([
        np.vstack(masker.transform(fmri_list[run * n_tasks: (run + 1) * n_tasks]))
        for run in range(n_runs)])

    return fmri_list_masked


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
              n_alpha=5, kernel=None, penalty=10, time_window=3,
              n_iterations=1, classify=False):
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

    if classify:
        mask = np.logical_not(one_hot_test[:, 0])
        class_regression, class_one_hot = prediction[mask], one_hot_test[mask]
        class_prediction = np.zeros_like(class_regression)
        for scan in range(class_regression.shape[0]):
            class_prediction[scan][np.argmax(class_regression[scan])] = 1
        class_score = metrics.accuracy_score(class_one_hot, class_prediction)
        return prediction, class_score

    # Score
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
