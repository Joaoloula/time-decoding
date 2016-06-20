# Performs multinomial logistic regression on activation data created from the
# Haxby dataset
# Score : 0.60 accuracy for 8 classes
from sklearn.cross_validation import LeavePLabelOut
from nilearn.input_data import NiftiMasker
from nistats import hemodynamic_models
from sklearn import linear_model
from sklearn import metrics
from nilearn import datasets
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PARAMETERS
tr = 2.5
n_scans = 1452
n_sessions = 12
n_subjects = 6
oversample = 1  # Only handles oversample = 1 for the moment
plot_subject = 5  # Subject to plot
gap = 3  # Number of scans on the gap between fmri data and stimuli on Haxby
hrf_model = 'glover'

# Train and test parameters
lplo_p = 2  # Number of labels to leave out for cross validation
test_size = (n_scans/n_sessions) * (lplo_p)
train_size = n_scans - test_size

# Calculate frame times
frame_times = np.arange(n_scans) * tr

# Calculate the toeplitz matrix for the discrete convolution by the HRF
hrf_filter = hemodynamic_models.glover_hrf(tr, oversampling=oversample)
hrf_filter_train = [hrf_filter[i] if i < len(hrf_filter) else 0
                    for i in range(train_size-gap)]
hrf_filter_test = [hrf_filter[i] if i < len(hrf_filter) else 0
                   for i in range(test_size-gap)]
hrf_matrix_train = linalg.toeplitz(hrf_filter_train, np.zeros(train_size-gap))
hrf_matrix_test = linalg.toeplitz(hrf_filter_test, np.zeros(test_size-gap))

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# Create sessions id
sessions_id = [x/(n_scans/n_sessions) for x in range(n_scans)]


def read_data(subject):
    """Returns indiviudal images, labels and session id for subject subject"""
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[subject],
                           delimiter=" ")
    sessions_id = labels['chunks']
    target = labels['labels']
    categories = np.unique(target)
    # Make 'rest' be the first category
    categories = np.roll(categories,
                         len(categories) - np.where(categories == 'rest')[0])

    # Initialize series array
    series_ = np.zeros(n_scans)
    for c, category in enumerate(categories):
        series_[target == category] = c

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[subject]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, sessions=sessions_id)
    func_filename = haxby_dataset.func[subject]
    # fmri[str(subject)] = nifti_masker.fit_transform(func_filename)
    # series[str(subject)] = series_
    return (nifti_masker.fit_transform(func_filename), series_,
            sessions_id, categories)

# MODEL

# Create Leave P Label Out cross validation
lplo = LeavePLabelOut(sessions_id, p=lplo_p)

# Initialize mean score and mean_hrf_score
mean_score = 0.
mean_deconv_score = 0.

sns.set_style('darkgrid')
figure, axes = plt.subplots(4, 2)
for subject in range(n_subjects):
    # Read data and remove 'rest' category
    fmri, series, sessions_id, categories = read_data(subject)

    # Create lists to store all hrf predictions of a subject
    all_hrf_train = []
    all_hrf_test = []

    for category in range(len(categories)):
        # Create experimental conditions
        onsets = [scan*tr for scan in range(1, len(series))
                  if series[scan] == category and series[scan-1] != category]
        durations = np.zeros(len(onsets))
        durations.fill(9*tr)  # All stimulus last 9 scans
        amplitudes = np.zeros(len(onsets))
        amplitudes.fill(1)  # Normalized amplitude
        exp_conditions = np.vstack((onsets, durations, amplitudes))
        # Compute hrf signal
        signal, _ = hemodynamic_models.compute_regressor(exp_conditions,
                                                         hrf_model,
                                                         frame_times,
                                                         oversampling=oversample
                                                         )
        for train_index, test_index in lplo:
            # Separate data into train and test sets
            fmri_train = fmri[train_index]
            fmri_test = fmri[test_index]
            series_train = series[train_index]
            series_test = series[test_index]
            signal_train = signal[train_index]
            signal_test = signal[test_index]
            # Do only one CV step for faster prototyping

        # Fit multinomial logistic regression
        # We choose the best C between Cs values on a logarithmic scale
        # between 1e-4 and 1e4
        # We have to remember to correct for the gap in the Haxby dataset
        ridge = linear_model.RidgeCV()
        ridge.fit(fmri_train[:-gap], signal_train[gap:])

        # HRF function prediction
        hrf_prediction_train = ridge.predict(fmri_train[:-gap])
        hrf_prediction_test = ridge.predict(fmri_test[:-gap])

        # Deconvolve the hrf signals to obtain an approximation of the original
        # stimulus function
        ridge.fit(hrf_matrix_train, hrf_prediction_train)
        hrf_prediction_train = ridge.coef_[0]
        ridge.fit(hrf_matrix_test, hrf_prediction_test)
        hrf_prediction_test = ridge.coef_[0]
        all_hrf_train.append(hrf_prediction_train)
        all_hrf_test.append(hrf_prediction_test)

    all_hrf_train = np.asarray(all_hrf_train).T
    all_hrf_test = np.asarray(all_hrf_test).T

    # Fit the ridge classifier
    ridge_class = linear_model.RidgeClassifierCV()
    ridge_class.fit(all_hrf_train, series_train[gap:])

    # Make prediction for the test set and calculate score
    class_prediction = ridge_class.predict(all_hrf_test)
    score = ridge_class.score(all_hrf_test, series_test[gap:])

    mean_score += score

    # PLOT
    # For better visualization, it's interesting to project both the stimuli
    # and the regressed probabilities on only one of the classes and then
    # plot the result

    if subject == plot_subject:
        for category in range(1, len(categories)):
            # Filter only the face stimuli and regression
            cat_stimuli = [int(x == category) for x in series_test[gap:]]

            r2_score = metrics.r2_score(cat_stimuli,
                                        all_hrf_test[:, category])
            x, y = (category - 1) % 4, (category - 1) / 4
            axes[x, y].plot(all_hrf_test[:, category])
            axes[x, y].plot(cat_stimuli)
            axes[x, y].set_title('Category %(cat)s, R2 score %(r2score).2f: '
                                 % {'cat': categories[category],
                                    'r2score': r2_score}
                                 )

mean_score = mean_score/n_subjects
figure.suptitle('Stimuli and predictions for subject %(subject)d: '
                % {'subject': plot_subject} +
                'total accuracy %(total_score).2f'
                % {'total_score': mean_score})

plt.show()
