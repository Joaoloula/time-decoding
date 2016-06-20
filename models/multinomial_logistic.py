# Performs multinomial logistic regression on activation data created from the
# Haxby dataset
from sklearn.cross_validation import LeavePLabelOut
from nilearn.input_data import NiftiMasker
from sklearn import linear_model
from sklearn import metrics
from nilearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PARAMETERS
n_scans = 1452
n_sessions = 12
n_c = 5  # number of Cs to use in logistic regression CV
n_jobs = 2  # number of jobs to use in logistic regression CV
n_subjects = 6

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# Create sessions id
sessions_id = [x / (n_scans / n_sessions) for x in range(n_scans)]

# Initialize series and fmri dictionaries
series = {}
fmri = {}


def read_data(subject):
    """Returns indiviudal images, labels and session id for subject subject"""
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[subject],
                           delimiter=" ")
    sessions_id = labels['chunks']
    target = labels['labels']
    categories = np.unique(target)

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
# Initialize train and test sets
series_train = {}
series_test = {}
fmri_train = {}
fmri_test = {}

# Initialize mean score and score counter
mean_score = 0.
score_count = 0


sns.set_style('darkgrid')
f, axes = plt.subplots(3, 3)
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = read_data(subject)

    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        series_train = series[train_index]
        series_test = series[test_index]
        fmri_train = fmri[train_index]
        fmri_test = fmri[test_index]

        # Fit multinomial logistic regression
        # We choose the best C between Cs values on a logarithmic scale
        # between 1e-4 and 1e4
        log = linear_model.LogisticRegressionCV(Cs=n_c, n_jobs=n_jobs)
        log.fit(fmri_train, series_train)

        # SCORE
        mean_score += log.score(fmri_test, series_test)

        # TEST
        prediction = log.predict(fmri_test)
        prediction_proba = log.predict_proba(fmri_test)

        # PLOT
        for k in range(9):
            # Make array with only face trials
            cat_stimuli = [int(x == k) for x in series_test]
            # Plot it along with the probability prediction for the face label
            axes[k % 3, k / 3].plot(cat_stimuli)
            axes[k % 3, k / 3].plot(prediction_proba[:, k])

            # Calculate R2 score for stimulus approximation
            r2_score = metrics.r2_score(cat_stimuli, prediction_proba[:, k])

            # Add subject number and train score to title
            axes[k % 3, k / 3].set_title('Category %(cat)s, '
                                         % {'cat': categories[k]} +
                                         'R2 score %(score).2f'
                                         % {'score': r2_score}
                                         )

        # Update score counter
        score_count += 1

# Calculate and print the mean score
mean_score = mean_score / score_count
f.suptitle('Predictions and R2 scores for subject 5, accuracy = %.2f'
           % mean_score)
print("The accuracay is %.4f" % mean_score)

plt.show()
