# Performs multinomial logistic regression on activation data created from the
# Haxby dataset
from sklearn.cross_validation import LeavePLabelOut
from nilearn.input_data import NiftiMasker
from sklearn import linear_model
from nilearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PARAMETERS
n_scans = 1452
n_sessions = 12
n_c = 5  # number of Cs to use in logistic regression CV
n_jobs = 2  # number of jobs to use in logistic regression CV
n_subjects = 1

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# Create categories
categories = ['face', 'house', 'bottle', 'chair']

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
        series_[target == category] = c + 1

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[subject]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, sessions=sessions_id)
    func_filename = haxby_dataset.func[subject]
    # fmri[str(subject)] = nifti_masker.fit_transform(func_filename)
    # series[str(subject)] = series_
    return nifti_masker.fit_transform(func_filename), series_, sessions_id

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
f, axes = plt.subplots(3, 2)
for i in range(n_subjects):
    # Flag for fitting the first example for each subject
    first = True

    fmri, series, sessions_id = read_data(i)
    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        series_train[str(i)] = series[str(i)][train_index]
        series_test[str(i)] = series[str(i)][test_index]
        fmri_train[str(i)] = fmri[str(i)][train_index]
        fmri_test[str(i)] = fmri[str(i)][test_index]

        # Fit multinomial logistic regression
        # We choose the best C between Cs values on a logarithmic scale
        # between 1e-4 and 1e4
        log = linear_model.LogisticRegressionCV(Cs=n_c, n_jobs=n_jobs)
        log.fit(fmri_train[str(i)], series_train[str(i)])

        # SCORE
        mean_score += log.score(fmri_test[str(i)], series_test[str(i)])

        if first:
            # TEST
            prediction = log.predict(fmri_test[str(i)])
            prediction_proba = log.predict_proba(fmri_test[str(i)])

            # PLOT

            # Make array with only face trials
            faces = [1 if x == 1 else 0 for x in series_test[str(i)]]
            # Plot it along with the probability prediction for the face label
            axes[i % 3, i / 3].plot(range(len(prediction_proba)),
                                  faces)
            axes[i % 3, i / 3].plot(range(len(prediction_proba)),
                                  prediction_proba[:, 1])
            # Add subject number and train score to title
            axes[i % 3, i / 3].set_title(
                'Subject %(subject)d, score %(score).2f'
                % {'subject': i,
                   'score': log.score(fmri_test[str(i)], series_test[str(i)])}
            )

            first = False

        # Update score counter
        score_count += 1

# Calculate and print the mean score
mean_score = mean_score / score_count
print("The accuracy is %.4f" % mean_score)

plt.show()
