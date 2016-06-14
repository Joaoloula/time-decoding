# Performs multinomial logistic regression on activation data created from the
# Haxby dataset
from sklearn.cross_validation import LeavePLabelOut
from nilearn.input_data import NiftiMasker
from sklearn import linear_model
from nilearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=6)

# Create sessions id
sessions_id = [x/121 for x in range(1452)]

# Initialize Leave P Label Out cross validation
lplo = LeavePLabelOut(sessions_id, p=2)

series_train = {}
series_test = {}
fmri_train = {}
fmri_test = {}
# Loop through all subjects
for i in range(6):
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[i], delimiter=" ")
    target = labels['labels']

    # Create labeled time series
    faces = [1 if x == 'face' else 0 for x in target]
    houses = [2 if x == 'house' else 0 for x in target]
    bottle = [3 if x == 'bottle' else 0 for x in target]
    chair = [4 if x == 'chair' else 0 for x in target]

    # Add series and put them in the dictionary
    series = np.add(np.add(faces, houses), np.add(bottle, chair))

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[i]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, smoothing_fwhm='fast')
    func_filename = haxby_dataset.func[i]
    fmri = nifti_masker.fit_transform(func_filename)

    for train_index, test_index in lplo:
        series_train[str(i)] = series[train_index]
        series_test[str(i)] = series[test_index]
        fmri_train[str(i)] = fmri[train_index]
        fmri_test[str(i)] = fmri[test_index]

# MODEL

sns.set_style('darkgrid')
f, axes = plt.subplots(3, 2)
for i in range(6):
    # Fit multinomial logistic regression
    log = linear_model.LogisticRegressionCV(Cs=5, n_jobs=2)
    log.fit(fmri_train[str(i)], series_train[str(i)])

    # TEST
    prediction = log.predict(fmri_test[str(i)])
    prediction_proba = log.predict_proba(fmri_test[str(i)])

    # PLOT

    # Make array with only face trials
    faces = [1 if x == 1 else 0 for x in series_test[str(i)]]
    # Plot it along with the probability prediction for the face label
    axes[i % 3, i/3].plot(range(len(prediction_proba)),
                          faces)
    axes[i % 3, i/3].plot(range(len(prediction_proba)),
                          prediction_proba[:, 1])
    # Add subject number and train score to title
    axes[i % 3, i/3].set_title('Subject %(subject)d, score %(score).2f'
        % {
           'subject': i,
           'score': log.score(fmri_test[str(i)], series_test[str(i)])
           }
        )
plt.show()
