# Performs linear regression on a contrast function created from the
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

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=6)

# Create sessions id
sessions_id = [x/(n_scans/n_sessions) for x in range(n_scans)]

# Initialize contrast and fmri dictionaries
contrast = {}
fmri = {}

# Loop through all subjects
for i in range(6):
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[i], delimiter=" ")
    target = labels['labels']

    # Create faces and houses time series
    faces = [1 if x == 'face' else 0 for x in target]
    houses = [1 if x == 'house' else 0 for x in target]

    # Create contrast time series
    contrast[str(i)] = np.subtract(faces, houses)

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[i]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, sessions=sessions_id)
    func_filename = haxby_dataset.func[i]
    fmri[str(i)] = nifti_masker.fit_transform(func_filename)

# MODEL

# Initialize Leave P Label Out cross validation
lplo = LeavePLabelOut(sessions_id, p=2)

# Initialize train and test sets
contrast_train = {}
contrast_test = {}
fmri_train = {}
fmri_test = {}

# Initialize mean score and score counter
mean_score = 0.
score_count = 0


sns.set_style('darkgrid')
f, axes = plt.subplots(3, 2)
for i in range(6):
    # Flag for plotting the first example for each subject
    first = True

    # Divide in train and test sets
    for train_index, test_index in lplo:
        contrast_train[str(i)] = contrast[str(i)][train_index]
        contrast_test[str(i)] = contrast[str(i)][test_index]
        fmri_train[str(i)] = fmri[str(i)][train_index]
        fmri_test[str(i)] = fmri[str(i)][test_index]

        # Fit ridge regression with cross-validation
        ridge = linear_model.RidgeCV()
        ridge.fit(fmri_train[str(i)], contrast_train[str(i)])

        # SCORE
        mean_score += ridge.score(fmri_test[str(i)], contrast_test[str(i)])

        if first:
            # TEST
            prediction = ridge.predict(fmri_test[str(i)])

            # PLOT
            axes[i % 3, i/3].plot(range(len(prediction)), contrast_test[str(i)])
            axes[i % 3, i/3].plot(range(len(prediction)), prediction)
            # Print train score
            axes[i % 3, i/3].set_title('Subject %(subject)d, score %(score).2f'
                % {'subject': i,
                   'score': ridge.score(fmri_test[str(i)],
                                        contrast_test[str(i)])}
            )

            first = False

        # Update score counter
        score_count += 1

# Calculate and print the mean score
mean_score = mean_score/score_count
print("The mean R2 score is %.4f" % mean_score)

plt.show()
