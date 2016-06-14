# Performs linear regression on a contrast function created from the
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

contrast_train = {}
contrast_test = {}
fmri_train = {}
fmri_test = {}
# Loop through all subjects
for i in range(6):
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[i], delimiter=" ")
    target = labels['labels']

    # Create faces and houses time series
    faces = [1 if x == 'face' else 0 for x in target]
    houses = [1 if x == 'house' else 0 for x in target]

    # Create contrast time series
    contrast = np.subtract(faces, houses)

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[i]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, sessions=sessions_id)
    func_filename = haxby_dataset.func[i]
    fmri = nifti_masker.fit_transform(func_filename)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        contrast_train[str(i)] = contrast[train_index]
        contrast_test[str(i)] = contrast[test_index]
        fmri_train[str(i)] = fmri[train_index]
        fmri_test[str(i)] = fmri[test_index]

# MODEL

sns.set_style('darkgrid')
f, axes = plt.subplots(3, 2)
for i in range(6):
    # Fit linear model
    clf = linear_model.LinearRegression()
    clf.fit(fmri_train[str(i)], contrast_train[str(i)])

    # TEST
    prediction = clf.predict(fmri_test[str(i)])

    # Plot
    axes[i % 3, i/3].plot(range(len(prediction)), contrast_test[str(i)])
    axes[i % 3, i/3].plot(range(len(prediction)), prediction)
    # Print train score
    axes[i % 3, i/3].set_title('Subject %(subject)d, score %(score).2f'
        % {
            'subject': i,
            'score': clf.score(fmri_test[str(i)], contrast_test[str(i)])
          }
         )
plt.show()
