# Performs linear regression on a contrast function created from the
# Haxby dataset
from sklearn import linear_model
from nilearn import datasets
from nilearn.input_data import NiftiMasker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=6)

contrasts = {}
fmris = {}
# Loop through all subjects
for i in range(6):
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[1], delimiter=" ")
    target = labels['labels']

    # Create faces and houses time series
    faces = [1 if x == 'face' else 0 for x in target]
    houses = [1 if x == 'house' else 0 for x in target]

    # Create contrast time series
    contrasts[str(i)] = np.subtract(faces, houses)

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[i]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, smoothing_fwhm='fast')
    func_filename = haxby_dataset.func[i]
    fmris[str(i)] = nifti_masker.fit_transform(func_filename)

contrasts_train = np.hstack((contrasts['0'], contrasts['1'], contrasts['2'],
                             contrasts['3'], contrasts['4']))
fmris_train = np.hstack((fmris['0'], fmris['1'], fmris['2'], fmris['3'],
                         fmris['4']))
contrasts_test = contrasts['5']
fmris_test = fmris['5']

# MODEL

# Fit linear model
clf = linear_model.LinearRegression()
clf.fit(fmris_train, contrasts_train)

# TEST
prediction = clf.predict(fmris_test)

# Show time-series
plt.plot(np.arange(0, 1452), contrasts_test)
plt.plot(np.arange(0, 1452), prediction)
plt.ylim([-2, 2])
plt.title = ('Test fit')
plt.show()

# Print train score
clf.score(fmris_test, contrasts_test)
