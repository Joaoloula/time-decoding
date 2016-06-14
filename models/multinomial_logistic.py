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

series = {}
fmris = {}
# Loop through all subjects
for i in range(6):
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[1], delimiter=" ")
    target = labels['labels']

    # Create labeled time series
    faces = [ 1 if x == 'face' else 0 for x in target]
    houses = [ 2 if x == 'house' else 0 for x in target]
    bottle = [ 3 if x == 'bottle' else 0 for x in target]
    chair = [ 4 if x == 'chair' else 0 for x in target]

    # Add series and put them in the dictionary
    series[str(i)] = np.add(np.add(faces, houses), np.add(bottle, chair))

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[i]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, smoothing_fwhm='fast')
    func_filename = haxby_dataset.func[i]
    fmris[str(i)] = nifti_masker.fit_transform(func_filename)

series_train = np.vstack((series['0'], series['1'], series['2'],
                          series['3'], series['4']))
fmris_train = np.vstack((fmris['0'], fmris['1'], fmris['2'], fmris['3'],
                         fmris['4']))
series_test = series['5']
fmris_test = fmris['5']

# MODEL

# Fit multinomial logistic regression
log = linear_model.LogisticRegression()
log.fit(fmri_masked, labels)

# TEST
prediction = log.predict(fmris_test)
prediction_proba = log.predict_proba(fmris_test)

# Show time-series
total_faces = [1 if x==1 else 0 for x in series] # Face time series
plt.plot(np.arange(0, 1452), total_faces)
plt.plot(np.arange(0, 1452), prediction_proba[:, 1])
plt.ylim([0, 2])
plt.title = ('Test fit')
plt.show()

# Print train score
log.score(fmris_test, contrasts_test)
