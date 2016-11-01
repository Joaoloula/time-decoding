from time_decoding.data_reading import read_data_haxby
from nilearn.plotting import plot_stat_map, show
from sklearn.linear_model import RidgeCV
from nilearn import datasets
import time_decoding.decoding as de
import matplotlib.pyplot as plt
import numpy as np

haxby_dataset = datasets.fetch_haxby()

# Parameters
k = 10000
tr = 2.5
model = 'Logistic Deconvolution'

# Ridge parameters
hrf_model = 'spm'
time_window = 10
delay = 0

fmri, stimuli, onsets, conditions, masker = read_data_haxby(2, masker=True)
session_id_fmri = [[session] * len(fmri[session])
                   for session in range(len(fmri))]

durations = [[24] * 8] * len(fmri)
design = [de.design_matrix(len(fmri[session]), tr, onsets[session],
                           conditions[session], hrf_model=hrf_model,
                           durations=durations[session])
          for session in range(len(fmri))]
fmri, stimuli, session_id_fmri = (fmri, stimuli,
                                  np.array(session_id_fmri))
# Stack the BOLD signals and the design matrices
fmri = np.vstack(fmri)
design = np.vstack(design)
stimuli = np.vstack(stimuli)
session_id_fmri = np.hstack(session_id_fmri)

"""
mask = np.logical_or(conditions == 'face', conditions == 'house')
fmri = fmri[mask]
session_id_fmri = session_id_fmri[mask]
conditions = conditions[mask]
"""

train_index = np.where(session_id_fmri != 6)
test_index = np.where(session_id_fmri == 6)
# Split into train and test sets
fmri_train, fmri_test = (fmri[train_index], fmri[test_index])
design_train, design_test = design[train_index], design[test_index]
stimuli_train, stimuli_test = stimuli[train_index], stimuli[test_index]

ridge = RidgeCV()
ridge.fit(fmri_train, design_train)
prediction = ridge.predict(fmri_test)
ridge_coef = - ridge.coef_[3] + ridge.coef_[4]  # 'face' vs. 'house'
coef_img = masker.inverse_transform(ridge_coef)
coef_map = coef_img.get_data()
threshold = np.percentile(np.abs(coef_map), 98)

# Plot stat map
plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0],
              display_mode='z', cut_coords=[-1],
              title=model+" weights")
"""
# Plot time-series
onset = int(onsets[6][3]/tr)
time_series = prediction[onset + delay: onset + delay + tr, 1]
plt.plot(time_series)
"""
show()
