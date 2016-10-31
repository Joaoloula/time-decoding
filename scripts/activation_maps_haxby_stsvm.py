from time_decoding.data_reading import read_data_haxby
from nilearn.plotting import plot_stat_map, show
from sklearn.svm import SVC
from nilearn import datasets
import time_decoding.decoding as de
import numpy as np

haxby_dataset = datasets.fetch_haxby()

# Parameters
k = 10000
tr = 2.5
model = 'Spatiotemporal SVM'

# SVM parameters
time_window = 10
delay = 0

fmri, stimuli, onsets, conditions, masker = read_data_haxby(2, masker=True)
session_id_onset = [[session] * len(onsets[session])
                    for session in range(len(onsets))]
fmri_windows = de.apply_time_window(fmri, stimuli, time_window, delay)

fmri_windows = np.vstack(fmri_windows)
session_id_onset = np.hstack(session_id_onset)
conditions = np.hstack(conditions)

mask = np.logical_or(conditions == 'face', conditions == 'house')
fmri_windows = fmri_windows[mask]
session_id_onset = session_id_onset[mask]
conditions = conditions[mask]

train_index = np.where(session_id_onset != 6)
test_index = np.where(session_id_onset == 6)
# Split into train and test sets
fmri_windows_train, fmri_windows_test = (fmri_windows[train_index],
                                         fmri_windows[test_index])
conditions_train, conditions_test = (conditions[train_index],
                                     conditions[test_index])

"""
# Feature selection
fmri_windows_train, fmri_windows_test, anova = de.feature_selection(
    fmri_windows_train, fmri_windows_test, conditions_train, k=k,
    selector=True)
"""

svc = SVC(kernel='linear')
svc.fit(fmri_windows_train, conditions_train)
svc_coef = svc.coef_[0]  # 'face' vs. 'house'
# svc_coef = anova.inverse_transform(svc_coef)
coef_img = masker.inverse_transform(svc_coef.reshape(10, -1)[1])
coef_map = coef_img.get_data()
threshold = np.percentile(np.abs(coef_map), 98)
plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0],
              display_mode='z',cut_coords=[-5],
              title=model+" weights", threshold=threshold)

show()
