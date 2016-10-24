from time_decoding.data_reading import read_data_haxby
from nilearn.plotting import plot_stat_map, show
from nilearn import datasets
import time_decoding.decoding as de
import numpy as np

haxby_dataset = datasets.fetch_haxby()

# Parameters
tr = 2.5
model = 'GLM'
k = 10000

# GLM parameters
hrf_model = 'spm'

fmri, stimuli, onsets, conditions, masker = read_data_haxby(2, masker=True)
session_id_onset = [[session] * len(onsets[session])
					for session in range(len(onsets))]
durations = [[24] * 8] * len(fmri)
betas, reg = de.glm(fmri, tr, onsets, durations=durations,
					hrf_model=hrf_model, model=model)

betas = np.vstack(betas)
session_id_onset = np.hstack(session_id_onset)
conditions = np.hstack(conditions)

train_index = np.where(session_id_onset != 6)
test_index = np.where(session_id_onset == 6)
# Split into train and test sets
betas_train, betas_test = (betas[train_index], betas[test_index])
conditions_train, conditions_test = (conditions[train_index],
                                     conditions[test_index])

"""
# Feature selection
betas_train, betas_test, anova = de.feature_selection(
    betas_train, betas_test, conditions_train, k=k,
    selector=True)

betas_test = anova.inverse_transform(betas_test[0])
"""
coef_img = masker.inverse_transform(betas_test[0])
coef_map = coef_img.get_data()
# threshold = np.max(np.abs(coef_map)) * 0.05
plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0],
              display_mode='z', cut_coords=1,
              title=model+" weights")

show()
