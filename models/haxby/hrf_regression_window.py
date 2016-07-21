from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.cross_validation import LeavePLabelOut
from sklearn.metrics import accuracy_score
from nilearn import datasets
import helper_functions as hf
import matplotlib.pyplot as plt
import numpy as np

"""
Performs multinomial logistic regression on activation data created from the
Haxby dataset, using a custom time window
Accuracy: 0.89 with 8 categories

Author: Joao Loula
"""
print(__doc__)

# PARAMETERS
n_scans = 1452
n_c = 5  # number of Cs to use in logistic regression CV
n_subjects = 6
plot_subject = 99  # ID of the subject to plot
time_window = 3
cutoff = 0
delay = 1  # Correction of the fmri scans in relation to the stimuli
model = 'ridge'  # 'ridge' for Ridge CV, 'log' for logistic regression CV
classify = True

# RKHS parameters
penalty = 1
n_iterations = 1
kernel = 'voxel_weighing'

# PREPROCESSING
# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# MODEL
# Initialize mean score and score counter
if model == 'log':
    all_scores = np.zeros(n_subjects)

elif model == 'ridge':
    all_scores = np.zeros((n_subjects, 9))
    softmax_scores = np.zeros((n_subjects, 9))

plt.style.use('ggplot')
f, axes = plt.subplots(3, 3)
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = hf.read_data(subject, haxby_dataset,
                                                         whole_brain=True)
    paradigm = hf.create_paradigm(series, categories, tr=2.5)

    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        series_train = series[train_index]
        series_test = series[test_index]
        fmri_train = fmri[train_index]
        fmri_test = fmri[test_index]

        feature_selection = SelectPercentile(f_classif, percentile=10)
        fmri_train = feature_selection.fit_transform(fmri_train, series_train)
        fmri_test = feature_selection.transform(fmri_test)

        # Apply time window and time correction
        fmri_train, series_train, sessions_id = hf.apply_time_window(fmri_train,
            series_train, sessions_id, time_window=time_window, delay=delay)

        # Apply time window and time correction
        fmri_test, series_test, sessions_id = hf.apply_time_window(fmri_test,
            series_test, sessions_id, time_window=time_window, delay=delay)

        one_hot_train = hf.to_one_hot(series_train)
        one_hot_test = hf.to_one_hot(series_test)

        if model == 'log':
            prediction, prediction_proba, score = hf.fit_log(
                fmri_train, fmri_test, series_train, series_test, n_c=n_c)
            mask = series_test != 0
            score = accuracy_score(series_test[mask], prediction[mask])

        elif model == 'ridge':
            prediction, score = hf.fit_ridge(
                fmri_train, fmri_test, one_hot_train, one_hot_test,
                paradigm=paradigm, cutoff=cutoff, kernel=kernel,
                penalty=penalty, time_window=time_window,
                n_iterations=n_iterations, classify=classify)

        # PLOT
        if subject == plot_subject:
            for k in range(len(categories)):
                # Plot it along with the probability prediction
                x, y = k % 3, k / 3
                axes[x, y].plot(one_hot_test[:, k])
                axes[x, y].plot(prediction[:, k])
                axes[x, y].set_title(
                    'Category {cat}, score {score:.2f}'
                    .format(cat=categories[k], score=score[k]), fontsize=20)
                axes[x, y].axes.get_xaxis().set_visible(False)
                axes[x, y].axes.get_yaxis().set_visible(False)
                axes[x, y].set_xlim([0, len(prediction)])
                axes[x, y].set_ylim([-0.4, 1.2])

        all_scores[subject] = score
        break  # Only run one CV step per subject for fast prototyping

    print('processing subject ' + str(subject))

# Calculate and print the mean score
"""f.suptitle('Predictions and scores of %s model for subject %d, '
           % (model, plot_subject) +
           'time window of %d scans '
           % (time_window),
           # 'low-pass cutoff of %.2fs' % (cutoff),
           fontsize=20)"""

print(all_scores)
plt.show()
