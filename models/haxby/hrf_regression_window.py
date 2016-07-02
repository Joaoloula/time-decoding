# Performs multinomial logistic regression on activation data created from the
# Haxby dataset, using a custom time window
# Accuracy: 0.89 with 8 categories
# from hrf_estimation.savitzky_golay import savgol_filter
from sklearn.cross_validation import LeavePLabelOut
from nilearn import datasets
import helper_functions as hf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PARAMETERS
n_scans = 1452
n_c = 5  # number of Cs to use in logistic regression CV
n_subjects = 1
plot_subject = 99  # ID of the subject to plot
time_window = 1
cutoff = 0
delay = 0  # Correction of the fmri scans in relation to the stimuli
model = 'ridge'  # 'ridge' for Ridge CV, 'log' for logistic regression CV

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

sns.set_style('darkgrid')
f, axes = plt.subplots(3, 3)
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = hf.read_data(subject, haxby_dataset)
    # Apply time window and time correction
    fmri, series, sessions_id = hf.apply_time_window(fmri, series, sessions_id,
                                                     time_window=time_window,
                                                     delay=delay)

    paradigm = hf.create_paradigm(series, categories, tr=2.5)

    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        series_train = series[train_index]
        series_test = series[test_index]
        fmri_train = fmri[train_index]
        fmri_test = fmri[test_index]

        one_hot_train = hf.to_one_hot(series_train)
        one_hot_test = hf.to_one_hot(series_test)

        if model == 'log':
            prediction, prediction_proba, score = hf.fit_log(
                fmri_train, fmri_test, series_train, series_test, n_c=4)

        elif model == 'ridge':
            prediction, score = hf.fit_ridge(fmri_train, fmri_test,
                                             one_hot_train, one_hot_test,
                                             paradigm=paradigm, cutoff=cutoff,
                                             n_alpha=n_c)

        # PLOT
        if subject == plot_subject:
            for k in range(len(categories)):
                # Plot it along with the probability prediction
                x, y = k % 3, k / 3
                axes[x, y].plot(one_hot_test[:, k])
                axes[x, y].plot(prediction[:, k])
                axes[x, y].set_title('Category {cat}, score {score:.2f}'
                                     .format(cat=categories[k], score=score[k]),
                                     fontsize=16)
                axes[x, y].axes.get_xaxis().set_visible(False)
                axes[x, y].axes.get_yaxis().set_visible(False)
                axes[x, y].set_xlim([0, len(prediction)])
                axes[x, y].set_ylim([-0.4, 1.2])

        all_scores[subject] = score
        break  # Only run one CV step per subject for fast prototyping

    print('processing subject ' + str(subject))

# Calculate and print the mean score
f.suptitle('Predictions and scores of %s model for subject %d, '
           % (model, plot_subject) +
           'time window of %d scans '
           % (time_window),
           # 'low-pass cutoff of %.2fs' % (cutoff),
           fontsize=20)

print(all_scores)
plt.show()
