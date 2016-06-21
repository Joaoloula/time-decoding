# Performs multinomial logistic regression on activation data created from the
# Haxby dataset, using a custom time window
# Accuracy: 0.89 with 8 categories
from sklearn.cross_validation import LeavePLabelOut
from sklearn import linear_model
from sklearn import metrics
from nilearn import datasets
from helper_functions import read_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PARAMETERS
n_scans = 1452
n_sessions = 12
n_c = 5  # number of Cs to use in logistic regression CV
n_jobs = 2  # number of jobs to use in logistic regression CV
n_subjects = 1
plot_subject = 5  # ID of the subject to plot
time_window = 5

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# Initialize series and fmri dictionaries
series = {}
fmri = {}

# MODEL

# Initialize mean score and score counter
subject_accuracies = np.zeros(n_subjects)
categories_r2_scores = np.zeros((n_subjects, 9))
count = 1

sns.set_style('darkgrid')
f, axes = plt.subplots(3, 3)
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = read_data(subject, haxby_dataset,
                                                      n_scans)
    # Apply a time window of 'time_window'
    fmri_window = np.asarray([fmri[scan: scan + time_window]
                              for scan in range(len(fmri) - time_window)])
    fmri_window = fmri_window.reshape((n_scans - time_window),
                                      time_window * np.shape(fmri)[1])
    series = series[: -time_window]
    sessions_id = sessions_id[: -time_window]

    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        series_train = series[train_index]
        series_test = series[test_index]
        fmri_window_train = fmri_window[train_index]
        fmri_window_test = fmri_window[test_index]

        # Fit multinomial logistic regression
        # We choose the best C between Cs values on a logarithmic scale
        # between 1e-4 and 1e4
        log = linear_model.LogisticRegressionCV(Cs=n_c, n_jobs=n_jobs)
        log.fit(fmri_window_train, series_train)

        # SCORE
        subject_accuracies[subject] += log.score(fmri_window_test, series_test)

        # TEST
        prediction = log.predict(fmri_window_test)
        prediction_proba = log.predict_proba(fmri_window_test)

        # R2 SCORES
        r2_scores = np.zeros(len(categories))
        for k in range(len(categories)):
            # Create stimulus array for the category
            cat_stimuli = [int(x == k) for x in series_test]

            # Calculate R2 score for stimulus approximation
            r2_scores[k] = metrics.r2_score(cat_stimuli, prediction_proba[:, k])

        categories_r2_scores[subject] += r2_scores

        # PLOT
        if subject == plot_subject:
            for k in range(len(categories)):
                # Create stimulus array for the category
                cat_stimuli = [int(x == k) for x in series_test]
                # Plot it along with the probability prediction
                x, y = k % 3, k / 3
                axes[x, y].plot(cat_stimuli)
                axes[x, y].plot(prediction_proba[:, k])

                # Calculate R2 score for stimulus approximation
                r2_score = metrics.r2_score(cat_stimuli, prediction_proba[:, k])


# Calculate and print the mean score
# f.suptitle('Predictions and R2 scores for subject 5, time window of %d, '
#            % time_window +
#            'accuracy = %.2f'
#            % mean_score)

for sub in range(n_subjects):
    subject_accuracies[sub] = subject_accuracies[sub]/66
    for cat in range(len(categories)):
        categories_r2_scores[sub][cat] = [sub][cat]/66

plt.show()