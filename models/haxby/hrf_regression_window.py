# Performs multinomial logistic regression on activation data created from the
# Haxby dataset, using a custom time window
# Accuracy: 0.89 with 8 categories
# from hrf_estimation.savitzky_golay import savgol_filter
from nilearn.signal import clean
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
n_subjects = 6
plot_subject = 5  # ID of the subject to plot
time_window = 8
time_correction = 3  # Correction of the fmri scans in relation to the stimuli
low_pass = 0.1
model = 'ridge'  # 'ridge' for Ridge CV, 'log' for logistic regression CV

# PREPROCESSING
# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# MODEL
# Initialize mean score and score counter
subject_accuracies = np.zeros(n_subjects)
categories_r2_scores = np.zeros(9)
categories_r2_scores2 = np.zeros(9)
count = 0

sns.set_style('darkgrid')
f, axes = plt.subplots(3, 3)
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = read_data(subject, haxby_dataset,
                                                      n_scans)
    # Apply time window and time correction
    series = series[time_correction: -(time_window-time_correction)]
    sessions_id = sessions_id[time_correction: -(time_window-time_correction)]
    fmri_window = np.asarray([fmri[scan: scan + time_window]
                              for scan in range(len(fmri) - time_window)])
    fmri_window = fmri_window.reshape((np.shape(fmri_window)[0],
                                      (time_window) * np.shape(fmri)[1]))

    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        series_train = series[train_index]
        series_test = series[test_index]
        fmri_window_train = fmri_window[train_index]
        fmri_window_test = fmri_window[test_index]

        if model == 'log':
            # Fit multinomial logistic regression
            # We choose the best C between Cs values on a logarithmic scale
            # between 1e-4 and 1e4
            log = linear_model.LogisticRegressionCV(Cs=n_c, n_jobs=n_jobs)
            log.fit(fmri_window_train, series_train)

            # SCORE
            subject_accuracies[subject] += log.score(fmri_window_test,
                                                     series_test)

            # TEST
            prediction = log.predict(fmri_window_test)
            prediction_proba = log.predict_proba(fmri_window_test)

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
                    r2_score = metrics.r2_score(cat_stimuli,
                                                prediction_proba[:, k])

        if model == 'ridge':
            alphas = [0.1, 1, 10, 100, 1000]
            ridge = linear_model.RidgeCV(alphas=alphas)
            for k in range(len(categories)):
                # Create stimulus array for the category
                cat_stimuli_train = [int(cat == k) for cat in series_train]
                cat_stimuli_test = [int(cat == k) for cat in series_test]

                # Fit the Ridge regression
                ridge.fit(fmri_window_train, cat_stimuli_train)
                prediction = ridge.predict(fmri_window_test)
                prediction = prediction.reshape(len(prediction), 1)
                prediction = clean(prediction, standardize=False, detrend=False,
                                   low_pass=0.1)
                prediction = prediction.reshape(-1)
                # prediction = savgol_filter(prediction, 7, 3)
                score = metrics.r2_score(cat_stimuli_test, prediction)
                categories_r2_scores[k] += score

                # PLOT
                if subject == plot_subject:
                    # Plot it along with the probability prediction
                    x, y = k % 3, k / 3
                    axes[x, y].plot(cat_stimuli_test)
                    axes[x, y].plot(prediction)
                    axes[x, y].set_title('Category {cate}, R2 score {score:.2f}'
                                         .format(cate=categories[k],
                                                 score=score))

        count += 1
        break  # Only run one CV step per subject for fast prototyping

    print('processing subject ' + str(subject))

# Calculate and print the mean score
f.suptitle('Predictions and R2 scores for subject %d, time window of %d, '
           % (plot_subject, time_window) +
           'low pass cutoff of %.2f' % low_pass, fontsize=20)

categories_r2_scores /= count
print(categories_r2_scores)
plt.show()
