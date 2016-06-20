# Performs multinomial logistic regression on activation data created from the
# Haxby dataset
# Accuracy: 0.74 without confounds
# 0.68 with 5 confounds
from sklearn.cross_validation import LeavePLabelOut
from helper_functions import read_data
from sklearn import linear_model
from sklearn import metrics
from nilearn import datasets
from nilearn import signal
import matplotlib.pyplot as plt
import seaborn as sns

# PARAMETERS
n_scans = 1452
n_sessions = 12
n_c = 5  # number of Cs to use in logistic regression CV
n_jobs = 2  # number of jobs to use in logistic regression CV
n_subjects = 6
n_confounds = 3
plot_subject = 5  # ID of the subject to plot

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# Initialize series and fmri dictionaries
series = {}
fmri = {}

# MODEL

# Initialize mean score and score counter
mean_score = 0.
score_count = 0

sns.set_style('darkgrid')
f, axes = plt.subplots(3, 3)
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = read_data(subject, haxby_dataset,
                                                      n_scans)
    # Calculate highest variance confounds
    confounds = signal.high_variance_confounds(fmri, n_confounds=n_confounds)
    # fmri = signal.clean(fmri, confounds=confounds)

    # Initialize Leave P Label Out cross validation
    lplo = LeavePLabelOut(sessions_id, p=2)

    # Divide in train and test sets
    for train_index, test_index in lplo:
        series_train = series[train_index]
        series_test = series[test_index]
        fmri_train = fmri[train_index]
        fmri_test = fmri[test_index]

        # Fit multinomial logistic regression
        # We choose the best C between Cs values on a logarithmic scale
        # between 1e-4 and 1e4
        log = linear_model.LogisticRegressionCV(Cs=n_c, n_jobs=n_jobs)
        log.fit(fmri_train, series_train)

        # SCORE
        mean_score += log.score(fmri_test, series_test)

        # TEST
        prediction = log.predict(fmri_test)
        prediction_proba = log.predict_proba(fmri_test)

        # PLOT
        if subject == plot_subject:
            for k in range(9):
                # Create array for the given stimulus
                cat_stimuli = [int(x == k) for x in series_test]
                # Plot it along with the probability prediction
                x, y = k % 3, k / 3
                axes[x, y].plot(cat_stimuli)
                axes[x, y].plot(prediction_proba[:, k])

                # Calculate R2 score for stimulus approximation
                r2_score = metrics.r2_score(cat_stimuli, prediction_proba[:, k])

                # Add subject number and train score to title
                axes[x, y].set_title('Category %(cat)s, R2 score %(score).2f'
                                     % {'cat': categories[k], 'score': r2_score}
                                     )

        # Update score counter
    score_count += 1

# Calculate and print the mean score
mean_score = mean_score / score_count
f.suptitle('Predictions and R2 scores for subject 5, accuracy = %.2f'
           % mean_score)
print("The accuracay is %.4f" % mean_score)

plt.show()
