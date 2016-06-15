# Performs multinomial logistic regression on activation data created from the
# Haxby dataset
from sklearn.cross_validation import LeavePLabelOut
from nilearn.input_data import NiftiMasker
from nistats import hemodynamic_models
from sklearn import linear_model
from nilearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PARAMETERS
tr = 2.5
n_scans = 1452
frame_times = np.arange(n_scans) * tr
hrf_model = 'glover'
oversample = 16

# PREPROCESSING

# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=6)

# Create categories
categories = ['face', 'house', 'bottle', 'chair']

# Create sessions id
sessions_id = [x/121 for x in range(1452)]

# Initialize the stimuli and fmri dictionaries
stimuli = {}
fmri = {}

# Initialize the experimental condition dictionaries
onsets = {}
durations = {}
amplitudes = {}
exp_conditions = {}

# Initiliaze the signal dictionary
signal = {}

# Initialize the dictionary for each label, and the series for each category
for i in range(6):
    stimuli[str(i)] = {}
    fmri[str(i)] = {}
    onsets[str(i)] = {}
    durations[str(i)] = {}
    amplitudes[str(i)] = {}
    exp_conditions[str(i)] = {}
    signal[str(i)] = {}
    for category in categories:
        stimuli[str(i)][category] = []
        onsets[str(i)][category] = []
        durations[str(i)][category] = []
        amplitudes[str(i)][category] = []
        signal[str(i)][category] = []

# Loop through all subjects
for i in range(6):
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[i], delimiter=" ")
    target = labels['labels']

    # Create the labeled time series, onsets, durations and amplitudes
    for j in range(len(target)):
        for category in categories:
            # If a stimulus of the current category is being presented
            if target[j] == category:
                stimuli[str(i)][category].append(1)
                # Detect onset
                if j >= 1 and target[j-1] != category:
                    onsets[str(i)][category].append(tr*j)
                    # Suppose amplitude to be 1
                    amplitudes[str(i)][category].append(1)
            else:
                stimuli[str(i)][category].append(0)
                # Detect end of stimulus and calculate duration
                if j >= 1 and target[j-1] == category:
                    duration = (tr * j - onsets[str(i)][category][-1])
                    durations[str(i)][category].append(duration)

    # Summarize information in the experimental conditions and create HRFs
    for category in categories:
        exp_conditions[str(i)][category] = np.vstack((onsets[str(i)][category],
                                            durations[str(i)][category],
                                            amplitudes[str(i)][category]))
        signal[str(i)][category], _ = hemodynamic_models.compute_regressor(
            exp_conditions[str(i)][category], hrf_model, frame_times,
            con_id=category, oversampling = oversample)

    # Read activity data
    # Standardize and detrend per session
    mask_filename = haxby_dataset.mask_vt[i]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, sessions=sessions_id)
    func_filename = haxby_dataset.func[i]
    fmri[str(i)] = nifti_masker.fit_transform(func_filename)

# MODEL

# Create Leave P Label Out cross validation
lplo = LeavePLabelOut(sessions_id, p=2)

# Create train and test dictionaries
multi_stimuli_train = {}
multi_stimuli_test = {}
fmri_train = {}
fmri_test = {}

sns.set_style('darkgrid')
f, axes = plt.subplots(3, 2)
for i in range(6):
    # Create vector of all the stimuli combined
    multi_stimuli = np.zeros(n_scans)

    for j in range(len(categories)):
        # Threshold the HRF to convert it into categories
        threshold = .5 * (np.amax(signal[str(i)][categories[j]]) -
                          np.amin(signal[str(i)][categories[j]]))
        current_stimuli = [int(x>=threshold)
                           for x in signal[str(i)][categories[j]]]
        multi_stimuli += (j+1) * np.array(current_stimuli)

    # Separate data into train and test sets
    for train_index, test_index in lplo:
        multi_stimuli_train[str(i)] = multi_stimuli[train_index]
        multi_stimuli_test[str(i)] = multi_stimuli[test_index]
        fmri_train[str(i)] = fmri[str(i)][train_index]
        fmri_test[str(i)] = fmri[str(i)][test_index]

    # Fit multinomial logistic regression
    # We choose the best C between Cs values on a logarithmic scale
    # between 1e-4 and 1e4
    log = linear_model.LogisticRegressionCV(Cs=5, n_jobs=2)
    log.fit(fmri_train[str(i)], multi_stimuli_train[str(i)])

    # TEST
    prediction = log.predict(fmri_test[str(i)])
    prediction_proba = log.predict_proba(fmri_test[str(i)])

    # PLOT
    # Plot it along with the probability prediction for the face label
    axes[i % 3, i/3].plot(range(len(multi_stimuli_test[str(i)])),
                          [x==1 for x in multi_stimuli_test[str(i)]])
    axes[i % 3, i/3].plot(range(len(prediction_proba)),
                          prediction_proba[:, 1], color='g')

    # Add subject number and train score to title
    axes[i % 3, i/3].set_title('Subject %(subject)d, score %(score).2f'
        % {
           'subject': i,
           'score': log.score(fmri_test[str(i)], multi_stimuli_test[str(i)])
           }
        )
plt.show()
