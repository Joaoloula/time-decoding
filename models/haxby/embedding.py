from sklearn.cross_validation import LeavePOut
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifierCV
from nilearn import datasets
import helper_functions as hf
import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS
n_components = 2  # number of components to use in the PCA
n_neighbors = 5
n_subjects = 1
plot_subject = 0  # ID of the subject to plot
time_window = 8
cutoff = 0
delay = 3  # Correction of the fmri scans in relation to the stimuli
plot = True

# PREPROCESSING
# Import all subjects from the haxby dataset
haxby_dataset = datasets.fetch_haxby(n_subjects=n_subjects)

# Initialize mean score and score counter
mean_score = 0.
count = 0
plt.style.use('ggplot')
for subject in range(n_subjects):
    fmri, series, sessions_id, categories = hf.read_data(subject, haxby_dataset)
    # Apply time window and time correction
    fmri, series, sessions_id = hf.apply_time_window(fmri, series, sessions_id,
                                                     time_window=time_window,
                                                     delay=delay)

    embedding, labels = hf.create_embedding(fmri, series, categories)

    # Create custom mask
    custom_mask = np.hstack((np.where(labels == 6), np.where(labels == 7)))
    #                        np.where(labels == 4), np.where(labels == 0)))
    embedding = embedding[custom_mask]
    labels = labels[custom_mask]
    embedding = embedding.reshape(embedding.shape[1:])
    labels = labels.reshape(labels.shape[1:])

    lpo = LeavePOut(len(embedding), p=len(embedding)/10)

    # Divide in train and test sets
    for train_index, test_index in lpo:
        embedding_train = embedding[train_index]
        embedding_test = embedding[test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]

        # Model
        # knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=2)
        # knn.fit(embedding_train, labels_train)
        # mean_score += knn.score(embedding_test, labels_test)
        ridge = RidgeClassifierCV()
        ridge.fit(embedding_train, labels_train)
        mean_score += ridge.score(embedding_test, labels_test)

        count += 1

        if count >= 1:
            break

    if plot:
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(embedding)
        faces = embedding[np.where(labels == 6)]
        houses = embedding[np.where(labels == 7)]
        plt.scatter(faces[:, 0], faces[:, 1], marker="o", s=40,
                    color=plt.rcParams['axes.color_cycle'][0])
        plt.scatter(houses[:, 0], houses[:, 1], marker="o", s=40,
                    color=plt.rcParams['axes.color_cycle'][1])

        axes = plt.gca()
        axes.axes.get_xaxis().set_visible(False)
        axes.axes.get_yaxis().set_visible(False)

        plt.show()
    """
        handles = []
        for category in range(len(categories) - 1):
            handles.append(plt.plot(
                pca_embeddings[n_sessions*category: n_sessions*(category+1), 0],
                pca_embeddings[n_sessions*category: n_sessions*(category+1), 1],
                "o"))

        plt.legend(handles, categories[1:])
        plt.title("Embedded responses to stimuli for all 8 haxby categories, " +
                  "time window of %d scans" % (time_window), fontsize=20)
    plt.show()
    """


mean_score /= count
print(mean_score)
