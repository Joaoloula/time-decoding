from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle


dictionary = pickle.load(open('faces_houses_all_times.pickle', 'rb'))
faces = dictionary['faces']
houses = dictionary['houses']

centroid_faces = np.mean()

pca = PCA(n_components=2)
pca.fit(np.vstack((faces, houses)))
faces = pca.fit_transform(faces)
houses = pca.fit_transform(houses)

plt.style.use('ggplot')
plt.scatter(faces[:, 0], faces[:, 1], marker="o", s=40,
            color=plt.rcParams['axes.color_cycle'][0])
plt.scatter(houses[:, 0], houses[:, 1], marker="o", s=40,
            color=plt.rcParams['axes.color_cycle'][1])

axes = plt.gca()
axes.axes.get_xaxis().set_visible(False)
axes.axes.get_yaxis().set_visible(False)

plt.show()
