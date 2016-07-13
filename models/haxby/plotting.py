from nistats.hemodynamic_models import glover_hrf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pickle


def plot_barchart():
    """ """
    scores = pickle.load(open('time_window_accuracies.pickle', 'rb'))
    model = np.hstack((['time window of 1'] * 6, ['time window of 5'] * 6,
                       ['time window of 8'] * 6))
    subject_list = ['Subject ' + str(sub) for sub in range(1, 7)]
    subject = np.hstack((subject_list * 3))

    dict = {}
    dict['accuracy'] = scores
    dict['model'] = model
    dict['subject'] = subject
    data = pd.DataFrame(dict)

    plt.style.use('ggplot')

    sns.set_context('talk', font_scale=1.5)
    ax = sns.barplot(x='subject', y='accuracy', data=data, hue='model',
                     hue_order=['time window of 1', 'time window of 5',
                                'time window of 8'])
    ax.set_ylim(0.5, 1)
    plt.show()


def plot_hrfs():
    """ Plots overlapping HRFs """
    hrf = glover_hrf(tr=2.5)
    hrf_one = np.hstack((hrf, [0] * 143))
    hrf_two = np.hstack(([0] * 75, hrf, [0] * 68))

    plt.style.use('ggplot')

    plt.plot(np.linspace(0, 40, 347), hrf_one, linewidth=2)
    plt.plot(np.linspace(0, 40, 347), hrf_two, linewidth=2)
    ax = plt.gca()
    ax.axes.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', labelsize=15)

    plt.show()
