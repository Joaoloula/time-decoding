import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


def plot(prediction, stimuli, scores, accuracy, delay=3, time_window=8,
         two_classes=False, kernel=None, penalty=1):
    """ Plots predictions and ground truths for each of the classes, as well
    as their r2 scores. """
    plt.style.use('ggplot')
    if two_classes:
        fig, axes = plt.subplots(2)

        title = ('Ridge predictions for \'plain\' vs. \'reversed\', time window'
                 'of {tw}, delay of {delay}. Accuracy: {acc:.2f}').format(
                 tw=time_window, delay=delay, acc=accuracy)

        if kernel == 'time_smoothing':
            title += ' Kernel: time smoothing, penalty = {}'.format(penalty)

        elif kernel == 'voxel_weighing':
            title += ' Kernel: voxel weighing'

        fig.suptitle(title, fontsize=20)
        axes[0].plot(stimuli[:, 0])
        axes[0].plot(prediction[:, 0])
        axes[0].set_title(('Predictions for category \'plain\', r2 score of '
                           '{score:.2f}').format(score=scores[0]), fontsize=18)
        axes[1].plot(stimuli[:, 1])
        axes[1].plot(prediction[:, 1])
        axes[1].set_title(('Predictions for category \'reversed\', r2 score of '
                           '{score:.2f}').format(score=scores[1]), fontsize=18)
    else:
        cats = np.array(['junk', 'pl_ns', 'pl_sw', 'mr_ns', 'mr_sw'])
        fig, axes = plt.subplots(5)
        fig.suptitle('Ridge predictions for all classes, time window of {tw}'
                     .format(tw=time_window), fontsize=20)
        for cat in range(len(cats)):
            axes[cat].plot(stimuli[:, cat])
            axes[cat].plot(prediction[:, cat])
            axes[cat].set_title(('Prediction for category {cat}, R2 score of '
                                 '{score:.2f}').format(cat=cats[cat],
                                                       score=scores[cat]),
                                fontsize=18)

    plt.show()


def make_dataframe(score_list, model_list):
    n_subjects = len(score_list[0])
    n_models = len(model_list)

    scores = np.hstack(score_list)
    models = np.hstack([[model] * n_subjects for model in model_list])
    subjects = range(1, n_subjects + 1) * n_models
    dict = {}
    dict['accuracy'] = scores
    dict['model'] = models
    dict['subjects'] = subjects
    data = pd.DataFrame(dict)

    return data


def score_barplot(data):
    """ """
    plt.style.use('ggplot')
    sns.set_context('talk', font_scale=1.5)
    ax = sns.boxplot(x='model', y='accuracy', data=data, orient='h')
    ax.set_title('Classification accuracies for GLM and time-domain decoding')
    ax.set_ylim(0.5, 1)

    plt.show()


def figure_mrt_time_series():
    """ """
    # General settings
    sns.set_context('paper')
    sns.set(font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

    # Load data
    data = pickle.load(open('data_figure_mrt_time_series.pickle', 'rb'))

    # Plot
    f, (ax1, ax2) = plt.subplots(2, sharey=True)

    # First subplot
    ground_truth1 = ax1.plot(data['plain']['ground-truth'][:200],
                             label='Ground-truth')
    prediction1 = ax1.plot(data['plain']['prediction'][:200],
                           label='Prediction')
    ax1.text(180, 0.035, 'R2 score: {0:.2f}'.format(data['plain']['score']))
    ax1.set_ylabel('Plain word activation')
    ax1.get_xaxis().set_ticks([])
    ax1.yaxis.set_ticklabels([])
    f.legend([ground_truth1, prediction1], ['Ground-truth', 'Prediction'])

    # Second subplot
    ground_truth2 = ax2.plot(data['mirror']['ground-truth'][:200],
                             label='Ground-truth')
    prediction2 = ax2.plot(data['mirror']['prediction'][:200],
                           label='Prediction')
    ax2.text(180, 0.035, 'R2 score: {0:.2f}'.format(data['mirror']['score']))
    ax2.set_ylabel('Mirror word activation')
    ax2.get_xaxis().set_ticks([])

    plt.show()
