from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle


def general_settings():
    sns.set(font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    sns.set_context('paper', font_scale=2, rc={"lines.linewidth": 2.5})


def shift_value(rgb, shift):
    hsv = colors.rgb_to_hsv(rgb)
    hsv[-1] += shift
    return colors.hsv_to_rgb(hsv)


def color_palette(n_colors):
    orig_palette = sns.color_palette(n_colors=n_colors)
    shifts = np.linspace(-.3, .3, n_colors)
    alternate_shifts = shifts.copy()
    alternate_shifts[::2] = shifts[:len(shifts[::2])]
    alternate_shifts[1::2] = shifts[len(shifts[::2]):]
    palette = [shift_value(col, shift)
               for col, shift in zip(orig_palette, alternate_shifts)]
    return palette


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


def data_mrt_time_series(all_predictions_session, stimuli_session):
    """ """
    design = all_predictions_session[0]
    prediction = all_predictions_session[1]

    mirror = {}
    mirror['ground-truth'] = design[:, 1]
    mirror['prediction'] = prediction[:, 1]
    mirror['onsets'] = stimuli_session[:, 3]
    mirror['score'] = all_predictions_session[-1][1]

    plain = {}
    plain['ground-truth'] = design[:, 2]
    plain['prediction'] = prediction[:, 2]
    plain['onsets'] = stimuli_session[:, 2]
    plain['score'] = all_predictions_session[-1][2]

    dict = {}
    dict['plain'] = plain
    dict['mirror'] = mirror

    pickle.dump(dict, open('data_figure_mrt_time_series.pickle', 'wb'))


def figure_mrt_time_series():
    """ Create figure comparing ground-truth and predicted time-series for both
    classes in mirror-reversed text"""
    general_settings()

    # Load data
    data = pickle.load(open('data_figure_mrt_time_series.pickle', 'rb'))
    time = np.arange(200) * 2

    # Plot
    f, (ax1, ax2) = plt.subplots(2, sharey=True)

    # Colors
    cmap = color_palette(3)

    # First subplot
    ax1.plot(time, data['mirror']['ground-truth'][:200], linewidth=2.5,
             label='Ground-truth', color=cmap[0])
    ax1.plot(time, data['mirror']['prediction'][:200], linewidth=2.5,
             label='Prediction', color=cmap[1])
    ax1.vlines(np.where(data['mirror']['onsets'][:200])[0] * 2, -0.015, -0.005,
               linewidth=2.5, label='Onsets', color=cmap[2])
    ax1.text(346, 0.035, 'R2 score: {0:.2f}'.format(data['mirror']['score']))
    ax1.set_ylabel('Mirror word activation')
    ax1.get_xaxis().set_ticks([])
    ax1.legend(loc='upper left', ncol=2)

    # Second subplot
    ax2.plot(time, data['plain']['ground-truth'][:200], linewidth=2.5,
             label='Ground-truth', color=cmap[0])
    ax2.plot(time, data['plain']['prediction'][:200], linewidth=2.5,
             label='Prediction', color=cmap[1])
    ax2.vlines(np.where(data['plain']['onsets'][:200])[0] * 2, -0.015, -0.005,
               linewidth=2.5, label='Onsets', color=cmap[2])
    ax2.text(346, 0.035, 'R2 score: {0:.2f}'.format(data['plain']['score']))
    ax2.set_ylabel('Plain word activation')
    ax2.set_xlabel('Time (seconds)')
    ax2.yaxis.set_ticklabels([])

    # Remove frames
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    plt.show()


def model_image_create_onsets():
    """ """
    general_settings()

    # Offset for text
    h, v = 0.1 * 20, 0.1

    # Colors
    cmap = color_palette(3)

    plt.plot([0] * 100, color=cmap[0])
    plt.vlines([10, 40], 0, 0.5, color=cmap[0])
    plt.text(10 + h, 0.5 + v, '1', rotation=90)
    plt.text(40 + h, 0.5 + v, '4', rotation=90)
    plt.plot([2] * 100, color=cmap[1])
    plt.vlines([30, 60], 2, 2.5, color=cmap[1])
    plt.text(30 + h, 2.5 + v, '2', rotation=90)
    plt.text(60 + h, 2.5 + v, '5', rotation=90)
    plt.plot([4] * 100, color=cmap[2])
    plt.vlines([50, 80], 4, 4.5, color=cmap[2])
    plt.text(50 + h, 4.5 + v, '3', rotation=90)
    plt.text(80 + h, 4.5 + v, '6', rotation=90)
    plt.axis('off')
    plt.ylim(-.1, 5)

    plt.show()


def model_image_create_convoluted_onsets():
    """ """
    general_settings()
