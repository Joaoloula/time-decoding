from nistats.hemodynamic_models import _hrf_kernel
from matplotlib import colors
import matplotlib.patches as patches
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
    sns.set_context('paper')


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


def gauthier_barplot():
    """ """
    general_settings()
    plt.figure(figsize=(4, 6))

    # Load data
    data = pickle.load(open('../scripts/all_gauthier_separate.pickle',
                            'rb'))

    # Colors
    cmap = sns.color_palette("colorblind", n_colors=3)

    # Plot
    ax = sns.barplot(x='model', y='accuracy', hue='isi', data=data,
                     order=['GLM', 'GLMs', 'spatiotemporal SVM',
                            'logistic deconvolution'], palette=cmap)
    x_ticks = ['GLM', 'GLMs', 'Spatiotemporal\nSVM', 'Logistic\nDeconvolution']
    ax.set_xticklabels(x_ticks)
    ax.set_ylim(0.4, 0.8)

    plt.savefig('gauthier_boxplot_strip.png')

    plt.show()


def all_boxplot():
    """ """
    general_settings()
    plt.figure(figsize=(4, 6))

    # Load data
    data = pickle.load(open('../scripts/mrt_gauthier_all_dataframe.pickle',
                            'rb'))
    data = data.loc[data['dataset'] == 'mirror-reversed text']

    # Colors
    cmap = sns.color_palette("colorblind", n_colors=3)
    cmap.append(sns.color_palette("muted", 4)[3])

    # Plot
    ax = sns.boxplot(x='accuracy', y='dataset', hue='model', data=data,
                     palette=cmap, orient='h')
    """
    sns.swarmplot(x='accuracy', y='model', hue='dataset', data=data,
                  palette=cmap, split=True)
    """
    ax.legend(loc='lower left')

    y_ticks = ['Mirror-\nreversed\nText', 'Temporal\nTuning']
    ax.set_yticklabels(y_ticks)

    plt.savefig('all_boxplot_strip.png')

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
    probas = pickle.load(open('../scripts/plain_mirror_probabilities.pickle',
                              'rb'))
    time = np.arange(100) * 2
    up = 0.013
    left = -3

    # Plot
    f, (ax1, ax2) = plt.subplots(2, figsize=(4, 5), sharey=True)

    # Colors
    cmap = sns.color_palette("colorblind", n_colors=4)
    onset_color = sns.color_palette("muted", 4)[3]
    red_green = sns.color_palette(n_colors=3)
    red, green = red_green[2], red_green[1]
    grey = sns.color_palette("Accent", n_colors=8)[7]

    # First subplot
    ax1.plot(time, data['mirror']['ground-truth'][: 100],
             label='Input: HRF-convolved onsets', color=cmap[0])
    ax1.plot(time, data['mirror']['prediction'][: 100], linestyle='--',
             dashes=(3, 2), label='Prediction: estimates from brain activity',
             color=cmap[2])

    mirror_onsets = np.where(data['mirror']['onsets'][:100])[0] * 2
    ax1.vlines(mirror_onsets, -0.015, -0.005,  color=onset_color)
    ax1.text(160, -0.025, 'mirror onsets', color=onset_color)
    ax1.text(-30+left, 0+up, '\'Mirror\'\ncondition\ntime-series',
             multialignment='center', rotation='vertical')
    ax1.get_xaxis().set_ticks([])
    ax1.legend(loc=(0, 0.35), ncol=1)
    ax1.set_xlim(-1, 201)

    # Second subplot
    ax2.plot(time, data['plain']['ground-truth'][: 100],
             label='Input: HRF-convolved onsets', color=cmap[0])
    ax2.plot(time, data['plain']['prediction'][: 100], linestyle='--',
             dashes=(3, 2), label='Prediction: estimates from brain activity',
             color=cmap[2])

    plain_onsets = np.where(data['plain']['onsets'][:100])[0] * 2
    ax2.vlines(plain_onsets, 0.04, 0.05, label='Onsets',
               color=onset_color)
    ax2.text(160, 0.055, 'plain onsets', color=onset_color)
    ax2.text(-30+left, 0+up, '\'Plain\'\ncondition\ntime-series',
             rotation='vertical', multialignment='center')
    ax2.set_xlabel('Time (seconds)')
    ax2.yaxis.set_ticklabels([])
    ax2.set_xlim(-1, 201)

    # Drawing probabilities
    baseline = 0.115
    ax2.text(-30+left, baseline+up, 'Predicted\nprobability',
             rotation='vertical', multialignment='center')
    ax2.text(-10+left, baseline+0.02+up, '(        /          )',
             rotation='vertical')
    ax2.text(-10+left, baseline-0.0265+up, 'right', color=green,
             rotation='vertical')
    ax2.text(-10+left, baseline+0.0165+up, 'wrong', color=red,
             rotation='vertical')
    ax2.plot(np.arange(-1, 190), [baseline] * 191, color=grey)
    ax2.plot(np.arange(-1, 190), [baseline + 0.02] * 191,
             linestyle='--', color=grey)
    ax2.plot(np.arange(-1, 190), [baseline - 0.02] * 191,
             linestyle='--', color=grey)
    ax2.text(191, baseline+0.017, 'Mirror', color=grey)
    ax2.text(191, baseline-0.023, 'Plain', color=grey)
    for index, time in enumerate(plain_onsets):
        prob = (probas['plain probabilities'][index] - 0.5) * 0.04
        color = red if prob < 0 else green
        ax2.vlines(time, baseline, baseline - prob, color=color, linewidth=3.5)
    for index, time in enumerate(mirror_onsets):
        prob = (probas['mirror probabilities'][index] - 0.5) * 0.04
        color = red if prob < 0 else green
        ax2.vlines(time, baseline, baseline + prob, color=color, linewidth=3.5)

    # Remove frames
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    plt.savefig('mrt_time_series_version3.png')

    plt.show()


def model_image_create_onsets():
    """ """
    general_settings()

    # Offset for text
    h, v = -0.1 * 65, 0.1

    # Colors
    cmap = sns.color_palette("colorblind", n_colors=3)
    width = 10
    fsize = 60

    # Plot
    plt.plot([0] * 100, color=cmap[0], linewidth=width)
    plt.vlines([10, 40], 0, 0.5, color=cmap[0], linewidth=width)
    plt.text(10 + h, 0.5 + v, '1', rotation=90, fontsize=fsize)
    plt.text(40 + h, 0.5 + v, '4', rotation=90, fontsize=fsize)
    plt.plot([2] * 100, color=cmap[1], linewidth=width)
    plt.vlines([30, 60], 2, 2.5, color=cmap[1], linewidth=width)
    plt.text(30 + h, 2.5 + v, '2', rotation=90, fontsize=fsize)
    plt.text(60 + h, 2.5 + v, '5', rotation=90, fontsize=fsize)
    plt.plot([4] * 100, color=cmap[2], linewidth=width)
    plt.vlines([50, 80], 4, 4.5, color=cmap[2], linewidth=width)
    plt.text(50 + h, 4.5 + v, '3', rotation=90, fontsize=fsize)
    plt.text(80 + h, 4.5 + v, '6', rotation=90, fontsize=fsize)
    plt.axis('off')
    plt.ylim(-.1, 5)

    plt.show()


def model_image_create_convolved_onsets(noise=False):
    """ """
    general_settings()
    onsets1, onsets2, onsets3 = np.zeros((3, 100))
    onsets1[[10, 40]] = 1
    onsets2[[30, 60]] = 1
    onsets3[[50, 80]] = 1
    width = 10
    fsize = 60

    # Convolve the onsets
    hkernel = _hrf_kernel('spm', 1, oversampling=1)
    conv_onsets1 = np.array(
        [np.convolve(onsets1, h)[:onsets1.size] for h in hkernel])[0]
    conv_onsets2 = np.array(
        [np.convolve(onsets2, h)[:onsets2.size] for h in hkernel])[0]
    conv_onsets3 = np.array(
        [np.convolve(onsets3, h)[:onsets3.size] for h in hkernel])[0]

    # Normalize to same peak as onsets
    norm = np.max(conv_onsets1)
    conv_onsets1 /= (norm * 2)
    conv_onsets2 /= (norm * 2)
    conv_onsets3 /= (norm * 2)

    # Noise to simulate estimation
    if noise:
        conv_onsets1 += np.random.normal(0, .1, 100)
        conv_onsets2 += np.random.normal(0, .1, 100)
        conv_onsets3 += np.random.normal(0, .1, 100)

    # Offset for text
    h, v = -0.1 * 65, 0.1

    # Colors
    cmap = sns.color_palette("colorblind", n_colors=3)
    tw_color = sns.color_palette("muted", 4)[3]

    # Plot
    f, ax = plt.subplots(1)
    ax.plot(conv_onsets1, color=cmap[0], linewidth=width)
    ax.text(10 + h, 0.5 + v, '1', rotation=90, fontsize=fsize)
    ax.text(40 + h, 0.5 + v, '4', rotation=90, fontsize=fsize)
    ax.plot(2 + conv_onsets2, color=cmap[1], linewidth=width)
    ax.text(30 + h, 2.5 + v, '2', rotation=90, fontsize=fsize)
    ax.text(60 + h, 2.5 + v, '5', rotation=90, fontsize=fsize)
    ax.plot(4 + conv_onsets3, color=cmap[2], linewidth=width)
    ax.text(50 + h, 4.5 + v, '3', rotation=90, fontsize=fsize)
    ax.text(80 + h, 4.5 + v, '6', rotation=90, fontsize=fsize)
    ax.set_ylim(-1.5, 5.5)
    ax.axis('off')

    # Add time windows visualization on the estimation
    mult = 1
    if noise:
        ax.add_patch(patches.Rectangle((10, -1), 5, 6, fill=False,
                                       edgecolor=tw_color,
                                       linewidth=width*mult,
                                       linestyle='dashed'))
        ax.add_patch(patches.Rectangle((30, -1), 5, 6, fill=False,
                                       edgecolor=tw_color,
                                       linewidth=width*mult,
                                       linestyle='dashed'))
        ax.add_patch(patches.Rectangle((40, -1), 5, 6, fill=False,
                                       edgecolor=tw_color,
                                       linewidth=width*mult,
                                       linestyle='dashed'))
        ax.add_patch(patches.Rectangle((50, -1), 5, 6, fill=False,
                                       edgecolor=tw_color,
                                       linewidth=width*mult,
                                       linestyle='dashed'))
        ax.add_patch(patches.Rectangle((60, -1), 5, 6, fill=False,
                                       edgecolor=tw_color,
                                       linewidth=width*mult,
                                       linestyle='dashed'))
        ax.add_patch(patches.Rectangle((80, -1), 5, 6, fill=False,
                                       edgecolor=tw_color,
                                       linewidth=width*mult,
                                       linestyle='dashed'))

    print(ax.patches)

    plt.show()


def model_image_estimation_matrix():
    """ """
    general_settings()
    fsize = 60

    # Colors
    cmap = sns.color_palette("colorblind", n_colors=3)
    h, v, offset = 0.5, 8, -0.5
    plt.xlim(offset, 4)
    plt.ylim(0, 49)
    plt.axis('off')

    # Onset texts:
    plt.text(offset, 6 * v, "1: ", color='k', fontsize=fsize)
    plt.text(offset, 5 * v, "2: ", color='k', fontsize=fsize)
    plt.text(offset, 4 * v, "3: ", color='k', fontsize=fsize)
    plt.text(offset, 3 * v, "4: ", color='k', fontsize=fsize)
    plt.text(offset, 2 * v, "5: ", color='k', fontsize=fsize)
    plt.text(offset, 1 * v, "6: ", color='k', fontsize=fsize)

    # First column
    plt.text(0, 6 * v, "0.6", color=cmap[0], fontsize=fsize)
    plt.text(0, 5 * v, "0.2", color=cmap[0], fontsize=fsize)
    plt.text(0, 4 * v, "0.1", color=cmap[0], fontsize=fsize)
    plt.text(0, 3 * v, "0.7", color=cmap[0], fontsize=fsize)
    plt.text(0, 2 * v, "0.2", color=cmap[0], fontsize=fsize)
    plt.text(0, 1 * v, "0.3", color=cmap[0], fontsize=fsize)

    # Second column
    plt.text(h, 6 * v, "0.1", color=cmap[1], fontsize=fsize)
    plt.text(h, 5 * v, "0.7", color=cmap[1], fontsize=fsize)
    plt.text(h, 4 * v, "0.1", color=cmap[1], fontsize=fsize)
    plt.text(h, 3 * v, "0.2", color=cmap[1], fontsize=fsize)
    plt.text(h, 2 * v, "0.6", color=cmap[1], fontsize=fsize)
    plt.text(h, 1 * v, "0.2", color=cmap[1], fontsize=fsize)

    # Third column
    plt.text(2 * h, 6 * v, "0.3", color=cmap[2], fontsize=fsize)
    plt.text(2 * h, 5 * v, "0.1", color=cmap[2], fontsize=fsize)
    plt.text(2 * h, 4 * v, "0.8", color=cmap[2], fontsize=fsize)
    plt.text(2 * h, 3 * v, "0.1", color=cmap[2], fontsize=fsize)
    plt.text(2 * h, 2 * v, "0.2", color=cmap[2], fontsize=fsize)
    plt.text(2 * h, 1 * v, "0.5", color=cmap[2], fontsize=fsize)

    plt.show()
