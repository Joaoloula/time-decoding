from nilearn.input_data import MultiNiftiMasker
from nilearn.image import load_img
import pandas as pd
import numpy as np
import pickle
import glob
import os


def _uniform_masking(fmri_list, tr, high_pass=0.01, smoothing=5):
    """ Mask all the sessions uniformly, doing standardization, linear
    detrending, DCT high_pas filtering and gaussian smoothing.

    Parameters
    ----------

    fmri_list: array-like
        array containing multiple BOLD data from different sessions

    high_pass: float
        frequency at which to apply the high pass filter, defaults to 0.01

    smoothing: float
        spatial scale of the gaussian smoothing filter in mm, defaults to 5

    Returns
    -------

    fmri_list_masked: array-like
        array containing the masked data

    """
    masker = MultiNiftiMasker(mask_strategy='epi', standardize=True,
                              detrend=True, high_pass=0.01,
                              t_r=tr, smoothing_fwhm=smoothing)
    fmri_list_masked = masker.fit_transform(fmri_list)

    return fmri_list_masked


def _read_fmri_texture(sub, path, task):
    """ Reads BOLD signal data on Texture dataset """
    fmri_path = (('{path}{sub}/fmri/cra{sub}_td{task}.nii')
                 .format(path=path, task=task+1, sub=sub))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimuli_texture(stimuli_path, stim_set, n_tasks=6, tr=2.4, glm=False):
    """ Reads stimuli for the Texture dataset and returns a numpy array of
    shape [n_scans, n_classes] """

    if stim_set not in [1, 2]:
        raise NotImplementedError
    if stim_set == 1:
        conditions = np.load(stimuli_path + '1.npy')
        conditions = np.array([cond[:2] for cond in conditions]).reshape(6, -1)
    elif stim_set == 2:
        conditions = np.load(stimuli_path + '2.npy')
        conditions = np.array([cond[:2] for cond in conditions]).reshape(6, -1)
    onsets = [np.arange(len(cond)) * 4. for cond in conditions]

    if glm:
        return onsets, conditions

    classes = np.array(['rest', '01', '09', '12', '13', '14', '25', '0'])
    n_scans = 184 * n_tasks
    stimuli = np.zeros((n_scans, len(classes)))
    for t, stim in enumerate(conditions.ravel()):
        stim_class = np.where(classes == stim[:2])[0]
        scan = int(round(t * 4 / tr))

        stimuli[scan][stim_class] = 1

    # Fill the rest with category 'rest'
    rest_scans = np.where(np.sum(stimuli, axis=1) == 0)
    stimuli[rest_scans, 0] = 1
    return stimuli


def read_data_texture(subject, n_tasks=6, n_scans=205, tr=2.4,
                      path=('/home/loula/Programming/python/neurospin/Texture'
                            'Analysis/')):
    """
    Reads data from the Texture dataset.

    Parameters
    ----------

    subject: int from 0 to 3
        subject from which to read the data

    n_tasks: int from 1 to 6
        number of tasks to read from the subject, defaults to 6

    tr: float
        repetition time for the task (defaults to 2.4)

    n_scans: int
        number of scans per run, defaults to 184

    glm: bool
        whether to return data for use in a GLM model

    path: string
        path in which the dataset is located. Defaults to local path, but can be
        set to '/home/parietal/eickenbe/workspace/data/TextureAnalysis/' on
        drago

    Returns
    -------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    stimuli: numpy array of shape [n_scans, n_classes]
        labels for the stimuli in one-hot encoding

    """
    # Create list with subject ids and the set of stim for each of their runs
    subject_list = ['pf120155_1', 'pf120155_2', 'ns110383_1', 'ns110383_2',
                    'ap100009_1', 'ap100009_2', 'ap100009_3', 'pb120360_1',
                    'pb120360_2']
    stim_sets = {'pf120155_1': 1,
                 'pf120155_2': 2,
                 'ns110383_1': 1,
                 'ns110383_2': 2,
                 'ap100009_1': 1,
                 'ap100009_2': 1,
                 'ap100009_3': 2,
                 'pb120360_1': 1}

    # Read stimuli and fmri data
    sub = subject_list[subject]
    stimuli = _read_stimuli_texture(path+'stimuli/im_names_set', stim_sets[sub],
                                    glm=False)
    onsets, conditions = _read_stimuli_texture(path+'stimuli/im_names_set',
                                               stim_sets[sub], glm=True)
    path += sub[: -2] + '/'
    fmri = [_read_fmri_texture(sub, path, task)
            for task in range(n_tasks)]
    fmri = _uniform_masking(fmri, tr=tr)

    return fmri, stimuli, onsets, conditions


def _read_fmri_gauthier(sub, run, path):
    """ Reads BOLD signal data on Gauthier dataset """
    run = '_run00' + str(run + 1)
    fmri_path = (('{path}/BOLD/task001{run}/rabold.nii').format(
        path=path, sub=sub, run=run))
    fmri = load_img(fmri_path)

    return fmri


def read_data_gauthier(subject, n_runs=2, tr=1.5, n_scans=403, high_pass=0.01,
                       path=('/home/loula/Programming/python/neurospin/gauthier'
                             '2009resonance/results/')):
    """
    Reads data from the Gauthier dataset.

    Parameters
    ----------

    subject: int from 0 to 10
        subject from which to read the data

    n_runs: int from 1 to 6
        number of runs to read from the subject, defaults to 6

    tr: float
        repetition time for the task (defaults to 2)

    n_scans: int
        number of scans per run, defaults to 403

    two_classes: bool
        whether to restrict stimuli to two classes (text vs. reversed text), or
        to keep all the four original categories (which also include whether a
        given stimulus is the same as the one that precedes it)

    Returns
    -------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    stimuli: numpy array of shape [n_scans, n_classes]
        labels for the stimuli in one-hot encoding

    """
    if subject < 9:
        sub = 'sub00' + str(subject + 1)
    else:
        sub = 'sub0' + str(subject + 1)
    path += sub

    fmri = [_read_fmri_gauthier(sub, run, path) for run in range(n_runs)]
    fmri = _uniform_masking(fmri, tr=tr, high_pass=high_pass)

    info = pickle.load(open('gauthier_general_info_new.pickle', 'rb'))
    split_points = info['split_points']

    selected_fmri = [fmri[run][split: split + 20] for run in range(n_runs)
                     for split in split_points]

    onsets, conditions = info['onsets'], info['conditions']
    stimuli = info['stimuli']

    return selected_fmri, stimuli, onsets, conditions


def _read_fmri_mrt(sub, run, path):
    """ Reads BOLD signal data on Mirror-Reversed Text dataset """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    fmri_path = (('{path}/ses-test/func/r{sub}_ses-test_{task}'
                  '_{run}_bold.nii')
                 .format(path=path, sub=sub, task=task, run=run))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimulus_mrt(sub, run, path, n_scans, tr, glm=False):
    """ Reads stimulus data on Mirror-Reversed Text dataset """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    onsets_path = (('{path}/ses-test/func/{sub}_ses-test_{task}'
                    '_{run}_events.tsv')
                   .format(path=path, sub=sub, task=task, run=run))

    paradigm = pd.read_csv(onsets_path, sep='\t')
    onsets, durations, conditions = (np.array(paradigm['onset']),
                                     np.array(paradigm['duration']),
                                     np.array(paradigm['trial_type']))
    # Restrict classes to "plain" and "mirror" by slicing strings
    conditions = [conditions[event][:2] for event in range(len(conditions))]
    if glm:
        return onsets, conditions

    cats = np.array(['rest', 'ju', 'pl', 'mr'])
    stimuli = np.zeros((n_scans, len(cats)))
    for index, condition in enumerate(conditions):
        stim_onset = int(onsets[index])/tr
        stim_duration = 1 + int(durations[index])/tr
        (stimuli[stim_onset: stim_onset + stim_duration,
                 np.where(cats == condition)[0][0]]) = 1
    # Fill the rest with 'rest'
    stimuli[np.logical_not(np.sum(stimuli, axis=1).astype(bool)), 0] = 1

    return stimuli


def read_data_mrt(subject, n_runs=6, tr=2, n_scans=205,
                  high_pass=0.01, glm=False,
                  path=('/home/loula/Programming/python/neurospin/ds006_R2.0.0/'
                        'results/')):
    """
    Reads data from the Mirror-Reversed Text dataset.

    Parameters
    ----------

    subject: int from 0 to 13
        subject from which to read the data

    n_runs: int from 1 to 6
        number of runs to read from the subject, defaults to 6

    tr: float
        repetition time for the task (defaults to 2)

    n_scans: int
        number of scans per run, defaults to 205

    two_classes: bool
        whether to restrict stimuli to two classes (text vs. reversed text), or
        to keep all the four original categories (which also include whether a
        given stimulus is the same as the one that precedes it)

    Returns
    -------

    fmri: numpy array of shape [n_scans, n_voxels]
        data from the fmri sessions

    stimuli: numpy array of shape [n_scans, n_classes]
        labels for the stimuli in one-hot encoding

    """
    if subject < 9:
        sub = 'sub-0' + str(subject + 1)
    else:
        sub = 'sub-' + str(subject + 1)
    path += sub

    # Do not include missing runs
    os.chdir(path + "/ses-test/func/")
    runs = []
    for run in range(n_runs):
        if glob.glob("*run-0" + str(run + 1) + "*"):
            runs.append(run)

    stimuli = [_read_stimulus_mrt(sub, run, path, n_scans, tr)
               for run in runs]

    fmri = [_read_fmri_mrt(sub, run, path) for run in runs]
    fmri = _uniform_masking(fmri, tr=tr, high_pass=high_pass)

    glm_stimuli = [_read_stimulus_mrt(
        sub, run, path, n_scans, tr, glm=True)
        for run in runs]
    onsets = [glm_stimuli[run][0] for run in range(len(glm_stimuli))]
    conditions = [glm_stimuli[run][1] for run in range(len(glm_stimuli))]

    return fmri, stimuli, onsets, conditions
