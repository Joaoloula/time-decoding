from nilearn.input_data import MultiNiftiMasker
from nilearn.image import load_img
import pandas as pd
import numpy as np
import glob
import os


def _uniform_masking(fmri_list, high_pass=0.01, smoothing=5):
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
                              t_r=2, smoothing_fwhm=smoothing)
    fmri_list_masked = masker.fit_transform(fmri_list)

    return fmri_list_masked


def _read_fmri_texture(sub, run, path, task):
    """ Reads BOLD signal data on Texture dataset """
    run = sub + '_' + str(run + 1)
    fmri_path = (('{path}{run}/fmri/cra{run}_td{task}.nii')
                 .format(path=path, task=task+1, run=run))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimuli_texture(stimuli_path, stim_set, n_tasks=6, tr=2.4, glm=False):
    """ Reads stimuli for the Texture dataset and returns a numpy array of
    shape [n_scans, n_classes] """

    if stim_set not in [1, 2]:
        raise NotImplementedError
    if stim_set == 1:
        session_stimuli = np.load(stimuli_path + '1.npy')
    elif stim_set == 2:
        session_stimuli = np.load(stimuli_path + '2.npy')

    if glm:
        return session_stimuli

    classes = np.array(['rest', '01', '09', '12', '13', '14', '25', '0'])
    n_scans = 184 * n_tasks
    stimuli = np.zeros((n_scans, len(classes)))
    for t, stim in enumerate(session_stimuli):
        stim_class = np.where(classes == stim[:2])[0]
        scan = int(round(t * 4 / tr))

        stimuli[scan][stim_class] = 1

    # Fill the rest with category 'rest'
    rest_scans = np.where(np.sum(stimuli, axis=1) == 0)
    stimuli[rest_scans, 0] = 1
    return stimuli


def read_data_texture(subject, n_runs=2, n_tasks=6, glm=False, n_scans=205,
    tr=2.4, path='/home/loula/Programming/python/neurospin/TextureAnalysis/'):
    """
    Reads data from the Texture dataset.

    Parameters
    ----------

    subject: int from 0 to 3
        subject from which to read the data

    n_runs: int from 1 to 3
        number of runs to read from the subject, defaults to 2

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
    subject_list = ['pf120155', 'ns110383', 'ap100009', 'pb120360']
    stim_sets = {'pf120155': [1, 2],
                 'ns110383': [1, 2],
                 'ap100009': [1, 1, 2],
                 'pb120360': [1, 1]}

    sub = subject_list[subject]

    stimuli = [_read_stimuli_texture(path+'stimuli/im_names_set',
                                     stim_sets[sub][run], glm=glm)
               for run in range(n_runs)]

    path += sub + '/'
    fmri = [_read_fmri_texture(sub, run, path, task) for run in range(n_runs)
            for task in range(n_tasks)]

    return fmri, stimuli


def _read_fmri_gauthier(sub, run, path):
    """ Reads BOLD signal data on Gauthier dataset """
    run = '_run00' + str(run + 1)
    fmri_path = (('{path}/BOLD/task001{run}/rabold.nii').format(
        path=path, sub=sub, run=run))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimulus_gauthier(sub, run, path, n_scans, tr, glm=False):
    """ Reads stimulus data on Gauthier dataset """
    run = '_run00' + str(run + 1) + '.npy'
    paradigm = np.load('stimuli_' + sub + run)
    onsets = paradigm[:, 0].astype('float')
    durations = paradigm[:, 1].astype('float')
    conditions = paradigm[:, -1]

    if glm:
        return onsets, durations, conditions

    cats = np.array(['rest', 'face', 'house', '1.0', '2.0', '3.0', '4.0', '5.0',
                     '6.0', '7.0'])
    stimuli = np.zeros((n_scans, len(cats)))
    for index, condition in enumerate(conditions):
        stim_onset = int(round(onsets[index]/tr))
        stimuli[stim_onset, np.where(cats == condition)[0][0]] = 1
    # Fill the rest with 'rest'
    stimuli[np.logical_not(np.sum(stimuli, axis=1).astype(bool)), 0] = 1

    return stimuli


def read_data_gauthier(subject, n_runs=2, tr=1.5, n_scans=403, high_pass=0.01,
    path=('/home/loula/Programming/python/neurospin/gauthier2009resonance/'
          'results/')):
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
        sub = 'sub00' + str(subject + 1)
    else:
        sub = 'sub0' + str(subject + 1)
    path += sub

    stimuli = [_read_stimulus_gauthier(sub, run, path, n_scans, tr)
               for run in range(n_runs)]

    glm_stimuli = np.array(
        [_read_stimulus_gauthier(sub, run, path, n_scans, tr, glm=True)
         for run in range(n_runs)])
    onsets, durations, conditions = (glm_stimuli[:, 0, :].astype(float),
                                     glm_stimuli[:, 1, :].astype(float),
                                     glm_stimuli[:, 2, :])

    fmri = [_read_fmri_gauthier(sub, run, path) for run in range(n_runs)]
    fmri = _uniform_masking(fmri, high_pass=high_pass)

    session_id_fmri = [[session] * len(fmri[session])
                       for session in range(len(fmri))]
    session_id_onset = [[session] * len(onsets[session])
                        for session in range(len(onsets))]

    return (fmri, stimuli, onsets, conditions, durations, session_id_fmri,
            session_id_onset)


def _read_fmri_mrt(sub, run, path):
    """ Reads BOLD signal data on Mirror-Reversed Text dataset """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    fmri_path = (('{path}/ses-test/func/r{sub}_ses-test_{task}'
                  '_{run}_bold.nii')
                 .format(path=path, sub=sub, task=task, run=run))
    fmri = load_img(fmri_path)

    return fmri


def _read_stimulus_mrt(sub, run, path, n_scans, tr, two_classes, glm=False):
    """ Reads stimulus data on Mirror-Reversed Text dataset """
    run = 'run-0' + str(run + 1)
    task = 'task-livingnonlivingdecisionwithplainormirrorreversedtext'
    onsets_path = (('{path}/ses-test/func/{sub}_ses-test_{task}'
                    '_{run}_events.tsv')
                   .format(path=path, sub=sub, task=task, run=run))

    paradigm = pd.read_csv(onsets_path, sep='\t')
    onsets, durations, conditions = (paradigm['onset'], paradigm['duration'],
                                     paradigm['trial_type'])
    if glm:
        return onsets, conditions

    cats = np.array(['rest', 'junk', 'pl_ns', 'pl_sw', 'mr_ns', 'mr_sw'])
    stimuli = np.zeros((n_scans, len(cats)))
    for index, condition in enumerate(conditions):
        stim_onset = int(onsets.loc[index])/tr
        stim_duration = 1 + int(durations.loc[index])/tr
        (stimuli[stim_onset: stim_onset + stim_duration,
                 np.where(cats == condition)[0][0]]) = 1
    # Fill the rest with 'rest'
    stimuli[np.logical_not(np.sum(stimuli, axis=1).astype(bool)), 0] = 1

    if two_classes:
        stimuli = np.array([stimuli[:, 0],
                            stimuli[:, 2] + stimuli[:, 3],
                            stimuli[:, 4] + stimuli[:, 5],
                            stimuli[:, 1]]).T

    return stimuli


def read_data_mrt(subject, n_runs=6, tr=2, n_scans=205, two_classes=False,
    path='/home/loula/Programming/python/neurospin/ds006_R2.0.0/results/',
    glm=False):
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

    os.chdir(path + "/ses-test/func/")
    runs = []
    for run in range(n_runs):
        if glob.glob("*run-0" + str(run + 1) + "*"):
            runs.append(run)

    stimuli = [_read_stimulus_mrt(sub, run, path, n_scans, tr, two_classes, glm)
               for run in runs]

    fmri = [_read_fmri_mrt(sub, run, path) for run in runs]

    labels = []
    for run_n in range(len(runs)):
        if glm:
            labels += [runs[run_n]] * len(stimuli[run_n][0])
        else:
            labels += [runs[run_n]] * n_scans

    return fmri, stimuli, labels
