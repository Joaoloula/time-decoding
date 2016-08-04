import numpy as np


def response_estimation(fmri, stimuli, time_window):
    """ Takes the BOLD signal and the one-hot encoded stimuli and returns the
    stereotypical response by averaging across all stimulus onsets """
    n_classes = stimuli.shape[1]
    n_scans, n_voxels = fmri.shape

    responses = np.empty((n_classes, n_voxels, time_window))
    for category in range(n_classes):
        all_responses = [fmri[scan: scan + time_window]
                         for scan in range(n_scans)
                         if stimuli[scan][category] == 1]
        average_response = np.sum(all_responses, axis=0) / len(all_responses)
        responses[category] = average_response

    return responses



