from nilearn.input_data import NiftiMasker
import numpy as np


def read_data(subject, haxby_dataset, n_scans):
    """Returns indiviudal images, labels and session id for subject subject"""
    # Read labels
    labels = np.recfromcsv(haxby_dataset.session_target[subject], delimiter=" ")
    sessions_id = labels['chunks']
    target = labels['labels']
    categories = np.unique(target)
    # Make 'rest' be the first category in the list
    categories = np.roll(categories,
                         len(categories) - np.where(categories == 'rest')[0])

    # Initialize series array
    series_ = np.zeros(n_scans)
    for c, category in enumerate(categories):
        series_[target == category] = c

    # Read activity data
    # Standardize and detrend
    mask_filename = haxby_dataset.mask_vt[subject]
    nifti_masker = NiftiMasker(mask_img=mask_filename, standardize=True,
                               detrend=True, sessions=sessions_id)
    func_filename = haxby_dataset.func[subject]
    # fmri[str(subject)] = nifti_masker.fit_transform(func_filename)
    # series[str(subject)] = series_
    return (nifti_masker.fit_transform(func_filename), series_,
            sessions_id, categories)


def conditions_onsets(series, categories, tr):
    """ Creates conditions and onsets from series, categories, number of scans
    and temporal resolution """
    onsets = []
    con_id = []
    n_scans = len(series)
    for scan in range(1, n_scans):
        for category in range(len(categories)):
            if series[scan] == category and series[scan - 1] != category:
                onsets.append(scan*tr)
                con_id.append(categories[category])
    return con_id, onsets
