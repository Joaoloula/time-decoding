from time_decoding.data_reading import read_data_gauthier
import time_decoding.decoding as de
import pandas as pd
import numpy as np
import itertools

# Parameters
subject_list = range(11)
k = 10000
tr = 1.5
model = 'GLM'
n_tests = 100

# GLM parameters
hrf_model = 'spm'

scores, subjects, models, isis = [], [], [], []
for subject in subject_list:
    # Read data
    fmri, stimuli, onsets, conditions = read_data_gauthier(subject)
    session_id_onset = np.load('sessions_id_onset.npy')
    session_id_onset2 = [[round(19.2 / len(onsets[session]), 2)] *
                         len(onsets[session]) for session in range(len(onsets))]
    isi_id = [round(19.2 / len(onsets[session]), 2)
              for session in range(len(onsets))]
    betas, reg = de.glm(fmri, tr, onsets, durations=session_id_onset,
                        hrf_model=hrf_model, drift_model='blank', model=model)

    betas = np.vstack(betas)
    conditions = np.hstack(conditions)
    session_id_onset = np.hstack(session_id_onset)

    combinations_face = [
        [item for item in
         itertools.combinations(np.where(session_id_onset == block)[0][::2], 2)]
        for block in range(12)]

    combinations_house = [
        [item for item in
         itertools.combinations(np.where(session_id_onset == block)[0][1::2],
                                2)]
        for block in range(12)]

    blocks = [0, 0]
    while blocks[0] < 11:
        if blocks[1] < 11:
            blocks[1] += 1
        else:
            blocks[0] += 1
            blocks[1] = blocks[0] + 1

        # Split into train and test sets
        test_index = np.union1d(np.where(session_id_onset == blocks[0])[0],
                                np.where(session_id_onset == blocks[1])[0])
        conditions_test, betas_test = conditions[test_index], betas[test_index]

        n_points = len(conditions_test)
        if n_points == 12 * 2:
            isi = 1.6

        elif n_points == 6 * 2:
            isi = 3.2

        elif n_points == 4 * 2:
            isi = 4.8

        else:
            continue

        balanced_isi_id = np.delete(isi_id, blocks)
        new_combinations_face = np.delete(combinations_face, blocks)
        new_combinations_house = np.delete(combinations_house, blocks)

        block_score = 0
        for train_iteration in range(n_tests):
            """
            balanced_trials = np.union1d(
                np.random.choice(np.where(balanced_isi_id == 1.6)[0], 2, False),
                np.union1d(
                    np.random.choice(np.where(balanced_isi_id == 3.2)[0], 2,
                                     False),
                    np.random.choice(np.where(balanced_isi_id == 4.8)[0], 2,
                                     False)))
            """
            balanced_trials = np.where(balanced_isi_id == isi)
            balanced_combinations_face = new_combinations_face[balanced_trials]
            balanced_combinations_house = new_combinations_house[
                balanced_trials]
            train_index = np.hstack([np.union1d(
                balanced_combinations_face[trial][np.random.randint(len(
                    balanced_combinations_face[trial]))],
                balanced_combinations_house[trial][np.random.randint(len(
                    balanced_combinations_house[trial]))])
                for trial in range(len(balanced_combinations_face))])
            betas_train, conditions_train = (betas[train_index],
                                             conditions[train_index])
            # Feature selection
            selected_betas_train, selected_betas_test = de.feature_selection(
                betas_train, betas_test, conditions_train, k=k)

            # Fit a logistic regression to score the model
            accuracy = de.glm_scoring(selected_betas_train, selected_betas_test,
                                      conditions_train, conditions_test)

            block_score += accuracy

        block_score /= n_tests

        scores.append(block_score)
        subjects.append(subject + 1)
        models.append(model)
        isis.append(isi)
        print('Score for isi of {isi}: {score}'.format(isi=isi,
                                                       score=block_score))

    print('finished subject ' + str(subject))

dict = {}
dict['subject'] = subjects
dict['model'] = models
dict['accuracy'] = scores
dict['isi'] = isis

data = pd.DataFrame(dict)
print(np.mean(data.loc[data['isi'] == 1.6]['accuracy']))
print(np.mean(data.loc[data['isi'] == 3.2]['accuracy']))
print(np.mean(data.loc[data['isi'] == 4.8]['accuracy']))
