import numpy as np

for subject in range(1, 12):
    if subject <= 9:
        sub = 'sub00' + str(subject)
    else:
        sub = 'sub0' + str(subject)

    path = ('/home/loula/Programming/python/neurospin/gauthier2009resonance/'
            '{sub}/model/model001/onsets/task001_run001/'.format(sub=sub))
    all_stimuli = []
    for condition in range(1, 11):
        if condition <= 9:
            cond = 'cond00' + str(condition) + '.txt'
        else:
            cond = 'cond0' + str(condition) + '.txt'

        stim = np.loadtxt(path + cond)

        # Treat high frequencies as events in and of themselves
        if condition not in [8, 9, 10]:
            labels = [[condition], [condition]]

        # Special case for the three frequencies of interest
        else:
            if condition == 8:
                duration = 1.6
            elif condition == 9:
                duration = 3.2
            else:
                duration = 4.8

            # Create 'face' and 'house' labels and corresponding onsets
            repeats = int(round(19.2 / duration))
            stim = np.repeat(stim, repeats, axis=0)
            labels = [['face'], ['house']] * repeats
            for repeat in range(repeats):
                stim[repeat, 0] += repeat * duration
                stim[repeat, 1] = duration
                stim[len(stim)/2 + repeat, 0] += repeat * duration
                stim[len(stim)/2 + repeat, 1] = duration

        stim = np.hstack((stim, labels))

        all_stimuli.append(stim)

    all_stimuli = np.vstack(all_stimuli)
    all_stimuli = all_stimuli[all_stimuli[:, 0].astype('float').argsort()]
    np.save('stimuli_' + sub + '_run001.npy', all_stimuli)
