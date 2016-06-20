# time-decoding

Current approaches to decoding fMRI signals rely on deconvolution to summarize brain activation data, thus getting rid of the temporal structure of both the hemodynamic response and the noise. We propose a new framework for decoding in the time domain that seeks to exploit these features of the data.

##Contrast Function

Suppose we have two different kinds of stimuli, say, a picture of a face and one of a house. If we present these stimuli to a subject in an alternate manner, interspersed with resting periods, we can represent the subject's stimulation as a function of time by a contrast function with, for example, the value +1 meaning 'face', -1 meaning 'house' and 0 meaning 'rest'

We can then consider the task of classifying the stimulus at a given time as 'face' or 'house' the task of finding an approximation for the function described above (we should note that the latter is a slightly harder task, since we also have to account for the state 'rest').

##Predicting many classes

When treating cases where the number of classes is n>2, it is necessary to substitute the contrast functions for n step-functions representing the apparition of each of the stimuli.

In order to perform classification in this setting, we can use multinomial logistic regression to decide between the given categories (including 'rest') at each time-step.

##Accounting for the HRF

The prediction of a step-function is, of course, a gross simplification: the blood flow in the brain caused by a stimulus is not an immediate activation that stays constant in time, but rather a dynamic with a slow onset followed by a brusque undershoot: the Hemodynamic Response Function. We can use the HRF to link the presentation of the stimuli to the brain's response over time, and then perform our regression with respect to the signal we obtain.

Seeing as the HRF varies between subjects, we will estimate it for each subject by using a variation of the method introduced by [1].

[1] Fabian Pedregosa, Michael Eickenberg, Philippe Ciuciu, Bertrand Thirion, Alexandre Gramfort.
Data-driven HRF estimation for encoding and decoding models. NeuroImage, 2015.
