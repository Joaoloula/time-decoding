import pandas as pd
import numpy as np
import pickle


def normalize_haxby():
    data = pickle.load(open('haxby_all.pickle', 'rb'))
    new_data = pd.DataFrame([])
    models = ['GLM', 'GLMs', 'spatiotemporal SVM', 'logistic deconvolution']
    for subject in range(1, 7):
        mean = np.mean(data.loc[data['subject'] == subject]['accuracy'])
        for model in models:
            model_acc = np.mean(data.loc[np.logical_and(
                data['subject'] == subject, data['model'] == model)]
                ['accuracy'])

            new_row = {}
            new_row['accuracy'] = (model_acc - mean)/mean
            new_row['dataset'] = 'haxby'
            new_row['model'] = model
            new_row['subject'] = subject

            new_data = new_data.append(new_row, ignore_index=True)

    return new_data


def normalize_mrt():
    data = pickle.load(open('mrt_all_dataframe.pickle', 'rb'))
    new_data = pd.DataFrame([])
    models = ['GLM', 'GLMs', 'spatiotemporal SVM', 'logistic deconvolution']
    for subject in range(1, 15):
        mean = np.mean(data.loc[data['subject'] == subject]['accuracy'])
        for model in models:
            model_acc = np.mean(data.loc[np.logical_and(
                data['subject'] == subject, data['model'] == model)]
                ['accuracy'])

            new_row = {}
            new_row['accuracy'] = (model_acc - mean)/mean
            new_row['dataset'] = 'mirror-reversed text'
            new_row['model'] = model
            new_row['subject'] = subject

            new_data = new_data.append(new_row, ignore_index=True)

    return new_data


def normalize_texture():
    data = pickle.load(open('texture_all.pickle', 'rb'))
    new_data = pd.DataFrame([])
    models = ['GLM', 'GLMs', 'spatiotemporal SVM', 'logistic deconvolution']
    for subject in range(7):
        mean = np.mean(data.loc[data['subject'] == subject]['accuracy'])
        for model in models:
            model_acc = np.mean(data.loc[np.logical_and(
                data['subject'] == subject, data['model'] == model)]
                ['accuracy'])

            new_row = {}
            new_row['accuracy'] = (model_acc - mean)/mean
            new_row['dataset'] = 'texture decoding'
            new_row['model'] = model
            new_row['subject'] = subject

            new_data = new_data.append(new_row, ignore_index=True)

    return new_data


def normalize_gauthier():
    data = pickle.load(open('all_gauthier_separate.pickle', 'rb'))
    new_data = pd.DataFrame([])
    models = ['GLM', 'GLMs', 'spatiotemporal SVM', 'logistic deconvolution']
    isis = [1.6, 3.2, 4.8]
    for subject in range(1, 12):
        for isi in isis:
            mean = np.mean(data.loc[np.logical_and(data['subject'] == subject,
                                                   data['isi'] == isi)]
                           ['accuracy'])
            for model in models:
                model_acc = np.mean(data.loc[np.logical_and(np.logical_and(
                    data['subject'] == subject, data['model'] == model),
                    data['isi'] == isi)]['accuracy'])

                new_row = {}
                new_row['accuracy'] = (model_acc - mean)/mean
                new_row['dataset'] = 'temporal tuning'
                new_row['model'] = model
                new_row['subject'] = subject
                new_row['isi'] = isi

                new_data = new_data.append(new_row, ignore_index=True)

    return new_data


def average_gauthier():
    data = pickle.load(open('all_gauthier_separate.pickle', 'rb'))
    new_data = pd.DataFrame([])
    models = ['GLM', 'GLMs', 'spatiotemporal SVM', 'logistic deconvolution']
    isis = [1.6, 3.2, 4.8]
    for subject in range(1, 12):
        for isi in isis:
            for model in models:
                model_acc = np.mean(data.loc[np.logical_and(np.logical_and(
                    data['subject'] == subject, data['model'] == model),
                    data['isi'] == isi)]['accuracy'])

                new_row = {}
                new_row['accuracy'] = model_acc
                new_row['dataset'] = 'temporal tuning'
                new_row['model'] = model
                new_row['subject'] = subject
                new_row['isi'] = isi

                new_data = new_data.append(new_row, ignore_index=True)

    return new_data
