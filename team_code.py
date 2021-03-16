#!/usr/bin/env python

# Edit this script to add your team's training code.

from ecgdetectors import Detectors
from helper_code import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split # To be deleted
from scipy import signal
from biosppy.signals import ecg
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, Dropout, Flatten, Concatenate, Reshape, Activation, BatchNormalization, SeparableConv1D, Add
from keras.models import Model
from keras.regularizers import l2
import tensorflow_addons as tfa
from keras.optimizers import Adam
import numpy as np, os, sys, joblib
from scipy.signal import butter, sosfiltfilt
import math
from sklearn import preprocessing
from keras import backend

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

others = 'others'
two_lead_model_filename = '2_lead_model'


# Train your model. This function is **required**. Do **not** change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    scored_labels = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002',
                     '39732003',
                     '164909002', '251146004', '698252002', '10370003', '284470004', '427172004',
                     '164947007', '111975006', '164917005', '47665007', '59118001', '427393009',
                     '426177001', '426783006', '427084000', '63593006', '164934002', '59931005', '17338001']

    my_scored_labels = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002',
                     '39732003',
                     '164909002', '251146004', '698252002', '10370003', '284470004', '427172004',
                     '164947007', '111975006', '164917005', '47665007', '427393009',
                     '426177001', '426783006', '427084000', '164934002', '59931005']

    # dict that maps labels to integers, and the reverse
    labels_map = {scored_labels[i]: i for i in range(len(scored_labels))}
    my_labels = {my_scored_labels[i]: i for i in range(len(my_scored_labels))}

    # Extract features and labels from dataset.
    print('Extracting features and labels...')
    training_list = []
    training_labels = []
    bpm_list = []
    for i in range(num_recordings):
        print('    {}/{}...'.format(i + 1, num_recordings))
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        current_labels = get_labels(header)
        true_labels = list(set(scored_labels) & set(current_labels))
        if len(true_labels) != 0:
            processed_1, bpm_feat = get_features_2(header, recording, ['II', 'V5'])
            encoded_label = one_hot_encode(true_labels, labels_map, my_labels)
            for x in range(len(processed_1)):
                if len(processed_1[x]) == 550:
                    training_list.append(processed_1[x])
                    bpm_list.append(bpm_feat)
                    training_labels.append(encoded_label)
                else:
                    pass

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')
    training_data = np.zeros((len(training_list),550))
    bpm_data = np.zeros((len(bpm_list), 6))
    for i in range(len(training_list)):
        if len(training_list[i]) == 550:
            training_data[i,:] = training_list[i]
            bpm_data[i, :] = bpm_list[i]
        else:
            pass
    labels = np.array(training_labels)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    training_data = np.concatenate((training_data, bpm_data), axis=1)

    Xtrain = training_data

    x_train, x_val, y_train, y_val = train_test_split(Xtrain, labels, stratify=labels, test_size=0.2, random_state=1)


    Xtrain_II = x_train[:, 0:275]
    Xtrain_V5 = x_train[:, 275:550]
    bpm_data_train = x_train[:, 550:]

    xval_II = x_val[:, 0:275]
    xval_V5 = x_val[:, 275:550]
    bpm_data_val = x_val[:, 550:]

    sum_26 = np.sum(y_train, axis=0)

    class_weights = assign_weigths(sum_26)

    model = define_model()

    checkpoint_filepath = 'Checkpoints'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=False,
                                                                   monitor='val_f1_score', mode='max',
                                                                   save_best_only=True)

    stop_me = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=100, verbose=1, mode='max',
                                               baseline=None, restore_best_weights=True)

    where_am_I = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.1, patience=75, verbose=1,
                                                      mode='max', min_delta=0.001, cooldown=0, min_lr=0)

    history = model.fit(x=[Xtrain_II, bpm_data_train], y=y_train, epochs=1, batch_size=250, verbose=1,
                        validation_data=([xval_II, bpm_data_val], y_val),
                        class_weight=class_weights, callbacks=[model_checkpoint_callback, stop_me, where_am_I])

    model2 = define_model2()


    history2 = model2.fit(x=[Xtrain_II, Xtrain_V5, bpm_data_train], y=y_train, epochs=900, batch_size=250, verbose=1,
                         validation_data=([xval_II, xval_V5, bpm_data_val], y_val), class_weight=class_weights, callbacks=[model_checkpoint_callback, stop_me, where_am_I])


    filename1 = os.path.join(model_directory,others)
    filename2 = os.path.join(model_directory, two_lead_model_filename)

    model.save(filename1)
    model2.save(filename2)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, others)
    return load_model_2(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory,others)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory,others)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model_2(filename)

# Generic function for loading a model.
def load_model(filename):
    return tf.keras.models.load_model(filename, custom_objects={"F1Score": tfa.metrics.F1Score})

def load_model_2(filename):
    return tf.keras.models.load_model(filename, custom_objects={"F1Score": tfa.metrics.F1Score})

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model2(model, header, recording)

# Run your trained 6-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model2(model, header, recording)

# Generic function for running a trained model.
def run_model2(model, header, recording):
    scored_labels = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002',
                     '39732003',
                     '164909002', '251146004', '698252002', '10370003', '284470004', '427172004',
                     '164947007', '111975006', '164917005', '47665007', '59118001', '427393009',
                     '426177001', '426783006', '427084000', '63593006', '164934002', '59931005', '17338001']
    leads = ['II','V5']
    classifier = model

    # Load features.
    splitted, bpm_features = get_features_2(header, recording, leads)
    bpm_data = np.zeros((len(splitted), 6))
    for i in range(len(splitted)):
        bpm_data[i, :] = bpm_features[0:6]

    a = splitted.reshape(len(splitted), 550, 1)
    a1 = a[:,0:275]
    a2 = a[:,275:]
    try:
        probabilities1 = classifier.predict([a1,a2, bpm_data])
    except ValueError:
        return scored_labels, np.array([1]), np.array([0.6])

    row_index = np.sum(probabilities1,axis=0).argmax()
    probabilities2 = np.sum(probabilities1,axis=0) / np.sum(probabilities1,axis=0)[row_index]
    a = np.zeros(24)
    a[np.argwhere(probabilities2 > 0.25)] = 1
    label1, probabilities3 = convert_to_real_labels(a, np.round(probabilities2, 3))
    a1 = np.asarray(label1, dtype=int)

    return scored_labels, a1, probabilities3

def run_model(model, header, recording):
    scored_labels = ['270492004', '164889003', '164890007', '426627000', '713427006', '713426002', '445118002',
                     '39732003',
                     '164909002', '251146004', '698252002', '10370003', '284470004', '427172004',
                     '164947007', '111975006', '164917005', '47665007', '59118001', '427393009',
                     '426177001', '426783006', '427084000', '63593006', '164934002', '59931005', '17338001']
    leads = ['II']
    classifier = model

    splitted, bpm_features = get_features(header, recording, leads)
    bpm_data = np.zeros((len(splitted), 6))
    for i in range(len(splitted)):
        bpm_data[i, :] = bpm_features[0:6]

    try:
        probabilities1 = classifier.predict([splitted.reshape(len(splitted), 275, 1), bpm_data])
    except ValueError:
        return scored_labels, np.array([1]), np.array([0.6])

    row_index = np.sum(probabilities1,axis=0).argmax()
    probabilities2 = np.sum(probabilities1,axis=0) / np.sum(probabilities1,axis=0)[row_index]
    a = np.zeros(24)
    a[np.argwhere(probabilities2 > 0.25)] = 1
    label1, probabilities3 = convert_to_real_labels(a, np.round(probabilities2, 3))
    a1 = np.asarray(label1, dtype=int)

    return scored_labels, a1, probabilities3

################################################################################
#
# Other functions
#
################################################################################
def butter_bandpass(lowcut, highcut, fs, order=20):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data,freq):
    lowcut, highcut, fs, order = 2, 35, freq, 30
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

def one_hot_encode(tags, mapping, my_labels):
    my_encoding = np.zeros(len(my_labels), dtype='uint8')
    for tag in tags:
        if '59118001' in tag:
            tag = '713427006'
        if '63593006' in tag:
            tag = '284470004'
        if '17338001' in tag:
            tag = '427172004'
        my_encoding[my_labels[tag]] = 1
    return my_encoding

def convert_to_real_labels(labels, probs):
    probs_to_return = np.zeros(27)
    labels_to_return = np.zeros(27)
    probs_to_return[0:18] = probs[0:18]
    labels_to_return[0:18] = labels[0:18]
    probs_to_return[18] = probs[4]
    labels_to_return[18] = labels[4]
    probs_to_return[19:23] = probs[18:22]
    labels_to_return[19:23] = labels[18:22]
    probs_to_return[23] = probs[12]
    labels_to_return[23] = labels[12]
    probs_to_return[24:26] = probs[22:24]
    labels_to_return[24:26] = labels[22:24]
    probs_to_return[26] = probs[13]
    labels_to_return[26] = labels[13]
    return labels_to_return, probs_to_return

def assign_weigths(sums):
    if np.count_nonzero(sums) != 27:
        sums[np.where(sums == 0)[0]] = 1
    total = np.sum(sums)
    sums_div = 1 / (sums / total)
    weights_dict = {}
    for i in range(len(sums)):
        weights_dict.__setitem__(i, sums_div[i] / 27)
    return weights_dict


def define_model(in_shape=(300, 1), out_shape=24):
    inputA = Input(shape=(275,1))
    inputB = Input(shape=(6,1))
    Dense_bpm = bpm_dense(inputB)
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same', kernel_regularizer=l2(0.0002))(inputA)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Incept_1, 64, 2)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    flat_1 = Flatten()(Pool1)
    Dense_1 = Dense(128, activation='relu', kernel_initializer='GlorotNormal')(flat_1)
    layer_out = concatenate([Dense_bpm, Dense_1])
    Dropout1 = Dropout(0.4)(layer_out)
    out = Dense(out_shape, activation='sigmoid')(Dropout1)
    BerkenLeNet = Model(inputs=[inputA, inputB], outputs=out)
    BerkenLeNet.summary()
    # compile model
    opt = Adam(learning_rate=0.0003)
    BerkenLeNet.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy', 'Recall',tfa.metrics.F1Score(num_classes=24,threshold=0.5,average='macro')])
    return BerkenLeNet


def inception_module_1(layer_in):
    conv1 = Conv1D(32, 4, padding='same', activation='relu', kernel_initializer='GlorotNormal',
                   kernel_regularizer=l2(0.0002))(layer_in)
    conv4 = Conv1D(32, 16, padding='same', activation='relu', kernel_initializer='GlorotNormal',
                   kernel_regularizer=l2(0.0002))(layer_in)
    layer_out = concatenate([conv1, conv4], axis=-1)
    x3 = BatchNormalization()(layer_out)
    return x3


def res_net_block1(input_data, filters, conv_size):
    x = Conv1D(filters, conv_size, activation='relu', padding='same', kernel_regularizer=l2(0.0002))(input_data)
    x = BatchNormalization()(x)
    x = Conv1D(filters, conv_size, activation=None, padding='same', kernel_regularizer=l2(0.0002))(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_data])
    x = Activation('relu')(x)
    return x


def res_net_block_trans(input_data, filters, conv_size):
    input_trans = Conv1D(filters, 1, activation='relu', padding='same', kernel_regularizer=l2(0.0002))(input_data)
    x0 = Conv1D(filters, conv_size, activation='relu', padding='same', kernel_regularizer=l2(0.0002))(input_data)
    x1 = BatchNormalization()(x0)
    x2 = Conv1D(filters, conv_size, activation=None, padding='same', kernel_regularizer=l2(0.0002))(x1)
    x3 = BatchNormalization()(x2)
    x4 = Add()([x3, input_trans])
    x = Activation('relu')(x4)
    return x


def bpm_dense(bpm_input):
    flat = Flatten()(bpm_input)
    Dense_bpm1 = Dense(8, activation="relu")(flat)
    Dense_bpm2 = Dense(16, activation="relu")(Dense_bpm1)
    Dense_bpm3 = Dense(32, activation="relu")(Dense_bpm2)
    return Dense_bpm3


def Lead_II_way(lead_II):
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same',
                       kernel_regularizer=l2(0.0002))(lead_II)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Incept_1, 64, 2)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    # flat = Flatten()(Pool1)
    return Pool1


def Lead_V5_way(lead_V5):
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same',
                       kernel_regularizer=l2(0.0002))(lead_V5)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 8)
    layer_out_1 = res_net_block_trans(layer_out_0, 64, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_1)
    Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Incept_1, 64, 2)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    # flat = Flatten()(Pool1)
    return Pool1


def define_model2(in_shape=(600, 1), out_shape=24):
    input_II = Input(shape=(275, 1))
    input_V5 = Input(shape=(275, 1))
    input_bpm = Input(shape=(6, 1))
    Dense_bpm = bpm_dense(input_bpm)
    out_II = Lead_II_way(input_II)
    out_V5 = Lead_V5_way(input_V5)
    layer_out = concatenate([out_II, out_V5], axis=-1)
    sep1 = SeparableConv1D(128, 4, activation='relu', kernel_initializer='GlorotNormal', padding='same',
                           kernel_regularizer=l2(0.0002))(layer_out)
    flat = Flatten()(sep1)
    Dense_1 = Dense(128, activation='relu')(flat)
    layer_out = concatenate([Dense_bpm, Dense_1])
    Dropout1 = Dropout(0.4)(layer_out)
    out = Dense(out_shape, activation='sigmoid')(Dropout1)
    BerkenLeNet = Model(inputs=[input_II, input_V5, input_bpm], outputs=out)
    BerkenLeNet.summary()
    # compile model
    opt = Adam(learning_rate=0.0003)
    # opt = SGD(lr=0.01, momentum=0.9, nesterov=False)
    BerkenLeNet.compile(optimizer=opt, loss='binary_crossentropy', metrics=['Recall', 'accuracy', tfa.metrics.F1Score(num_classes=24, threshold=0.5, average='macro')])
    return BerkenLeNet

def get_correlated_ones(templates):
    df = pd.DataFrame(templates.transpose())
    a = df.corr(method='pearson')
    a1 = a.sum(axis=1)
    a1 = a1 / a1[a1.argmax()]
    df = df.drop(df.columns[a1[a1<0.8].index.tolist() ], axis=1, inplace=False)
    templates = df.to_numpy().transpose()
    return templates

def get_bpm_feature(bpm_values, freq):
    if np.any(np.abs(np.diff(bpm_values)) < 30):
        bpm_values= np.delete(bpm_values,np.where(np.abs(np.diff(bpm_values))<30))
    bpm_values = 60 / (np.abs(np.diff(bpm_values))/freq)
    diff_nni = np.abs(np.diff(bpm_values))
    # Basic statistics
    mean_nni = np.mean(bpm_values)
    if mean_nni > 200:
        print('DUR')
    median_nni = np.median(bpm_values)
    range_nni = max(bpm_values) - min(bpm_values)
    sdsd = np.std(bpm_values)
    rmssd = np.sqrt(np.mean(bpm_values ** 2))
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / len(bpm_values)
    return np.array([mean_nni, median_nni, range_nni, sdsd, rmssd, pnni_20])

def get_features(header, recording, leads):
    freq = get_frequency(header)
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    amplitudes = get_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    num_samples = get_num_samples(header)
    for i in range(num_leads):
        recording[i, :] = amplitudes[i] * recording[i, :] - baselines[i]
    if freq < 500 and num_samples > 20000:
        recording = recording[0:2, 1:16000]
    features = ecg.ecg(recording[0],sampling_rate=freq, show=False)
    bpm_features = get_bpm_feature(features['rpeaks'], freq)

    if bpm_features[0] > 180:
        r_peaks_1 = ecg.extract_heartbeats(signal=recording[0], rpeaks=features['rpeaks'], sampling_rate=freq, before=0.2, after=0.2)
    else:
        r_peaks_1 = ecg.extract_heartbeats(signal=recording[0], rpeaks=features['rpeaks'], sampling_rate=freq, before=0.25, after=0.3)

    try:
        bpm_features = get_bpm_feature(features['rpeaks'], freq)
    except ValueError:
        bpm_features = np.zeros((1,6))

    templates = r_peaks_1['templates']
    df = pd.DataFrame(templates.transpose())
    a = df.corr(method='pearson')
    a1 = a.sum(axis=1)
    a1 = a1 / a1[a1.argmax()]
    df = df.drop(df.columns[a1[a1<0.8].index.tolist() ], axis=1, inplace=False)
    templates = df.to_numpy().transpose()
    if freq != 500 or len(templates[0]) != 275:
        templates = signal.resample(templates, 275, axis=-1)

    try:
        output1 = preprocessing.normalize(templates, norm="l2", axis=1)
    except ValueError:
        output1 = np.zeros(1,325)

    return output1, bpm_features


#
# Extract features from the header and recording.
def get_features_2(header, recording, leads):
    freq = get_frequency(header)
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    amplitudes = get_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    num_samples = get_num_samples(header)
    for i in range(num_leads):
        recording[i, :] = amplitudes[i] * recording[i, :] - baselines[i]
    if freq < 500 and num_samples > 20000:
        recording = recording[0:2, 1:16000]

    features_1 = ecg.ecg(recording[0],sampling_rate=freq, show=False)
    bpm_features = get_bpm_feature(features_1['rpeaks'], freq)
    if bpm_features[0] > 180:
        r_peaks_1 = ecg.extract_heartbeats(signal=recording[0], rpeaks=features_1['rpeaks'], sampling_rate=freq, before=0.2, after=0.2)
    else:
        r_peaks_1 = ecg.extract_heartbeats(signal=recording[0], rpeaks=features_1['rpeaks'], sampling_rate=freq, before=0.25, after=0.3)

    try:
        features_2 = ecg.ecg(recording[1],sampling_rate=freq, show=False)
    except ValueError:
        features_2 = ecg.ecg(recording[0],sampling_rate=freq, show=False)

    if bpm_features[0] > 180:
        r_peaks_2 = ecg.extract_heartbeats(signal=recording[1], rpeaks=features_2['rpeaks'], sampling_rate=freq, before=0.2, after=0.2)
    else:
        r_peaks_2 = ecg.extract_heartbeats(signal=recording[1], rpeaks=features_2['rpeaks'], sampling_rate=freq, before=0.25, after=0.3)

    templates = r_peaks_1['templates']
    templates_2 = r_peaks_2['templates']
    templates = get_correlated_ones(templates)
    templates_2 = get_correlated_ones(templates_2)
    if freq != 500 or len(templates[0]) != 275:
        templates = signal.resample(templates, 275, axis=-1)
        templates_2 = signal.resample(templates_2, 275, axis=-1)

    if len(templates_2) != len(templates):
        a = len(templates_2) - len(templates)
        if a > 0:
            temp_temp = np.sum(templates, axis=0)/len(templates)
            while a > 0:
                templates = np.concatenate((templates, np.reshape(temp_temp,(1,len(temp_temp)))), axis=0)
                a = a - 1
        elif a < 0:
            temp_temp = np.sum(templates_2, axis=0)/len(templates_2)
            while a < 0:
                templates_2 = np.concatenate((templates_2, np.reshape(temp_temp,(1,len(temp_temp)))), axis=0)
                a = a + 1
    try:
        output1 = preprocessing.normalize(templates, norm="l2", axis=1)
        output2 = preprocessing.normalize(templates_2, norm="l2", axis=1)
        output = np.concatenate((output1, output2), axis=1)
    except ValueError:
        output = np.zeros(1,550)

    return output, bpm_features