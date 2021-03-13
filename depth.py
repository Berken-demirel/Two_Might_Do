from keras.models import Sequential
from matplotlib import pyplot  # To be deleted
from sklearn.model_selection import train_test_split  # To be deleted
from scipy import signal
import pandas as pd
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, AveragePooling1D, SeparableConv1D, Dropout, Flatten, Concatenate, Reshape, \
    Activation, BatchNormalization, SeparableConv1D, Add, Activation, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam, SGD
import numpy as np, os, sys, joblib
from scipy.signal import butter, sosfiltfilt
import math
from sklearn import preprocessing
import tensorflow_addons as tfa


def assign_weigths(sums):
    if np.count_nonzero(sums) != 27:
        sums[np.where(sums == 0)[0]] = 1
    total = np.sum(sums)
    sums_div = 1 / (sums / total)
    weights_dict = {}
    for i in range(len(sums)):
        weights_dict.__setitem__(i, sums_div[i] / 27)
    return weights_dict

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 20  
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def inception_module_1(layer_in):
    conv1 = Conv1D(32, 4, padding='same', activation='relu', kernel_initializer='GlorotNormal',kernel_regularizer=l2(0.0002))(layer_in)
    conv4 = Conv1D(32, 8, padding='same', activation='relu', kernel_initializer='GlorotNormal',kernel_regularizer=l2(0.0002))(layer_in)
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
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same',kernel_regularizer=l2(0.0002))(lead_II)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_0)
    Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Incept_1, 64, 2)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    #flat = Flatten()(Pool1)
    return Pool1

def Lead_V5_way(lead_V5):
    layer_out = Conv1D(32, 8, activation='relu', kernel_initializer='GlorotNormal', padding='same',kernel_regularizer=l2(0.0002))(lead_V5)
    Batch1 = BatchNormalization()(layer_out)
    layer_out_0 = res_net_block_trans(Batch1, 32, 4)
    Pool1 = AveragePooling1D(2, padding='same')(layer_out_0)
    Incept_1 = inception_module_1(Pool1)
    res2 = res_net_block_trans(Incept_1, 64, 2)
    Pool1 = AveragePooling1D(2, padding='same')(res2)
    #flat = Flatten()(Pool1)
    return Pool1

def define_model(in_shape=(600, 1), out_shape=27):
    input_II = Input(shape=(275,1))
    input_V5 = Input(shape=(275,1))
    input_bpm = Input(shape=(6,1))
    Dense_bpm = bpm_dense(input_bpm)
    out_II = Lead_II_way(input_II)
    out_V5 = Lead_V5_way(input_V5)
    layer_out = concatenate([out_II, out_V5], axis=-1)
    sep1 = SeparableConv1D(128, 2, activation='relu', kernel_initializer='GlorotNormal', padding='same',kernel_regularizer=l2(0.0002))(layer_out)
    flat = Flatten()(sep1)
    Dense_1 = Dense(128, activation='relu')(flat)
    layer_out = concatenate([Dense_bpm, Dense_1])
    Dropout1 = Dropout(0.4)(layer_out)
    out = Dense(out_shape, activation='sigmoid')(Dropout1)
    BerkenLeNet = Model(inputs=[input_II, input_V5, input_bpm], outputs=out)
    BerkenLeNet.summary()
    # compile model
    opt = Adam(learning_rate=0.0003)
    #opt = SGD(lr=0.01, momentum=0.9, nesterov=False)
    BerkenLeNet.compile(optimizer=opt, loss='binary_crossentropy', metrics=['Recall', 'accuracy',
                                                                            tfa.metrics.F1Score(num_classes=27,
                                                                                                threshold=0.5,
                                                                                                average='macro')])
    return BerkenLeNet


a = np.load('./berken_corr/all_with_v1.npz')
training_data = a['arr_0']
labels = a['arr_1']

Xtrain = training_data

x_train, x_val, y_train, y_val = train_test_split(Xtrain, labels, stratify=labels, test_size=0.2, random_state=1)

Xtrain_1 = x_train[:,0:550]
Xtrain_II = x_train[:,0:275]
Xtrain_V5 = x_train[:,275:550]
bpm_data_train = x_train[:,550:]

xval = x_val[:,0:550]
xval_II = x_val[:,0:275]
xval_V5 = x_val[:,275:550]
bpm_data_val = x_val[:,550:]

sum_26 = np.sum(y_train, axis=0)

class_weights = assign_weigths(sum_26)

model = define_model()

checkpoint_filepath = 'Checkpoints_all_with_v1'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                               monitor='val_f1_score', mode='max', save_best_only=True)

stop_me = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', min_delta=0, patience=100, verbose=1, mode='max',
                                           baseline=None, restore_best_weights=True)

where_am_I = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.1, patience=75, verbose=1,
                                                  mode='max', min_delta=0.001, cooldown=0, min_lr=0)

history = model.fit(x=[Xtrain_II, Xtrain_V5, bpm_data_train], y=y_train, epochs=900, batch_size=250, verbose=1, validation_data=([xval_II, xval_V5, bpm_data_val], y_val), class_weight=class_weights, callbacks=[model_checkpoint_callback, stop_me, where_am_I])

hist_df = pd.DataFrame(history.history)
pd.DataFrame.from_dict(history.history).to_csv('all_with_v1.csv', index=False)
    
model.save(checkpoint_filepath)