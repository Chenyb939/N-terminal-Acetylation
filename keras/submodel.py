import dp
import time
import pickle
import matplotlib
import numpy as np
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Convolution1D
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, LearningRateScheduler, History, TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten, Input, LSTM
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2
import keras.metrics
import matplotlib.pyplot as plt
from keras.optimizers import Nadam, Adam, SGD
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import configparser
from sklearn.model_selection import train_test_split, StratifiedKFold
from LossCheckPoint import LossModelCheckpoint
from keras.models import load_model


def OnehotNetwork(trainX, trainY, valX, valY, Oneofkey_input, folds, train_time=None):

    if (train_time == 0):
        # x = conv.Convolution1D(201, 2, init='glorot_normal', W_regularizer=l1(0), border_mode="same")(Oneofkey_input)
        # x = Dropout(0.4)(x)
        # x = Activation('softsign')(x)

        x = conv.Convolution1D(101, 3, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(Oneofkey_input)
        x = Dropout(0.4)(x)
        x = Activation('relu')(x)

        x = conv.Convolution1D(101, 5, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = Activation('relu')(x)

        x = conv.Convolution1D(101, 7, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = Activation('relu')(x)

        x = core.Flatten()(x)
        x = BatchNormalization()(x)

        # x = Dense(256, init='glorot_normal', activation='relu')(x)
        # x = Dropout(0.3)(x)

        x = Dense(128, init='glorot_normal', activation="relu")(x)
        x = Dropout(0)(x)

        Oneofkey_output = Dense(100, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)

        OnehotNetwork = Model(Oneofkey_input, Oneofkey_output)
        optimization = 'Nadam'
        OnehotNetwork.compile(loss='categorical_crossentropy', optimizer=optimization)
    else:
        OnehotNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'OnehotNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Onehotweight.h5',
                                              verbose=1, save_best_only=True, monitor='val_loss', mode='min',
                                              save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        loss_checkpointer = LossModelCheckpoint(model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'OnehotNetwork.h5',
                                                monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Onehotloss.json',
                                                verbose=1, save_best_only=True, monitor='val_loss', mode='min')
        fitHistory = OnehotNetwork.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2, validation_data=(valX, valY), shuffle=True,
                                       class_weight='auto', callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return OnehotNetwork


def AlphaturnpropensityNetwork(trainX, trainY, valX, valY, physical_A_input, folds, train_time=None):

    if (train_time == 0):
        x = conv.Convolution1D(201, 2, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(physical_A_input)
        x = Dropout(0.4)(x)
        x = Activation('softsign')(x)

        x = conv.Convolution1D(151, 3, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.3)(x)
        x = Activation('softsign')(x)

        x = conv.Convolution1D(101, 5, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.2)(x)
        x = Activation('softsign')(x)

        x = conv.Convolution1D(51, 7, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.1)(x)
        x = Activation('softsign')(x)

        x = core.Flatten()(x)
        x = BatchNormalization()(x)
        physical_A_output = Dense(100, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)

        AlphaturnpropensityNetwork = Model(physical_A_input, physical_A_output)
        optimization = keras.optimizers.Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        AlphaturnpropensityNetwork.compile(loss='categorical_crossentropy', optimizer=optimization)
    else:
        AlphaturnpropensityNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'AlphaturnpropensityNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Alphaturnpropensityweight.h5', verbose=1,
                                              save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=70)
        loss_checkpointer = LossModelCheckpoint(model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'AlphaturnpropensityNetwork.h5',
                                                monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Alphaturnpropensityloss.json',
                                                verbose=1, save_best_only=True, monitor='val_loss', mode='min')

        fitHistory = AlphaturnpropensityNetwork.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2,
                                                    validation_data=(valX, valY), shuffle=True, class_weight='auto',
                                                    callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return AlphaturnpropensityNetwork


def BetapropensityNetwork(trainX, trainY, valX, valY, physical_B_input, folds, train_time=None):

    if (train_time == 0):
        x = conv.Convolution1D(201, 2, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(physical_B_input)
        x = Dropout(0.3)(x)
        x = Activation('softsign')(x)

        x = conv.Convolution1D(151, 3, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.2)(x)
        x = Activation('softsign')(x)

        x = conv.Convolution1D(101, 5, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.1)(x)
        x = Activation('softsign')(x)

        x = core.Flatten()(x)
        x = BatchNormalization()(x)
        physical_B_output = Dense(100, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)

        BetapropensityNetwork = Model(physical_B_input, physical_B_output)

        optimization = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        BetapropensityNetwork.compile(loss='categorical_crossentropy', optimizer=optimization)
    else:
        BetapropensityNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'BetapropensityNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Betapropensityweight.h5',
                                              verbose=1,
                                              save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=60)
        loss_checkpointer = LossModelCheckpoint(
                                            model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'BetapropensityNetwork.h5',
                                            monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Betapropensityloss.json',
                                            verbose=1, save_best_only=True, monitor='val_loss', mode='min')

        fitHistory = BetapropensityNetwork.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2,
                                           validation_data=(valX, valY), shuffle=True, class_weight='auto',
                                           callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return BetapropensityNetwork


def CompositionNetwork(trainX, trainY, valX, valY, physical_C_input, folds, train_time=None):

    if (train_time == 0):
        x = conv.Convolution1D(101, 2, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(physical_C_input)
        x = Dropout(0.3)(x)
        x = Activation('relu')(x)

        x = conv.Convolution1D(101, 3, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.3)(x)
        x = Activation('relu')(x)

        x = core.Flatten()(x)
        x = BatchNormalization()(x)

        # x = Dense(64, init='glorot_normal', activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.1)(x)

        physical_C_output = Dense(100, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)

        CompositionNetwork = Model(physical_C_input, physical_C_output)

        optimization = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

        CompositionNetwork.compile(loss='categorical_crossentropy', optimizer=optimization)

    else:
        CompositionNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'CompositionNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Compositionweight.h5',
                                              verbose=1,
                                              save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=200)
        loss_checkpointer = LossModelCheckpoint(
                                                model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'CompositionNetwork.h5',
                                                monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Compositionloss.json',
                                                verbose=1, save_best_only=True, monitor='val_loss', mode='min')

        fitHistory = CompositionNetwork.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2,
                                            validation_data=(valX, valY), shuffle=True, class_weight='auto',
                                            callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return CompositionNetwork


def HydrophobicityNetwork(trainX, trainY, valX, valY, physical_H_input, folds, train_time=None):

    if (train_time == 0):
        x = core.Flatten()(physical_H_input)
        x = BatchNormalization()(x)

        x = Dense(1024, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(512, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(256, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(128, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        physical_H_output = Dense(100, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)

        HydrophobicityNetwork = Model(physical_H_input, physical_H_output)

        # optimization = SGD(lr=0.01, momentum=0.9, nesterov= True)
        optimization = 'Nadam'
        HydrophobicityNetwork.compile(loss='categorical_crossentropy', optimizer=optimization)
    else:
        HydrophobicityNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'HydrophobicityNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Hydrophobicityweight.h5',
                                              verbose=1,
                                              save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        loss_checkpointer = LossModelCheckpoint(
                                            model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'HydrophobicityNetwork.h5',
                                            monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Hydrophobicityloss.json',
                                            verbose=1, save_best_only=True, monitor='val_loss', mode='min')

        fitHistory = HydrophobicityNetwork.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2,
                                               validation_data=(valX, valY), shuffle=True, class_weight='auto',
                                               callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return HydrophobicityNetwork


def PhysicochemicalNetwork(trainX, trainY, valX, valY, physical_P_input, folds, train_time=None):

    if (train_time == 0):
        x = core.Flatten()(physical_P_input)
        x = BatchNormalization()(x)

        x = Dense(512, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(256, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(128, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(32, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)

        x = Dense(8, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)

        physical_P_output = Dense(100, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)

        PhysicochemicalNetwork = Model(physical_P_input, physical_P_output)

        optimization = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

        PhysicochemicalNetwork.compile(loss='categorical_crossentropy', optimizer=optimization)
    else:
        PhysicochemicalNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'PhysicochemicalNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Physicochemicalweight.h5',
                                              verbose=1,
                                              save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=200)
        loss_checkpointer = LossModelCheckpoint(
                                            model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'PhysicochemicalNetwork.h5',
                                            monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Physicochemicalloss.json',
                                            verbose=1, save_best_only=True, monitor='val_loss', mode='min')

        fitHistory = PhysicochemicalNetwork.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2,
                                                validation_data=(valX, valY), shuffle=True, class_weight='auto',
                                                callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return PhysicochemicalNetwork


def OtherNetwork(trainX, trainY, valX, valY, physical_O_input, folds, train_time=None):

    if (train_time == 0):
        x = core.Flatten()(physical_O_input)
        x = BatchNormalization()(x)

        x = Dense(256, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(128, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(64, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(32, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)

        physical_O_output = Dense(100, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)

        OtherNetwork = Model(physical_O_input, physical_O_output)

        optimization = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

        OtherNetwork.compile(loss='categorical_crossentropy', optimizer=optimization)
    else:
        OtherNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'OtherNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Otherweight.h5',
                                              verbose=1, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=200)
        loss_checkpointer = LossModelCheckpoint(model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'OtherNetwork.h5',
                                                monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Otherloss.json',
                                                verbose=1, save_best_only=True, monitor='val_loss', mode='min')

        fitHistory = OtherNetwork.fit(trainX, trainY, batch_size=512, epochs=5000, verbose=2,
                                      validation_data=(valX, valY), shuffle=True, class_weight='auto',
                                      callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return OtherNetwork


def mixallCNNmodel(trainX, trainY, valX, valY, physical_C_input, folds, train_time=None):

    if (train_time == 0):

        x = conv.Convolution1D(201, 2, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(physical_C_input)
        x = Dropout(0.4)(x)
        x = Activation('relu')(x)

        x = conv.Convolution1D(151, 3, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = Activation('relu')(x)

        x = conv.Convolution1D(101, 5, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.4)(x)
        x = Activation('relu')(x)

        x = conv.Convolution1D(51, 7, init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x)
        x = Dropout(0.1)(x)
        x = Activation('relu')(x)

        x = core.Flatten()(x)
        x = BatchNormalization()(x)
        physical_C_output = Dense(2, init='glorot_normal', activation='softmax', W_regularizer=l2(0.001))(x)

        mixallmodel = Model(physical_C_input, physical_C_output)

        optimization = 'Nadam'
        mixallmodel.compile(loss='binary_crossentropy', optimizer=optimization, metrics=[keras.metrics.binary_accuracy])
    else:
        mixallmodel = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'CNNNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'CNNweight.h5',
                                              verbose=1, save_best_only=True, monitor='val_loss', mode='min',
                                              save_weights_only=True)
        loss_checkpointer = LossModelCheckpoint(model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'CNNNetwork.h5',
                                                monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'CNNloss.json',
                                                verbose=1, save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=200)
        fitHistory = mixallmodel.fit(trainX, trainY, batch_size=4096, nb_epoch=50, shuffle=True,
                                     callbacks=[early_stopping, loss_checkpointer, weight_checkpointer],
                                     class_weight='auto', validation_data=(valX, valY))
    return mixallmodel


def mixallDNNmodel(trainX, trainY, valX, valY, physical_D_input, folds, train_time=None):

    if (train_time == 0):

        x = core.Flatten()(physical_D_input)
        x = BatchNormalization()(x)

        x = Dense(2048, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(512, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(128, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(64, init='glorot_normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        physical_D_output = Dense(2, init='glorot_normal', activation='softmax', W_regularizer=l2(0.001))(x)

        mixallDNNmodel = Model(physical_D_input, physical_D_output)

        optimization = 'Nadam'
        mixallDNNmodel.compile(loss='binary_crossentropy', optimizer=optimization,
                               metrics=[keras.metrics.binary_accuracy])
    else:
        mixallDNNmodel = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'DNNNetwork.h5')

    if (trainY is not None):

        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'DNNweight.h5',
                                              verbose=1, save_best_only=True,
                                              monitor='val_loss', mode='min', save_weights_only=True)
        loss_checkpointer = LossModelCheckpoint(model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'DNNNetwork.h5',
                                                monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'DNNloss.json',
                                                verbose=1, save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=200)
        fitHistory = mixallDNNmodel.fit(trainX, trainY, batch_size=4096, nb_epoch=50, shuffle=True,
                                        callbacks=[early_stopping, loss_checkpointer, weight_checkpointer],
                                        class_weight='auto', validation_data=(valX, valY))
    return mixallDNNmodel


def mixallRNNmodel(trainX, trainY, valX, valY, physical_R_input, folds, train_time=None):

    if (train_time == 0):
        x = LSTM(1024, init='glorot_normal', activation='relu', recurrent_activation='hard_sigmoid')(physical_R_input)
        # x = LSTM(512,init='glorot_normal',activation='relu',return_sequences=True)(physical_all_input)

        x = Dropout(0.4)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        physical_R_output = Dense(2, init='glorot_normal', activation='softmax', W_regularizer=l2(0.001))(x)

        mixallRNNmodel = Model(physical_R_input, physical_R_output)

        optimization = 'Nadam'
        mixallRNNmodel.compile(loss='binary_crossentropy', optimizer=optimization,
                               metrics=[keras.metrics.binary_accuracy])
    else:
        mixallRNNmodel = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'RNNNetwork.h5')

    if (trainY is not None):
        weight_checkpointer = ModelCheckpoint(filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'RNNweight.h5',
                                              verbose=1, save_best_only=True,
                                              monitor='val_loss', mode='min', save_weights_only=True)
        loss_checkpointer = LossModelCheckpoint(model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'RNNNetwork.h5',
                                                monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'RNNloss.json',
                                                verbose=1, save_best_only=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=200)
        fitHistory = mixallRNNmodel.fit(trainX, trainY, batch_size=4096, nb_epoch=50, shuffle=True,
                                        callbacks=[early_stopping, loss_checkpointer, weight_checkpointer],
                                        class_weight='auto', validation_data=(valX, valY))
    return mixallRNNmodel