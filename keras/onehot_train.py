import os
import dp
import math
import pickle
import check_data
import numpy as np
import keras.layers.convolutional as conv
import keras.layers.core as core
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Add, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2
from LossCheckPoint import LossModelCheckpoint
from keras import backend as K
import sys
from sklearn.metrics import matthews_corrcoef, roc_curve, auc, accuracy_score, average_precision_score
import tensorflow as tf

# 网络视野长度， 残差网络，损失函数 Focal Loss
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        y_pred = y_pred[:, :3]

        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def OnehotNetwork(trainX, trainY, valX, valY, Oneofkey_input, folds, train_time=0):
    if (train_time == 0):
        x = Oneofkey_input

        x1 = Conv2D(101, (3, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="same",
                    activation='sigmoid')(x)
        x1 = Dropout(0.4)(x1)  # （？，100，21，101）

        x2 = Add()([x, x1])
        x2 = Conv2D(1, (1, 1), init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x2)
        x2 = Conv2D(101, (5, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="same",
                    activation='sigmoid')(x2)
        x2 = Dropout(0.4)(x2)  # （？，100，21，101）

        x3 = Add()([x, x2])
        x3 = Conv2D(1, (1, 1), init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x3)
        x3 = Conv2D(101, (7, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="same",
                    activation='sigmoid')(x3)
        x3 = Dropout(0.4)(x3)  # （？，100，21，101）

        x4 = Add()([x, x3])
        Oneofkey_output = Conv2D(1, (1, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="valid",
                    activation='sigmoid')(x4)  # （？，100，1，1）

        OnehotNetwork = Model(Oneofkey_input, Oneofkey_output)
        OnehotNetwork.summary()
        optimizer = 'Nadam'
        OnehotNetwork.compile(loss=[binary_focal_loss(alpha=0.1, gamma=2)], metrics=["mae"], optimizer=optimizer)

    else:
        x = Oneofkey_input

        x1 = Conv2D(101, (3, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="same",
                    activation='sigmoid')(x)
        x1 = Dropout(0.4)(x1)

        x2 = Add()([x, x1])
        x2 = Conv2D(1, (1, 1), init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x2)
        x2 = Conv2D(101, (5, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="same",
                    activation='sigmoid')(x2)
        x2 = Dropout(0.4)(x2)

        x3 = Add()([x, x2])
        x3 = Conv2D(1, (1, 1), init='glorot_normal', W_regularizer=l2(0), border_mode="same")(x3)
        x3 = Conv2D(101, (7, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="same",
                    activation='sigmoid')(x3)
        x3 = Dropout(0.4)(x3)

        x4 = Add()([x, x3])
        Oneofkey_output = Conv2D(1, (1, 21), init='glorot_normal', W_regularizer=l2(0), border_mode="valid",
                                 activation='sigmoid')(x4)

        OnehotNetwork = Model(Oneofkey_input, Oneofkey_output)
        OnehotNetwork.summary()
        optimizer = 'Nadam'
        OnehotNetwork.compile(loss=[binary_focal_loss(alpha=0.1, gamma=5)], metrics=["mae"], optimizer=optimizer)

        # OnehotNetwork = load_model('model/' + str(folds) + '/model/' + str(train_time - 1) + 'OnehotNetwork.h5')
        OnehotNetwork.load_weights('./weigths.h5')
        return OnehotNetwork

    weight_checkpointer = ModelCheckpoint(
        filepath='./model/' + str(folds) + '/weight/' + str(train_time) + 'Onehotweight.h5',
        verbose=1, save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    loss_checkpointer = LossModelCheckpoint(
        model_file_path='model/' + str(folds) + '/model/' + str(train_time) + 'OnehotNetwork.h5',
        monitor_file_path='model/' + str(folds) + '/loss/' + str(train_time) + 'Onehotloss.json',
        verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    fitHistory = OnehotNetwork.fit(trainX, trainY, batch_size=16, epochs=2000, verbose=2, validation_data=(valX, valY),
                                   shuffle=True, callbacks=[early_stopping, loss_checkpointer, weight_checkpointer])
    return OnehotNetwork, fitHistory


def calculate_performance(labels, predict_score, num=100):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for label_n in range(len(labels)):
        label = labels[label_n][:num]
        predict_s = predict_score[label_n][:num]
        for index in range(len(label)):
            if label[index] == 1 and predict_s[index] >= 0.345:
                tp += 1
            elif label[index] == 1 and predict_s[index] < 0.345:
                fn += 1
            elif label[index] == 0 and predict_s[index] >= 0.345:
                fp += 1
            else:
                tn += 1
    test_num = len(labels * num)
    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + sys.float_info.epsilon)
    sensitivity = float(tp) / (tp + fn + sys.float_info.epsilon)
    specificity = float(tn) / (tn + fp + sys.float_info.epsilon)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + sys.float_info.epsilon)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)

    labels = labels.reshape(-1)
    predict_score = predict_score.reshape(-1)
    aps = average_precision_score(labels, predict_score)
    fpr, tpr, _ = roc_curve(labels, predict_score)
    aucResults = auc(fpr, tpr)

    strResults = 'tp ' + str(tp) + ' fn ' + str(fn) + ' tn ' + str(tn) + ' fp ' + str(fp)
    strResults = strResults + ' acc ' + str(acc) + ' precision ' + str(precision) + ' sensitivity ' + str(sensitivity)
    strResults = strResults + ' specificity ' + str(specificity) + ' f1 ' + str(f1) + ' mcc ' + str(mcc)
    strResults = strResults + ' aps ' + str(aps) + ' auc ' + str(aucResults)
    print(strResults)
    return strResults


if __name__ == '__main__':
    for folds in range(2):
        sequence2, label2 = check_data.get_data(
            os.path.join('/mnt/PTM/data/uniport/train/fasta/first3/val', str(folds), 'all.fa'))
        val_onehot_X, val_onehot_Y = dp.decode(sequence2, label2)
        val_onehotX, val_onehotY, _ = dp.reshape_n(val_onehot_X, val_onehot_Y)
        sequence1, label1 = check_data.get_data(
            os.path.join('/mnt/PTM/data/uniport/train/fasta/first3/train', str(folds), 'all.fa'))
        train_onehot_X, train_onehot_Y = dp.decode(sequence1, label1)
        train_onehotX, train_onehotY, onehot_input = dp.reshape_n(train_onehot_X, train_onehot_Y)

        sequence, label = check_data.get_data(
            os.path.join('/mnt/PTM/data/uniport/test/fasta/first3/m_test.fa'))
        test_onehot_X, test_onehot_Y = dp.decode(sequence, label)
        test_onehotX, test_onehotY, _ = dp.reshape_n(test_onehot_X, test_onehot_Y)

        # val_phyA_X, val_phyA_Y = dp.phy_decode_A(sequence2, label2)
        # val_phyAX, val_phyAY, _ = dp.reshape_n(val_phyA_X, val_phyA_Y)
        #
        # train_phyA_X, train_phyA_Y = dp.phy_decode_A(sequence1, label1)
        # train_phyAX, train_phyAY, phyA_input = dp.reshape_n(train_phyA_X, train_phyA_Y)
        #
        # test_phyA_X, test_phyA_Y = dp.phy_decode_A(sequence, label)
        # test_phyAX, test_phyAY, _ = dp.reshape_n(test_phyA_X, test_phyA_Y)

        struct_Onehot_model, fitHistory = OnehotNetwork(train_onehotX, train_onehotY, val_onehotX, val_onehotY,
                                                        onehot_input, folds, train_time=0)
        struct_Onehot_model.save_weights('./weigths.h5')

        # struct_Onehot_model = OnehotNetwork(train_onehotX, train_onehotY, val_onehotX, val_onehotY,
        #                                                         onehot_input, folds, train_time=1)

        predict = struct_Onehot_model.predict(test_onehotX)
        predict = predict.reshape(807, 100)
        calculate_performance(test_onehotY, predict, 3)

        # struct_Onehot_model, fitHistory = OnehotNetwork(train_phyAX, train_phyAY, val_phyAX, val_phyAY,
        #                                                 phyA_input, folds, train_time=0)
        # struct_Onehot_model.save_weights('./weigths.h5')
        # predict = struct_Onehot_model.predict(test_phyAX)
        # predict = predict.reshape(1192, 100)
        # calculate_performance(test_phyAY, predict, 100)