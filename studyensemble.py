""""""
import os
import sys
import csv
import copy
import math
import random
import pandas as pd
import numpy as np
import keras.metrics
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
# from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc, average_precision_score
from submodel import OnehotNetwork, AlphaturnpropensityNetwork, BetapropensityNetwork, CompositionNetwork, \
    HydrophobicityNetwork, PhysicochemicalNetwork, OtherNetwork, mixallCNNmodel, mixallDNNmodel, mixallRNNmodel
import json
import tensorflow as tf
import dp

gpu_id = '1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)


def calculate_performance(y_length, labels, predict_y, predict_score):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # for index in range(test_num):
    #     if (labels[index] == 1):
    #         if (labels[index] == predict_y[index]):
    #             tp += 1
    #         else:
    #             fn += 1
    #     else:
    #         if (labels[index] == predict_y[index]):
    #             tn += 1
    #         else:
    #             fp += 1

    step = 3
    list_true = [labels[i:i + step] for i in range(0, len(labels), step)]
    list_predict = [predict_y[i:i + step] for i in range(0, len(predict_y), step)]
    '''#严格的计算strict
    for i in range(len(list_true)):
        if ((1 in list_true[i]) and list_true[i] == list_predict[i]):  # 010预测为010
            tp = tp + 1;
        if ((1 in list_true[i]) and (1 not in list_predict[i])):  # 010预测为000
            fn = fn + 1;
        if ((1 in list_true[i]) and (1 in list_predict[i]) and (
                list_true[i] != list_predict[i])):  # 010预测为100,001,110,011,101,111
            fp = fp + 1;
        if ((1 not in list_true[i]) and 1 not in list_predict[i]):  # 000预测为000
            tn = tn + 1;
        if ((1 not in list_true[i]) and 1 in list_predict[i]):  # 000预测为100
            fp = fp + 1;
    '''

    for i in range(len(list_true)):  # 粗略的计算
        if ((1 in list_true[i]) and list_true[i] == list_predict[i]):  # 010预测为010
            tp = tp + 1;
        if ((1 in list_true[i]) and (1 not in list_predict[i])):  # 010预测为000
            fn = fn + 1;
        if ((1 in list_true[i]) and (1 in list_predict[i]) and (
                list_true[i] != list_predict[i])):  # 010预测为100,001,110,011,101,111
            tp = tp + 1;
        if ((1 not in list_true[i]) and 1 not in list_predict[i]):  # 000预测为000
            tn = tn + 1;
        if ((1 not in list_true[i]) and 1 in list_predict[i]):  # 000预测为100
            fp = fp + 1;

    test_num = int(y_length/3)
    # print("y_length/3",y_length/3)
    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp + sys.float_info.epsilon)
    sensitivity = float(tp) / (tp + fn + sys.float_info.epsilon)
    specificity = float(tn) / (tn + fp + sys.float_info.epsilon)
    f1 = 2 * precision * sensitivity / (precision + sensitivity + sys.float_info.epsilon)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)
    # mcc = float(tp*tn-fp*fn)/(np.sqrt(int((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))
    aps = average_precision_score(labels, predict_score)
    fpr, tpr, _ = roc_curve(labels, predict_score)
    aucResults = auc(fpr, tpr)

    strResults = 'tp ' + str(tp) + ' fn ' + str(fn) + ' tn ' + str(tn) + ' fp ' + str(fp)
    strResults = strResults + ' acc ' + str(acc) + ' precision ' + str(precision) + ' sensitivity ' + str(sensitivity)
    strResults = strResults + ' specificity ' + str(specificity) + ' f1 ' + str(f1) + ' mcc ' + str(mcc)
    strResults = strResults + ' aps ' + str(aps) + ' auc ' + str(aucResults)
    return strResults


def normalization(array):
    normal_array = []
    de = array.sum()
    for i in array:
        normal_array.append(float(i) / de)

    return normal_array


def normalization_softmax(array):
    normal_array = []
    de = 0
    for i in array:
        de += math.exp(i)
    for i in array:
        normal_array.append(math.exp(i) / de)

    return normal_array


def Newshufflewrr(data1_pos, data1_neg):
    ##### Create an index with nummber of posnum #####
    index = [i for i in range(len(data1_pos))]
    random.shuffle(index)
    data1_pos = pd.DataFrame(data1_pos)
    data1_pos = data1_pos.as_matrix()[index]
    data1_pos_ss = pd.DataFrame(data1_pos)

    index = [i for i in range(len(data1_neg))]
    random.shuffle(index)
    data1_neg = pd.DataFrame(data1_neg)
    data1_neg = data1_neg.as_matrix()[index]
    data1_neg_ss = pd.DataFrame(data1_neg)

    return data1_pos_ss, data1_neg_ss


def run_model(folds):
    pos_train, neg_train = dp.split_train('N_data/train/' + str(folds) + '/train.fasta')
    # ###########
    # """test"""
    sequence, label = dp.get_data_test('N_data/test/test.fa')
    test_X, test_Y = dp.decode(sequence, label)
    testX, testY, _ = dp.reshape(test_X, test_Y)
    test_phyA_X, test_phyA_Y = dp.phy_decode_A(sequence, label)
    test_phyAX, test_phyAY, _ = dp.reshape(test_phyA_X, test_phyA_Y)
    test_phyB_X, test_phyB_Y = dp.phy_decode_B(sequence, label)
    test_phyBX, test_phyBY, _ = dp.reshape(test_phyB_X, test_phyB_Y)
    test_phyC_X, test_phyC_Y = dp.phy_decode_C(sequence, label)
    test_phyCX, test_phyCY, _ = dp.reshape(test_phyC_X, test_phyC_Y)
    test_phyH_X, test_phyH_Y = dp.phy_decode_H(sequence, label)
    test_phyHX, test_phyHY, _ = dp.reshape(test_phyH_X, test_phyH_Y)
    test_phyO_X, test_phyO_Y = dp.phy_decode_O(sequence, label)
    test_phyOX, tes_phyOtY, _ = dp.reshape(test_phyO_X, test_phyO_Y)
    test_phyP_X, test_phyP_Y = dp.phy_decode_P(sequence, label)
    test_phyPX, test_phyPY, _ = dp.reshape(test_phyP_X, test_phyP_Y)
    del sequence, label
    print("Test data coding finished!")


    """val"""
    sequence2, label2 = dp.get_data_val('N_data/train/' + str(folds) + '/val.fasta')
    val_onehot_X, val_onehot_Y = dp.decode(sequence2, label2)
    val_onehotX, val_onehotY, _ = dp.reshape(val_onehot_X, val_onehot_Y)
    val_phyA_X, val_phyA_Y = dp.phy_decode_A(sequence2, label2)
    val_phyAX, val_phyAY, _ = dp.reshape(val_phyA_X, val_phyA_Y)
    val_phyB_X, val_phyB_Y = dp.phy_decode_B(sequence2, label2)
    val_phyBX, val_phyBY, _ = dp.reshape(val_phyB_X, val_phyB_Y)
    val_phyC_X, val_phyC_Y = dp.phy_decode_C(sequence2, label2)
    val_phyCX, val_phyCY, _ = dp.reshape(val_phyC_X, val_phyC_Y)
    val_phyH_X, val_phyH_Y = dp.phy_decode_H(sequence2, label2)
    val_phyHX, val_phyHY, _ = dp.reshape(val_phyH_X, val_phyH_Y)
    val_phyO_X, val_phyO_Y = dp.phy_decode_O(sequence2, label2)
    val_phyOX, val_phyOY, _ = dp.reshape(val_phyO_X, val_phyO_Y)
    val_phyP_X, val_phyP_Y = dp.phy_decode_P(sequence2, label2)
    val_phyPX, val_phyPY, _ = dp.reshape(val_phyP_X, val_phyP_Y)
    del sequence2, label2
    print("Val data coding finished!")

    # testX, testY = val_onehotX, val_onehotY
    # test_phyAX, test_phyAY = val_phyAX, val_phyAY
    # test_phyBX, test_phyBY = val_phyBX, val_phyBY
    # test_phyCX, test_phyCY = val_phyCX, val_phyCY
    # test_phyHX, test_phyHY = val_phyHX, val_phyHY
    # test_phyOX, tes_phyOtY = val_phyOX, val_phyOY
    # test_phyPX, test_phyPY = val_phyPX, val_phyPY

    iteration_times = 10  # 很多倍
    for t in range(0, iteration_times):
        ############
        print("iteration_times: %d" % t)
        pos_df = pos_train.sample(frac=1, random_state=1)
        neg_df = neg_train.sample(frac=1, random_state=1)
        n_df = neg_df[len(pos_df)*t:(len(pos_df)*(t+1))]
        p_df = pos_df

        df_all = p_df.append(n_df)
        df_all = df_all.sample(frac=1, random_state=1)

        sequence1, label1 = dp.cut_train(df_all, 50)
        train_onehot_X, train_onehot_Y = dp.decode(sequence1, label1)
        train_onehotX, train_onehotY, onehot_input = dp.reshape(train_onehot_X, train_onehot_Y)
        train_phyA_X, train_phyA_Y = dp.phy_decode_A(sequence1, label1)
        train_phyAX, train_phyAY, phyA_input = dp.reshape(train_phyA_X, train_phyA_Y)
        train_phyB_X, train_phyB_Y = dp.phy_decode_B(sequence1, label1)
        train_phyBX, train_phyBY, phyB_input = dp.reshape(train_phyB_X, train_phyB_Y)
        train_phyC_X, train_phyC_Y = dp.phy_decode_C(sequence1, label1)
        train_phyCX, train_phyCY, phyC_input = dp.reshape(train_phyC_X, train_phyC_Y)
        train_phyH_X, train_phyH_Y = dp.phy_decode_H(sequence1, label1)
        train_phyHX, train_phyHY, phyH_input = dp.reshape(train_phyH_X, train_phyH_Y)
        train_phyO_X, train_phyO_Y = dp.phy_decode_O(sequence1, label1)
        train_phyOX, train_phyOY, phyO_input = dp.reshape(train_phyO_X, train_phyO_Y)
        train_phyP_X, train_phyP_Y = dp.phy_decode_P(sequence1, label1)
        train_phyPX, train_phyPY, phyP_input = dp.reshape(train_phyP_X, train_phyP_Y)
        print("itreation %d times Train data coding finished!" % t)

        if (t == 0):
            struct_Onehot_model = OnehotNetwork(train_onehotX, train_onehotY, val_onehotX, val_onehotY, onehot_input, folds, train_time=t)
            physical_O_model = OtherNetwork(train_phyOX, train_phyOY, val_phyOX, val_phyOY, phyO_input, folds, train_time=t)
            physical_P_model = PhysicochemicalNetwork(train_phyPX, train_phyPY, val_phyPX, val_phyPY, phyP_input, folds, train_time=t)
            physical_H_model = HydrophobicityNetwork(train_phyHX, train_phyHY, val_phyHX, val_phyHY, phyH_input, folds, train_time=t)
            physical_C_model = CompositionNetwork(train_phyCX, train_phyCY, val_phyCX, val_phyCY, phyC_input, folds, train_time=t)
            physical_B_model = BetapropensityNetwork(train_phyBX, train_phyBY, val_phyBX, val_phyBY, phyB_input, folds, train_time=t)
            physical_A_model = AlphaturnpropensityNetwork(train_phyAX, train_phyAY, val_phyAX, val_phyAY, phyA_input, folds, train_time=t)
            print("itreation %d times training finished!" % t)
        else:
            struct_Onehot_model = OnehotNetwork(train_onehotX, train_onehotY, val_onehotX, val_onehotY, onehot_input,
                                                folds, train_time=t)
            physical_O_model = OtherNetwork(train_phyOX, train_phyOY, val_phyOX, val_phyOY, phyO_input, folds,
                                            train_time=t)
            physical_P_model = PhysicochemicalNetwork(train_phyPX, train_phyPY, val_phyPX, val_phyPY, phyP_input,
                                                      folds, train_time=t)
            physical_H_model = HydrophobicityNetwork(train_phyHX, train_phyHY, val_phyHX, val_phyHY, phyH_input,
                                                     folds, train_time=t)
            physical_C_model = CompositionNetwork(train_phyCX, train_phyCY, val_phyCX, val_phyCY, phyC_input, folds,
                                                  train_time=t)
            physical_B_model = BetapropensityNetwork(train_phyBX, train_phyBY, val_phyBX, val_phyBY, phyB_input,
                                                     folds, train_time=t)
            physical_A_model = AlphaturnpropensityNetwork(train_phyAX, train_phyAY, val_phyAX, val_phyAY, phyA_input,
                                                          folds, train_time=t)
            print("itreation %d times training finished!" % t)

        # struct_Onehot_model = load_model('model/' + str(folds) + '/model/' + str(t) + 'OnehotNetwork.h5')
        # physical_O_model = load_model('model/' + str(folds) + '/model/' + str(t) + 'OtherNetwork.h5')
        # physical_P_model = load_model('model/' + str(folds) + '/model/' + str(t) + 'PhysicochemicalNetwork.h5')
        # physical_H_model = load_model('model/' + str(folds) + '/model/' + str(t) + 'HydrophobicityNetwork.h5')
        # physical_C_model = load_model('model/' + str(folds) + '/model/' + str(t) + 'CompositionNetwork.h5')
        # physical_B_model = load_model('model/' + str(folds) + '/model/' + str(t) + 'BetapropensityNetwork.h5')
        # physical_A_model = load_model('model/' + str(folds) + '/model/' + str(t) + 'AlphaturnpropensityNetwork.h5')
        # print("itreation %d times training finished!" % t)

        monitor = 'val_loss'
        weights = []
        with open('model/' + str(folds) + '/loss/' + str(t) + 'Onehotloss.json', 'r') as checkpoint_fp:
            weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
        with open('model/' + str(folds) + '/loss/' + str(t) + 'Otherloss.json', 'r') as checkpoint_fp:
            weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
        with open('model/' + str(folds) + '/loss/' + str(t) + 'Physicochemicalloss.json', 'r') as checkpoint_fp:
            weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
        with open('model/' + str(folds) + '/loss/' + str(t) + 'Hydrophobicityloss.json', 'r') as checkpoint_fp:
            weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
        with open('model/' + str(folds) + '/loss/' + str(t) + 'Compositionloss.json', 'r') as checkpoint_fp:
            weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
        with open('model/' + str(folds) + '/loss/' + str(t) + 'Betapropensityloss.json', 'r') as checkpoint_fp:
            weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
        with open('model/' + str(folds) + '/loss/' + str(t) + 'Alphaturnpropensityloss.json', 'r') as checkpoint_fp:
            weights.append(1 / float(json.load(checkpoint_fp)[monitor]))

        weight_array = np.array(weights, dtype=np.float)
        del weights
        print("Loss chick point %d times finished!" % t)

        weight_array = normalization_softmax(weight_array)

        predict_weighted_merge = 0
        predict_temp = weight_array[0] * struct_Onehot_model.predict(testX)
        predict_weighted_merge += predict_temp
        predict_temp = weight_array[1] * physical_O_model.predict(test_phyOX)
        predict_weighted_merge += predict_temp
        predict_temp = weight_array[2] * physical_P_model.predict(test_phyPX)
        predict_weighted_merge += predict_temp
        predict_temp = weight_array[3] * physical_H_model.predict(test_phyHX)
        predict_weighted_merge += predict_temp
        predict_temp = weight_array[4] * physical_C_model.predict(test_phyCX)
        predict_weighted_merge += predict_temp
        predict_temp = weight_array[5] * physical_B_model.predict(test_phyBX)
        predict_weighted_merge += predict_temp
        predict_temp = weight_array[6] * physical_A_model.predict(test_phyAX)
        predict_weighted_merge += predict_temp

        predict_classes = copy.deepcopy(predict_weighted_merge[:, 1])
        for n in range(len(predict_classes)):
            if predict_classes[n] >= 0.5:
                predict_classes[n] = 1
            else:
                predict_classes[n] = 0
        #print("len(testY)",len(testY))
        # print("testY[:, 1]",testY[:, 1])
        # print("type(testY[:, 1])", type(testY[:, 1]))
        #print("predict_classes",predict_classes)
        #print("predict_weighted_merge[:, 1]", predict_weighted_merge[:, 1])
        with open('result/' + str(folds) + '/evaluation.txt', mode='a') as resFile:
            resFile.write(str(t) + " " + calculate_performance(len(testY), testY[:, 1].tolist(), predict_classes.tolist(),
                                                               predict_weighted_merge[:, 1]) + '\r\n')
        resFile.close()
        true_label = testY
        result = np.column_stack((true_label[:, 1], predict_weighted_merge[:, 1]))
        result = pd.DataFrame(result)
        result.to_csv(path_or_buf='result/' + str(folds) + '/result' + '-' + str(t) + '.txt', index=False, header=None, sep='\t',
                      quoting=csv.QUOTE_NONE)


def CNN_model(folds):
    pos_train, neg_train = dp.split_train('N_data/train/' + str(folds) + '/train.fasta')
    # ###########
    # """test"""
    sequence, label = dp.get_data_test('N_data/test/test.fa')
    test_X, test_Y = dp.phy_decode_all(sequence, label)
    testX, testY, _ = dp.reshape(test_X, test_Y)
    del sequence, label

    sequence2, label2 = dp.get_data_val('N_data/train/' + str(folds) + '/val.fasta')
    val_X, val_Y = dp.decode(sequence2, label2)
    valX, valY, _ = dp.reshape(val_X, val_Y)
    del sequence2, label2

    iteration_times = 10
    for t in range(0, iteration_times):
        ############
        print("iteration_times: %d" % t)
        pos_df = pos_train.sample(frac=1, random_state=1)
        neg_df = neg_train.sample(frac=1, random_state=1)
        n_df = neg_df[len(pos_df) * t:(len(pos_df) * (t + 1))]
        p_df = pos_df

        df_all = p_df.append(n_df)
        df_all = df_all.sample(frac=1, random_state=1)

        sequence1, label1 = dp.cut_train(df_all, 50)
        train_X, train_Y = dp.decode(sequence1, label1)
        trainX, trainY, input = dp.reshape(train_X, train_Y)

        if (t == 0):
            physical_all_model = mixallCNNmodel(trainX, trainY, valX, valY, input, folds, train_time=t)
        else:
            physical_all_model = mixallCNNmodel(trainX, trainY, valX, valY, input, folds, train_time=t)

        predict_weighted_merge = physical_all_model.predict(testX)
        predict_classes = copy.deepcopy(predict_weighted_merge[:, 1])
        for n in range(len(predict_classes)):
            if predict_classes[n] >= 0.5:
                predict_classes[n] = 1
            else:
                predict_classes[n] = 0

        with open('result/Cevaluation.txt', mode='a') as resFile:
            resFile.write(str(t) + " " + calculate_performance(len(testY), testY[:, 1], predict_classes,                                                            predict_weighted_merge[:, 1]) + '\r\n')
        resFile.close()
        true_label = testY
        result = np.column_stack((true_label[:, 1], predict_weighted_merge[:, 1]))
        result = pd.DataFrame(result)
        result.to_csv(path_or_buf='result/Cresult' + '-' + str(t) + '.txt', index=False, header=None, sep='\t',
                      quoting=csv.QUOTE_NONE)


def DNN_model(folds):
    pos_train, neg_train = dp.split_train('N_data/train/' + str(folds) + '/train.fasta')
    # ###########
    # """test"""
    sequence, label = dp.get_data_test('N_data/test/test.fa')
    test_X, test_Y = dp.phy_decode_all(sequence, label)
    testX, testY, _ = dp.reshape(test_X, test_Y)
    del sequence, label

    sequence2, label2 = dp.get_data_val('N_data/train/' + str(folds) + '/val.fasta')
    val_X, val_Y = dp.decode(sequence2, label2)
    valX, valY, _ = dp.reshape(val_X, val_Y)
    del sequence2, label2

    iteration_times = 10
    for t in range(0, iteration_times):
        ############
        print("iteration_times: %d" % t)
        pos_df = pos_train.sample(frac=1, random_state=1)
        neg_df = neg_train.sample(frac=1, random_state=1)
        n_df = neg_df[len(pos_df) * t:(len(pos_df) * (t + 1))]
        p_df = pos_df

        df_all = p_df.append(n_df)
        df_all = df_all.sample(frac=1, random_state=1)

        sequence1, label1 = dp.cut_train(df_all, 50)
        train_X, train_Y = dp.decode(sequence1, label1)
        trainX, trainY, input = dp.reshape(train_X, train_Y)

        if (t == 0):
            physical_all_model = mixallCNNmodel(trainX, trainY, valX, valY, input, folds, train_time=t)
        else:
            physical_all_model = mixallCNNmodel(trainX, trainY, valX, valY, input, folds, train_time=t)

        predict_weighted_merge = physical_all_model.predict(testX)
        predict_classes = copy.deepcopy(predict_weighted_merge[:, 1])
        for n in range(len(predict_classes)):
            if predict_classes[n] >= 0.5:
                predict_classes[n] = 1
            else:
                predict_classes[n] = 0

        with open('result/Devaluation.txt', mode='a') as resFile:
            resFile.write(str(t) + " " + calculate_performance(len(testY), testY[:, 1], predict_classes,
                                                               predict_weighted_merge[:, 1]) + '\r\n')
        resFile.close()
        true_label = testY
        result = np.column_stack((true_label[:, 1], predict_weighted_merge[:, 1]))
        result = pd.DataFrame(result)
        result.to_csv(path_or_buf='result/Dresult' + '-' + str(t) + '.txt', index=False, header=None, sep='\t',
                      quoting=csv.QUOTE_NONE)


def RNN_model(folds):
    pos_train, neg_train = dp.split_train('N_data/train/' + str(folds) + '/train.fasta')
    # ###########
    # """test"""
    sequence, label = dp.get_data_test('N_data/test/test.fa')
    test_X, test_Y = dp.phy_decode_all(sequence, label)
    testX, testY, _ = dp.reshape(test_X, test_Y)
    del sequence, label

    sequence2, label2 = dp.get_data_val('N_data/train/' + str(folds) + '/val.fasta')
    val_X, val_Y = dp.decode(sequence2, label2)
    valX, valY, _ = dp.reshape(val_X, val_Y)
    del sequence2, label2

    iteration_times = 10
    for t in range(0, iteration_times):
        ############
        print("iteration_times: %d" % t)
        pos_df = pos_train.sample(frac=1, random_state=1)
        neg_df = neg_train.sample(frac=1, random_state=1)
        n_df = neg_df[len(pos_df) * t:(len(pos_df) * (t + 1))]
        p_df = pos_df

        df_all = p_df.append(n_df)
        df_all = df_all.sample(frac=1, random_state=1)

        sequence1, label1 = dp.cut_train(df_all, 50)
        train_X, train_Y = dp.decode(sequence1, label1)
        trainX, trainY, input = dp.reshape(train_X, train_Y)

        if (t == 0):
            physical_all_model = mixallCNNmodel(trainX, trainY, valX, valY, input, folds, train_time=t)
        else:
            physical_all_model = mixallCNNmodel(trainX, trainY, valX, valY, input, folds, train_time=t)

        predict_weighted_merge = physical_all_model.predict(testX)
        predict_classes = copy.deepcopy(predict_weighted_merge[:, 1])
        for n in range(len(predict_classes)):
            if predict_classes[n] >= 0.5:
                predict_classes[n] = 1
            else:
                predict_classes[n] = 0

        with open('result/Revaluation.txt', mode='a') as resFile:
            resFile.write(str(t) + " " + calculate_performance(len(testY), testY[:, 1], predict_classes,
                                                               predict_weighted_merge[:, 1]) + '\r\n')
        resFile.close()
        true_label = testY
        result = np.column_stack((true_label[:, 1], predict_weighted_merge[:, 1]))
        result = pd.DataFrame(result)
        result.to_csv(path_or_buf='result/Rresult' + '-' + str(t) + '.txt', index=False, header=None, sep='\t',
                      quoting=csv.QUOTE_NONE)


def main():
    for i in range(1, 2):
        run_model(i)


if __name__ == '__main__':
    main()
