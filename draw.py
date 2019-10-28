import os
import dp
import sys
import json
import copy
import math
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
plt.switch_backend('agg')

gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.Session(config=tf_config)


def normalization_softmax(array):
    normal_array = []
    de = 0
    for i in array:
        de += math.exp(i)
    for i in array:
        normal_array.append(math.exp(i) / de)

    return normal_array


def calculate_performance(labels, predict_y, predict_score):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    step = 3
    list_true = [labels[i:i + step] for i in range(0, len(labels), step)]
    list_predict = [predict_y[i:i + step] for i in range(0, len(predict_y), step)]
    for i in range(len(list_true)):
        if (1 in list_true[i]) and list_true[i] == list_predict[i]:
            tp = tp + 1
        if (1 in list_true[i]) and (1 not in list_predict[i]):
            fn = fn + 1
        if (1 in list_true[i]) and (1 in list_predict[i]) and (list_true[i] != list_predict[i]):
            tp = tp + 1
        if (1 not in list_true[i]) and 1 not in list_predict[i]:
            tn = tn + 1
        if (1 not in list_true[i]) and 1 in list_predict[i]:
            fp = fp + 1
    precision = float(tp) / (tp + fp + sys.float_info.epsilon)
    recall = float(tp) / (tp + fn + sys.float_info.epsilon)
    fpr, tpr, _ = roc_curve(labels, predict_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, precision, recall


def fig1():
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

    struct_Onehot_model = load_model('model/8/model/5OnehotNetwork.h5')
    physical_O_model = load_model('model/8/model/5OtherNetwork.h5')
    physical_P_model = load_model('model/8/model/5PhysicochemicalNetwork.h5')
    physical_H_model = load_model('model/8/model/5HydrophobicityNetwork.h5')
    physical_C_model = load_model('model/8/model/5CompositionNetwork.h5')
    physical_B_model = load_model('model/8/model/5BetapropensityNetwork.h5')
    physical_A_model = load_model('model/5/model/5AlphaturnpropensityNetwork.h5')

    true_class = testY[:, 1]
    # onehot
    pred_proba = struct_Onehot_model.predict(testX, batch_size=2048)
    pred_score = pred_proba[:, 1]
    precision, recall, _ = precision_recall_curve(true_class, pred_score)
    average_precision = average_precision_score(true_class, pred_score)
    fpr, tpr, _ = roc_curve(true_class, pred_score)
    roc_auc = auc(fpr, tpr)
    # A
    pred_probaA = physical_A_model.predict(test_phyAX, batch_size=2048)
    pred_scoreA = pred_probaA[:, 1]
    precisionA, recallA, _ = precision_recall_curve(true_class, pred_scoreA)
    average_precisionA = average_precision_score(true_class, pred_scoreA)
    fprA, tprA, _ = roc_curve(true_class, pred_scoreA)
    roc_aucA = auc(fprA, tprA)
    # B
    pred_probaB = physical_B_model.predict(test_phyBX, batch_size=2048)
    pred_scoreB = pred_probaB[:, 1]
    precisionB, recallB, _ = precision_recall_curve(true_class, pred_scoreB)
    average_precisionB = average_precision_score(true_class, pred_scoreB)
    fprB, tprB, _ = roc_curve(true_class, pred_scoreB)
    roc_aucB = auc(fprB, tprB)
    # C
    pred_probaC = physical_C_model.predict(test_phyCX, batch_size=2048)
    pred_scoreC = pred_probaC[:, 1]
    precisionC, recallC, _ = precision_recall_curve(true_class, pred_scoreC)
    average_precisionC = average_precision_score(true_class, pred_scoreC)
    fprC, tprC, _ = roc_curve(true_class, pred_scoreC)
    roc_aucC = auc(fprC, tprC)
    # H
    pred_probaH = physical_H_model.predict(test_phyHX, batch_size=2048)
    pred_scoreH = pred_probaH[:, 1]
    precisionH, recallH, _ = precision_recall_curve(true_class, pred_scoreH)
    average_precisionH = average_precision_score(true_class, pred_scoreH)
    fprH, tprH, _ = roc_curve(true_class, pred_scoreH)
    roc_aucH = auc(fprH, tprH)
    # P
    pred_probaP = physical_P_model.predict(test_phyPX, batch_size=2048)
    pred_scoreP = pred_probaP[:, 1]
    precisionP, recallP, _ = precision_recall_curve(true_class, pred_scoreP)
    average_precisionP = average_precision_score(true_class, pred_scoreP)
    fprP, tprP, _ = roc_curve(true_class, pred_scoreP)
    roc_aucP = auc(fprP, tprP)
    # O
    pred_probaO = physical_O_model.predict(test_phyOX, batch_size=2048)
    pred_scoreO = pred_probaO[:, 1]
    precisionO, recallO, _ = precision_recall_curve(true_class, pred_scoreO)
    average_precisionO = average_precision_score(true_class, pred_scoreO)
    fprO, tprO, _ = roc_curve(true_class, pred_scoreO)
    roc_aucO = auc(fprO, tprO)
    # EN
    monitor = 'val_loss'
    weights = []
    with open('model/8/loss/5Onehotloss.json', 'r') as checkpoint_fp:
        weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
    with open('model/8/loss/5Otherloss.json', 'r') as checkpoint_fp:
        weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
    with open('model/8/loss/5Physicochemicalloss.json', 'r') as checkpoint_fp:
        weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
    with open('model/8/loss/5Hydrophobicityloss.json', 'r') as checkpoint_fp:
        weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
    with open('model/8/loss/5Compositionloss.json', 'r') as checkpoint_fp:
        weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
    with open('model/8/loss/5Betapropensityloss.json', 'r') as checkpoint_fp:
        weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
    with open('model/8/loss/5Alphaturnpropensityloss.json', 'r') as checkpoint_fp:
        weights.append(1 / float(json.load(checkpoint_fp)[monitor]))
    weight_array = np.array(weights, dtype=np.float)

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

    fprE, tprE, roc_aucE, precisionE, recallE = calculate_performance(testY[:, 1].tolist(), predict_classes.tolist(), predict_weighted_merge[:, 1])

    #  ################# Print PR####################
    plt.figure()
    plt.plot(fpr, tpr, color='#0000FF', lw=2, linestyle='-',
             label='One hot net(AUC=%0.2f)' % roc_auc)
    plt.plot(fprA, tprA, color='#00BFFF', lw=2, linestyle='-',
             label='alpha propensity net(AUC=%0.2f)' % roc_aucA)
    plt.plot(fprB, tprB, color='#00FFFF', lw=2, linestyle='-',
             label='beta propensity net(AUC=%0.2f)' % roc_aucB)
    plt.plot(fprC, tprC, color='#00FF00', lw=2, linestyle='-',
             label='Composition net(AUC=%0.2f)' % roc_aucC)
    plt.plot(fprH, tprH, color='#6B8E23', lw=2, linestyle='-',
             label='Hydrophobicity net(AUC=%0.2f)' % roc_aucH)
    plt.plot(fprP, tprP, color='#B8860B', lw=2, linestyle='-',
             label='Phy-chemi properties net(AUC=%0.2f)' % roc_aucP)
    plt.plot(fprO, tprO, color='#FFA500', lw=2, linestyle='-',
             label='Other properties net(AUC=%0.2f)' % roc_aucO)
    plt.plot(fprE, tprE, color='#FF0000', lw=2, linestyle='-',
             label='Ensemble net(AUC=%0.2f)' % roc_aucE)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='-.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig('./ROC.png')
    print('aa')

    # #  ################# Print ROC####################
    # plt.figure()
    # plt.step(recall, precision, color='#0000FF', lw=2, linestyle='-',
    #          label='One hot net(PRC=%0.2f)' % average_precision)
    # plt.step(recallA, precisionA, color='#00BFFF', lw=2, linestyle='-',
    #          label='αpropensity net(PRC=%0.2f)' % average_precisionA)
    # plt.step(recallB, precisionB, color='#00FFFF', lw=2, linestyle='-',
    #          label='β propensity net(PRC=%0.2f)' % average_precisionB)
    # plt.step(recallC, precisionC, color='#00FF00', lw=2, linestyle='-',
    #          label='Composition net(PRC=%0.2f)' % average_precisionC)
    # plt.step(recallH, precisionH, color='#6B8E23', lw=2, linestyle='-',
    #          label='Hydrophobicity net(PRC=%0.2f)' % average_precisionH)
    # plt.step(recallP, precisionP, color='#B8860B', lw=2, linestyle='-',
    #          label='Physico-chemical properties net(PRC=%0.2f)' % average_precisionP)
    # plt.step(recallO, precisionO, color='#FFA500', lw=2, linestyle='-',
    #          label='Other properties net(PRC=%0.2f)' % average_precisionO)
    # plt.step(recallE, precisionE, color='#FF0000', lw=2, linestyle='-',
    #          label='Ensemble net(PRC=%0.2f)' % average_precisionE)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # # plt.grid(True)
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc="lower left")
    # plt.savefig('./PC.png')


if __name__ == '__main__':
    fig1()
