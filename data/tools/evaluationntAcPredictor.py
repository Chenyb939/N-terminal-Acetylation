import math
import re
import numpy as np
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

pattern = re.compile('([^\s]+)')
liprecision = []
lirecall = []
tpsum = 0
tnsum = 0
fpsum = 0
fnsum = 0
Precison = 0
Recall = 0
label = []
pre_score = []

def draw_PR():
    # print("lirecall",lirecall)
    # print("liprecision",liprecision)
    # print("sum:",Precison,Recall)
    plt.plot(lirecall, liprecision, 'o', label='each kind')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.scatter(Recall, Precison, color='red', label='all')
    plt.legend()
    plt.show()


def fa_number_count(file):
    # fa file number
    fa_data = []
    fa_num = 0
    for line in open(file, "r"):
        fa_data.append(line)
        fa_num += 1
    fa_num /= 2
    print("fa_num:%d" % fa_num)
    return fa_data, fa_num


def prediction_number_count(file):
    res_data = []
    pre_num = 0
    n_num = 0
    for line in open(file, "r"):
        pre_num += 1  # number of prediction
        s1 = "unAc"
        if s1 in line:
            n_num += 1  # predict negative number
            label.append(0)
            # pre_score.append(0.1)
        else:
            res_data.append(line)
        # if line[27] == kind:
        #     res_data.append(line)   # predict positive number

    pre_num -= 2  # the two head line
    p_num = pre_num - n_num

    return res_data, pre_num, n_num, p_num


def factor_count(res_data, fa_data, kind):
    tp_num = 0
    for list in res_data[4:]:
        # for i in range(len(list)):
        #     print(i,list[i])
        pre_str = pattern.search(list).group()
        # print(pre_str)
        indexnum = list.index(pre_str) + len(pre_str) + 1
        pre_pos = int(list[indexnum:indexnum + 2])
        for i in range(0, len(fa_data), 2):
            name = fa_data[i].split('\t')[0][1:]
            pos = fa_data[i].split('\t')[1]
            if pre_str in name:
                if int(pre_pos) == int(pos):
                    tp_num += 1  # true positive number
                    # score = float(list[41:46])
                    label.append(1)
                    # pre_score.append(score)
                    # print('str:%s' % pre_str)
                    # print('name:%s' % name, score)
                else:
                    # score = float(list[41:46])
                    label.append(0)
                    # pre_score.append(score)
        # else:
        #     score = float(list[41:46])
        #     label.append(0)
        #     pre_score.append(score)
        # print('-----------')

    return tp_num


def performance(pre_num, tp_num, n_num, p_num, file_eva):
    fp_num = pre_num - tp_num
    fn_num = n_num
    tn_num = n_num*2

    # division by 0
    if tn_num == 0:
        tn_num += 1
    if tp_num == 0:
        tp_num += 1
    if fn_num == 0:
        fn_num += 1
    if fp_num == 0:
        fp_num += 1

    global tpsum
    global tnsum
    global fpsum
    global fnsum
    tpsum += tp_num
    tnsum += tn_num
    fpsum += fp_num
    fnsum += fn_num

    Accuracy = (tp_num + tn_num) / (tp_num + tn_num + fp_num + fn_num)
    Error_rate = (fp_num + fn_num) / (tp_num + tn_num + fp_num + fn_num)
    Sensitive = tp_num / (tp_num + fp_num)
    Precison = tp_num / (tp_num + fp_num)
    liprecision.append(Precison)
    Recall = tp_num / (tp_num + fn_num)
    lirecall.append(Recall)
    F1 = (2 * Precison * Recall) / (Precison + Recall)
    # F1 = 2*tp_num/(2*tp_num+fp_num+fn_num)
    # matthews correlation coefficient
    MCC = (tp_num * tn_num - tp_num * fn_num) / math.sqrt(
        (tp_num + fp_num) * (tp_num + fn_num) * (tn_num + fp_num) * (tn_num * fn_num))
    TPR = tp_num / (tp_num + fn_num)  # true positive rate
    SPC = tn_num / (tn_num + fp_num)  # specificity
    PPV = tp_num / (tp_num + fp_num)  # positive prediction value

    print("pre_num:  %d" % pre_num)
    print("p_num:  %d" % p_num)
    print("tp_num:  %d" % tp_num)
    print("fp_num:  %d" % fp_num)
    print("n_num:  %d" % (n_num * 3))
    print("tn_num:  %d" % tn_num)
    print("fn_num:  %d" % n_num)
    print('-------------------------------')
    print("ACC(accuracy):  %f" % Accuracy)
    print("error_rate:  %f" % Error_rate)
    print("sensitive:  %f" % Sensitive)
    print("SPC(specificity):  %f" % SPC)
    print("Precision:  %f" % Precison)
    print("Recall:  %f" % Recall)
    print("F1:  %f" % F1)
    print("MCC:  %f" % MCC)
    print("TPR:  %f" % TPR)
    print("PPV:  %f" % PPV)

    # f = open(file_eva, 'w', encoding='utf-8')
    # f.write("pre_num:  %d\n" % pre_num)
    # f.write("p_num:  %d\n" % p_num)
    # f.write("tp_num:  %d\n" % tp_num)
    # f.write("fp_num:  %d\n" % fp_num)
    # f.write("n_num:  %d\n" % (n_num*3))
    # f.write("tn_num:  %d\n" % tn_num)
    # f.write("fn_num:  %d\n" % n_num)
    # f.write('-------------------------------\n')
    # f.write("ACC(accuracy):  %f\n" % Accuracy)
    # f.write("error_rate:  %f\n" % Error_rate)
    # f.write("sensitive:  %f\n" % Sensitive)
    # f.write("SPC(specificity):  %f" % SPC)
    # f.write("Precision:  %f\n" % Precison)
    # f.write("Recall:  %f\n" % Recall)
    # f.write("F1:  %f\n" % F1)
    # f.write("MCC:  %f\n" % MCC)
    # f.write("TPR:  %f" % TPR)
    # f.write("PPV:  %f" % PPV)
    # f.close()


def data_processing(kind,file_fa, file_res, file_eva):
    print('**********************************************')

    # fa file number
    fa_data, fa_num = fa_number_count(file_fa)

    #prediction file number
    res_data, pre_num, n_num, p_num = prediction_number_count(file_res)

    tp_num = factor_count(res_data, fa_data, kind)
    print('-----------')

    performance(pre_num, tp_num, n_num, p_num, file_eva)


# if __name__ == '__main__':
#     likind = ["A", "C", "D", "G", "K", "M", "P", "S", "T", "V"]
def nt_AcPredictor_eva(likind):
    for kind in likind:
        file_fa = u"D:\\Users\\omen\\PycharmProjects\\N-terminal-Acetylation\\data\\uniport\\test\\fasta\\first3\\" + kind + ".fa"
        file_res = u"D:\\Users\\omen\\PycharmProjects\\N-terminal-Acetylation\\data\\uniport\\result\\nt-AcPredictor\\all\\" + kind + "-exist.txt"
        file_eva = "D:\\Users\\omen\\PycharmProjects\\N-terminal-Acetylation\\data\\uniport\\result\\nt-AcPredictor\\all\\" + kind +"-evaluation.txt"

        data_processing(kind,file_fa, file_res, file_eva)
        # global Precison
        # global Recall
        Precison = tpsum / (tpsum + fpsum)
        Recall = tpsum / (tpsum + fnsum)
    return lirecall, liprecision, Precison, Recall
    # draw_PR()
    # plt.show()

