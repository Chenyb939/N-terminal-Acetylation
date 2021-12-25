import math
import re
import numpy as np
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

pattern = re.compile('([^\s]+)')


def draw_PR(label, pre_score):
    y_label = np.array(label)
    y_pred = np.array(pre_score)
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
    lr_precision = lr_precision[:-1]
    lr_recall = lr_recall[:-1]
    # print('1111111', lr_precision)
    # print(lr_recall)
    plt.plot(lr_recall, lr_precision, lw=2, label='CNN')
    fontsize = 14
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.show()

    # print(len(label))
    # print(label)
    # print(len(pre_score))
    # print(pre_score)


def get_performence(file):
    label = []
    pre_score = []
    for line in open(file, "r"):
        # print(line)
        # print(line)
        res = line.split()
        pre_score.append(float(res[0]))
        if res[1] == '0.0':
            label.append(0)
        elif res[1] == '1.0':
            label.append(1)
    # print(label, pre_score)
    # precision = float(res[res.index("precision") + 1])
    # recall = float(res[res.index("sensitivity") + 1])
    return label, pre_score


# if __name__ == '__main__':
#     kind = "A"
def cnn_self_eva(kind):
    file_eva = r"D:\Users\omen\PycharmProjects\N-terminal-Acetylation\data\uniport\result\cnn_self_attention\m_" + kind +".txt"
    label, pre_score = get_performence(file_eva)
    # print(pre_score)
    # draw_PR(label, pre_score)
    return label, pre_score

