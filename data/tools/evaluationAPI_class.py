import re
import math
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import evaluationNetAcet
import evaluationntAcPredictor
import evaluationpall
import evaluationCnnSelfAttention

fig, ax = plt.subplots()


def draw_PRNetACet(label, pre_score):
    if label.count(1) != 0:
        y_label = np.array(label)
        y_pred = np.array(pre_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
        lr_precision = lr_precision[:-1]
        lr_recall = lr_recall[:-1]
        # print('lr_precision', lr_precision)
        # print('lr_recall', lr_recall)
        ax.plot(lr_recall, lr_precision, lw=2, label='NetAcet')
        fontsize = 14
        ax.axis([0.0, 1.0, 0.0, 1.0])
        ax.set_xlabel('Recall', fontsize=fontsize)
        ax.set_ylabel('Precision', fontsize=fontsize)
        ax.set_title('Precision Recall Curve')
        ax.legend()
        # plt.show()

        # print(len(label))
        # print(label)
        # print(len(pre_score))
        # print(pre_score)
    else:
        ax.axis([0.0, 1.0, 0.0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall')
        ax.scatter(0, 0, color='blue', label='NetAcet')
        ax.legend()
        # plt.show()


def draw_PRnt(lirecall, liprecision, Precison, Recall):
    ax.axis([0.0, 1.0, 0.0, 1.0])
    # ax.plot(lirecall, liprecision, 'o', label='each kind')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.scatter(Recall, Precison, color='red', label='nt_AcPredictor')
    ax.legend()
    # plt.show()


def draw_PRpall(Recall, Precison):
    ax.axis([0.0, 1.0, 0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.scatter(Recall, Precison, color='yellow', label='pall')
    ax.legend()
    # plt.show()


def draw_PRcnn(Recall, Precison):
    ax.axis([0.0, 1.0, 0.0, 1.0])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall')
    ax.scatter(Recall, Precison, color='green', label='cnn')
    ax.legend()
    # plt.show()


if __name__ == '__main__':
    # plt.figure()
    kind = "T"

    # NetAcet AGST
    label1, pre = evaluationNetAcet.NetAcet_eva(kind)
    print(label1, pre)
    draw_PRNetACet(label1, pre)

    # nt_AcPredictor ACDGKMPSTV
    lirecall, liprecision, Precison, Recall = evaluationntAcPredictor.nt_AcPredictor_eva(kind)
    draw_PRnt(lirecall, liprecision, Precison, Recall)

    # pall K
    # pallRecall, pallPrecision = evaluationpall.pall_eva()
    # draw_PRpall(pallRecall, pallPrecision)

    # cnn_self_sttention ACDGMPSTV
    cnnRecall, cnnPrecision = evaluationCnnSelfAttention.cnn_self_eva(kind)
    draw_PRcnn(cnnRecall, cnnPrecision)

    plt.savefig("comp\\" + kind + ".png")
    plt.show()
