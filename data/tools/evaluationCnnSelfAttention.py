import math
import re
import numpy as np
from matplotlib import pyplot as plt

pattern = re.compile('([^\s]+)')

# def draw_PR():
#     # print("lirecall",lirecall)
#     # print("liprecision",liprecision)
#     # print("sum:",Precison,Recall)
#     plt.plot(lirecall, liprecision, 'o', label='each kind')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall')
#     plt.scatter(Recall, Precison, color='red', label='all')
#     plt.legend()
#     plt.show()


def get_performence(file):
    res = []
    for line in open(file, "r"):
        # print(line)
        res = line.split()
    precision = float(res[res.index("precision") + 1])
    recall = float(res[res.index("sensitivity") + 1])
    return recall, precision


# if __name__ == '__main__':
#     kind = "A"
def cnn_self_eva(kind):
    file_eva = r"D:\Users\omen\PycharmProjects\N-terminal-Acetylation\data\uniport\result\cnn_self_attention\m_" + kind +".txt"
    recall, precision = get_performence(file_eva)
    return recall, precision

