import os
import math

import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc, average_precision_score
import sys
from torch.nn.functional import binary_cross_entropy_with_logits
from pytorchtools import EarlyStopping

sys.path.append("..")
import check_data
import dp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def calculate_performance(labels, predict_score, epoch, num=3):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for label_n in range(len(labels)):
        label = labels[label_n][:num]
        predict_s = predict_score[label_n][:num]
        for index in range(len(label)):
            if label[index] == 1 and predict_s[index] >= 0.5:
                tp += 1
            elif label[index] == 1 and predict_s[index] < 0.5:
                fn += 1
            elif label[index] == 0 and predict_s[index] >= 0.5:
                fp += 1
            else:
                tn += 1
    if tp == 0:
        tp += 1
    if fp == 0:
        fp += 1
    if tn == 0:
        tn += 1
    if fn == 0:
        fn += 1
    test_num = len(labels) * num
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

    strResults = 'epoch ' + str(epoch) + '\ttp ' + str(tp) + ' \tfn ' + str(fn) + ' \ttn ' + str(tn) + ' \tfp ' + str(
        fp)
    strResults = strResults + '\tacc ' + str(acc) + '\tprecision ' + str(precision) + '\tsensitivity ' + str(
        sensitivity)
    strResults = strResults + '\tspecificity ' + str(specificity) + '\tf1 ' + str(f1) + '\tmcc ' + str(mcc)
    strResults = strResults + '\taps ' + str(aps) + '\tauc ' + str(aucResults) + '\n'

    return strResults, f1, loss, acc


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(  # (100, 21)
            torch.nn.Conv2d(in_channels=1,
                            out_channels=101,
                            kernel_size=(3, 1),
                            padding=(1, 0),
                            ),
            torch.nn.BatchNorm2d(101),
            torch.nn.ReLU()
        )

        self.conv2 = torch.nn.Sequential(  # (101, 100, 21)
            torch.nn.Conv2d(101, 101, (5, 1), padding=(2, 0)),
            torch.nn.BatchNorm2d(101),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(  # (101, 100, 21)
            torch.nn.Conv2d(101, 101, (7, 1), padding=(3, 0)),
            torch.nn.BatchNorm2d(101),
            torch.nn.ReLU()
        )

        self.conv4 = torch.nn.Sequential(  # (101, 100, 1)
            torch.nn.Conv2d(101, 101, (1, 21), padding=(0, 0)),
            torch.nn.BatchNorm2d(101),
            torch.nn.ReLU()
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(101, 1, (1, 1), padding=(0, 0)),
            torch.nn.Sigmoid()
        )

        self.mlp = torch.nn.Linear(100 * 101, 100)

    def forward(self, x):
        layer1 = self.conv1(x)
        print("conv1d:", x.size(), "➡️", layer1.size())
        layer2 = self.conv2(layer1)
        # print("conv2d:", layer1.size(), "➡️", layer2.size())
        layer3 = self.conv3(layer2)
        # print("conv3d:", layer2.size(), "➡️", layer3.size())
        layer4 = self.conv4(layer3)
        # print("conv4d:", layer3.size(), "➡️", layer4.size())
        # layer5 = self.conv5(layer4)
        # print("conv4d:", layer4.size(), "➡️", layer5.size())
        # out = layer5.view(layer5.size(0), -1)
        x = layer4.view(layer4.size(0), -1)
        out = self.mlp(x)
        return out


def load_checkpoint(model, checkpoint, optimizer, loadOptimizer):
    if checkpoint != 'No':
        print("loading checkpoint...")
        model_dict = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
        # 如果不需要更新优化器那么设置为false
        if loadOptimizer == True:
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print('loaded! optimizer')
        else:
            print('not loaded optimizer')
    else:
        print('No checkpoint is included')
    return model, optimizer


for folds in range(5):

    # Hyper Parameters
    EPOCH = 500
    BATCH_SIZE = 16
    LR = 0.001
    top_seq = 100

    sequence2, label2 = check_data.get_data(
        os.path.join('D:/迅雷下载/PTM/data/uniport/train/fasta/first' + str(top_seq) + '/val', str(folds), 'all.fa'))
    val_onehot_X, val_onehot_Y = dp.decode(sequence2, label2, top_seq)
    val_onehotX, val_onehotY = dp.reshape(val_onehot_X, val_onehot_Y)
    sequence1, label1 = check_data.get_data(
        os.path.join('D:/迅雷下载/PTM/data/uniport/train/fasta/first' + str(top_seq) + '/train', str(folds), 'all.fa'))
    train_onehot_X, train_onehot_Y = dp.decode(sequence1, label1, top_seq)
    train_onehotX, train_onehotY = dp.reshape(train_onehot_X, train_onehot_Y)
    testX, testY = val_onehotX, val_onehotY
    print(len(train_onehotX))
    del sequence1, label1, sequence2, label2, train_onehot_X, train_onehot_Y, val_onehot_X, val_onehot_Y

    x = torch.from_numpy(train_onehotX).to(torch.float32)
    y = torch.from_numpy(train_onehotY).to(torch.float32)
    x1 = torch.from_numpy(val_onehotX).to(torch.float32)
    y1 = torch.from_numpy(val_onehotY).to(torch.float32)
    x2 = torch.from_numpy(testX).to(torch.float32)
    y2 = torch.from_numpy(testY).to(torch.float32)
    del train_onehotX, train_onehotY, val_onehotX, val_onehotY, testX, testY

    train_data = Data.TensorDataset(x, y)
    val_data = Data.TensorDataset(x1, y1)
    test_data = Data.TensorDataset(x2, y2)
    del x, y, x1, y1, x2, y2

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
    del train_data, val_data, test_data
    print('Load data finish.')

    cnn = CNNnet()
    print(cnn)
    cnn.to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    patience = 10
    early_stopping = EarlyStopping(patience, verbose=True)

    for epoch in range(EPOCH):
        val_pre_list = []
        val_lab_list = []
        for step, (b_x, b_y) in enumerate(train_loader):
            cnn.train()
            output = cnn(b_x.to(device))
            new_output = output[:, :top_seq]
            loss = binary_cross_entropy_with_logits(new_output, b_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(str.format('step:{:d}, loss:{:.3f}', step, loss))
        del b_x, b_y

        all_pred = []
        all_val = []
        for val_step, (val_x, val_y) in enumerate(val_loader):
            cnn.eval()
            val_output = cnn(val_x.to(device))
            val_output = val_output[:, :top_seq]
            all_pred.extend(val_output.cpu().detach().numpy().tolist())
            all_val.extend(val_y.cpu().detach().numpy().tolist())
        del val_x, val_y

        strResults, f1, loss, acc = calculate_performance(np.array(all_val), np.array(all_pred), epoch, top_seq)
        early_stopping(f1, cnn)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Epoch: ', epoch, '| val loss: %.8f' % loss, '| val accuracy: %.2f' % acc)
        print(strResults)

        if os.path.exists('result.txt') == True:
            with open('result.txt', 'a') as file:
                file.write(strResults)
        else:
            with open('result.txt', 'w') as file:
                file.write(strResults)
