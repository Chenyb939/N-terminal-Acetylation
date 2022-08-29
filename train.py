import os
import math
from torch import nn
import numpy as np

import torch
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc, average_precision_score
import sys
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN
sys.path.append("")
from torch.optim.lr_scheduler import ReduceLROnPlateau
import check_data
import torch.nn.functional as F
import dp
from sklearn.decomposition import PCA
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
# pip install torch-sparse -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
# pip install torch-cluster -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
# pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
# pip install torch-geometric

import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def calculate_performance(labels, predict_score, epoch, num=30):
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
    precision=average_precision_score(np.reshape(labels, (labels.shape[0] * labels.shape[1])),np.reshape(predict_score, (predict_score.shape[0] * predict_score.shape[1])))
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

    return tp,fp,precision,strResults, f1, loss, acc,mcc


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), LN(channels[i]))
        for i in range(1, len(channels))
    ])


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def get_heat_map(grid):
    sns.set()
    for i in range(len(grid)):
        sns.heatmap(data=grid[i].detach().cpu().numpy(), square=True, vmax=1, vmin=0, robust=True)
        plt.title("test")
        plt.savefig("./picture/grid"+str(i)+".png")
    flag = 'train'
    # plt.show()


class Self_attention_cnn(nn.Module):
    def __init__(self):
        super(Self_attention_cnn, self).__init__()
        # pass
    def forward(self,layer4):
        query = layer4
        key = layer4
        value = layer4
        d = query.shape[-1]
        # 设置 `transpose_b=True` 为了交换 `keys` 的最后两个维度
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        # scores = calculate_distance_for(queries,keys)
        attention_weights = torch.softmax(scores, dim=2)
        # if flag=='val':
        #     # get_heat_map(self.attention_weights)
        #     print()
        dropout = torch.nn.Dropout(0.5)
        out_attention = torch.bmm(dropout(attention_weights), value)
        return out_attention


class CNNnet(torch.nn.Module):

    def __init__(self):
        super(CNNnet, self).__init__()

        self.rnn = torch.nn.Sequential(
            nn.LSTM(input_size=60, hidden_size=60, num_layers=2),
        )
        # self.relu = torch.nn.ReLU()

        # self.ln = torch.nn.Sequential(
        #     torch.nn.LayerNorm([100]),torch.nn.Dropout(0.7)
        # )
        # self.ln_phy = torch.nn.LayerNorm([531*2])

        # self.rnn_out = nn.Linear(16, 100)  # 最后时刻的hidden映射

        self.phy_conv1 = torch.nn.Sequential(  # (100, 21)
            torch.nn.Conv2d(in_channels=1,
                            out_channels=10,
                            kernel_size=(3, 1),
                            stride=(1, 1),
                            padding=(1, 0),
                            ),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )

        self.dropout=torch.nn.Sequential(
            torch.nn.Dropout(0.5)
        )

        self.phy_conv2 = torch.nn.Sequential(  # (101, 100, 21)
            torch.nn.Conv2d(10, 10, (5, 1),stride=(1,1), padding=(2, 0)),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )

        self.phy_conv3 = torch.nn.Sequential(  # (101, 100, 1)
            # torch.nn.Conv2d(101, 101, (7, 5), padding=(3, 2)),
            torch.nn.Conv2d(10, 10, (7, 3), stride=(1,2),padding=(3, 0)),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )

        self.phy_conv4 = torch.nn.Sequential(  # (101, 100, 1)
            # torch.nn.Conv2d(101, 101, (7, 5), padding=(3, 2)),
            torch.nn.Conv2d(10, 10, (9, 3), stride=(1,2),padding=(4, 0)),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )

        self.attention_cnn = Self_attention_cnn()

        self.mlp_phy = torch.nn.Sequential(
            torch.nn.Linear(1320, 20),
        )

        self.neck=torch.nn.Sequential(
            torch.nn.Linear(40, 60),torch.nn.BatchNorm1d(60),torch.nn.ReLU(),torch.nn.Dropout(0.3)
        )

        self.mlp = torch.nn.Sequential(
            Lin(60, 30), BN(60), ReLU(), Dropout(0.3), Lin(30, 1)
        )

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(60, 1)
        # )

    def forward(self, x):
        layer1=x[:, :, :, :20]
        if layer1.shape[0] == 1:
            layer1 = torch.squeeze(layer1, 1)
            layer1 = layer1.transpose(1, 2)
        else:
            layer1 = layer1.squeeze().transpose(1, 2)
        # print(x.shape)
        r_out, (h_n, h_c) = self.rnn(layer1)
        # r_out = self.ln(r_out)
        # r_out = self.relu(r_out)
        r_out=r_out.permute(0, 2, 1)

        phy_layer1 = self.phy_conv1(x[:, :, :, -531:])
        # print("conv1d:", x.size(), "➡️", layer1.size())
        phy_layer2 = self.phy_conv2(phy_layer1)
        # print("conv2d:", layer1.size(), "➡️", layer2.size())
        phy_layer3 = self.phy_conv3(phy_layer2)
        # print("conv3d:", layer2.size(), "➡️", layer3.size())
        phy_layer4 = self.phy_conv4(phy_layer3)
        # phy_layer4 = phy_layer3
        # print(layer2.shape)
        phy_layer4=phy_layer4.permute(0, 2, 1, 3)

        phy_layer4=torch.reshape(phy_layer4, (len(phy_layer4), len(phy_layer4[0]), len(phy_layer4[0][0]) * len(phy_layer4[0][0][0])))

        phy_layer4=self.mlp_phy(phy_layer4)

        # phy_layer4 = self.relu(phy_layer4)

        phy_layer4=self.dropout(phy_layer4)
        r_out = self.dropout(r_out)

        out_emb=torch.cat([phy_layer4, r_out], dim=2)
        # out_emb_526=self.ln_phy(out_emb_526)

        out_emb = self.attention_cnn(out_emb)

        out_emb=self.neck(out_emb)

        out=self.mlp(out_emb)

        return out.squeeze(),out_emb




def del_topseq_label(seq,label,top_seq):
    temp_seq=[]
    temp_label=[]
    for i in range(len(label)):
        i_list=label[i].split(',')
        i_strr=''
        for j in i_list:
            if j!='' and int(j)<top_seq:
                i_strr=i_strr+j+','
        if i_strr!='':
            temp_seq.append(seq[i])
            temp_label.append(i_strr)

    return temp_seq,temp_label

class FocalLoss(nn.Module):
    def __init__(self, alpha=.0333333333333333333333, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        pos_weight = torch.FloatTensor([1.0]).to(device)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


patiences=[20,20,20,20,20]
# patiences=[120,120,120,120,120]
patiences=iter(patiences)

if __name__ == '__main__':
    for folds in ['1', '2', '3', '4', '5']:
        root_path=r'5-fold-data'
        focalloss=FocalLoss()
        EPOCH = 500
        BATCH_SIZE = 16
        LR = 0.001
        top_seq = 30
        # top_seq_loss = 30
        patience = next(patiences)
        sequence2 = []
        label2 = []
        divide = folds
        checkpoint='./model/checkpoint_ensemble'+divide+'.pt'
        divide_list = ['1', '2', '3', '4', '5']
        divide_list.remove(divide)
        for i in divide_list:
            sequence_temp, label_temp = check_data.get_data(os.path.join(root_path, i + '.fa'))
            sequence2.extend(sequence_temp)
            label2.extend(label_temp)

        sequence1, label1 = check_data.get_data(
            os.path.join(root_path, divide + '.fa'))

        sequence2, label2 = del_topseq_label(sequence2, label2, top_seq)
        sequence1, label1 =del_topseq_label(sequence1,label1,top_seq)

        # ------------------The physical and chemical properties----------------
        train_phy_X, train_phy_Y = dp.phy_decode_aaindex(sequence2, label2, top_seq)
        # train_phy_X=train_phy_X[:, :, :, :24]
        train_phyX, train_phyY = dp.reshape(train_phy_X, train_phy_Y)
        val_phy_X, val_phy_Y = dp.phy_decode_aaindex(sequence1, label1, top_seq)
        # val_phy_X=val_phy_X[:, :, :,: 24]
        val_phyX, val_phyY = dp.reshape(val_phy_X, val_phy_Y)

        # ------------------------Primary feature coding-----------------------------
        val_onehot_X, val_onehot_Y = dp.decode(sequence1, label1, top_seq)
        val_onehotX, val_onehotY = dp.reshape(val_onehot_X, val_onehot_Y)
        train_onehot_X, train_onehot_Y = dp.decode(sequence2, label2, top_seq)
        train_onehotX, train_onehotY = dp.reshape(train_onehot_X, train_onehot_Y)

        del sequence1, label1, sequence2, label2, train_phy_X, train_phy_Y, val_phy_X, val_phy_Y, val_onehot_X, val_onehot_Y, train_onehot_X, train_onehot_Y

        onehot_x = torch.from_numpy(train_onehotX).to(torch.float32)
        onehot_y = torch.from_numpy(train_onehotY).to(torch.float32)
        onehot_x1 = torch.from_numpy(val_onehotX).to(torch.float32)
        onehot_y1 = torch.from_numpy(val_onehotY).to(torch.float32)

        phy_x = torch.from_numpy(train_phyX).to(torch.float32)
        phy_y = torch.from_numpy(train_phyY).to(torch.float32)
        phy_x1 = torch.from_numpy(val_phyX).to(torch.float32)
        phy_y1 = torch.from_numpy(val_phyY).to(torch.float32)

        x = torch.cat([onehot_x, phy_x], dim=3)
        y = onehot_y
        x1 = torch.cat([onehot_x1, phy_x1], dim=3)
        y1 = onehot_y1
        x2 = x1
        y2 = y1
        del onehot_x, phy_x, onehot_y, phy_y, onehot_x1, phy_x1, onehot_y1, phy_y1

        x=x[:,:,:60,:]
        x1=x1[:,:,:60,:]
        x2=x2[:,:,:60,:]

        train_data = Data.TensorDataset(x, y)
        val_data = Data.TensorDataset(x1, y1)
        test_data = Data.TensorDataset(x2, y2)
        del x, y, x1, y1, x2, y2

        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
        del train_data, val_data, test_data
        print('Load data finish.')

        cnn = CNNnet()
        print(cnn)
        cnn.to(device)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)  # , weight_decay=0.0001
        # optimizer=torch.optim.SGD(filter(lambda p: p.requires_grad, change_model_dict.parameters()), lr=0.01, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)

        early_stopping = EarlyStopping(patience, verbose=True,path=checkpoint)
        val_losses=[]

        for epoch in range(EPOCH):
            val_pre_list = []
            val_lab_list = []
            for step, (b_x, b_y) in enumerate(train_loader):
                cnn.train()
                output,embedding = cnn(b_x.to(device))
                new_output = output[:, :top_seq]
                loss=focalloss(new_output, b_y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            all_pred = []
            all_val = []
            for val_step, (val_x, val_y) in enumerate(val_loader):
                cnn.eval()
                val_output,embedding = cnn(val_x.to(device))
                val_output = val_output[:, :top_seq]
                loss = focalloss(val_output, val_y.to(device))
                val_losses.append(loss.item())
                val_output = torch.sigmoid(val_output)
                all_pred.extend(val_output.cpu().detach().numpy().tolist())
                all_val.extend(val_y.cpu().detach().numpy().tolist())
            del val_x, val_y
            print(all_pred[0])
            print(all_val[0])

            val_loss=np.mean(val_losses)

            tp,fp,precision,strResults, f1, loss, acc,mcc = calculate_performance(np.array(all_val), np.array(all_pred), epoch, top_seq)
            early_stopping(val_loss, cnn)

            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Epoch: ', epoch, '| val loss: %.8f' % loss, '| val accuracy: %.2f' % acc)
            print(strResults)
