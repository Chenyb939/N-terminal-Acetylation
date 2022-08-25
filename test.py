import os
import math
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc, average_precision_score,precision_recall_curve
import sys
from torch.nn.functional import binary_cross_entropy_with_logits
from pytorchtools import EarlyStopping
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN, LayerNorm as LN
sys.path.append("..")
import check_data
import dp
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F
import warnings
import numpy as np

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def calculate_performance(labels, predict_score, domain_divide, num):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for label_n in range(len(labels)):
        label = labels[label_n][:num]
        predict_s = predict_score[label_n][:num]
        for index in range(len(label)):
            if label[index] == 1 and predict_s[index] >= domain_divide:
                tp += 1
            elif label[index] == 1 and predict_s[index] < domain_divide:
                fn += 1
            elif label[index] == 0 and predict_s[index] >= domain_divide:
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
    label_1=np.reshape(labels, (labels.shape[0] * labels.shape[1]))
    score_1=np.reshape(predict_score, (predict_score.shape[0] * predict_score.shape[1]))
    precision_value=average_precision_score(label_1,score_1)
    precision, recalls, thresholds=precision_recall_curve(label_1, score_1)
    f1_scores = (2 * precision * recalls) / (precision + recalls)

    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    sensitivity = float(tp) / (tp + fn + sys.float_info.epsilon)
    specificity = float(tn) / (tn + fp + sys.float_info.epsilon)
    # f1 = 2 * precision * sensitivity / (precision + sensitivity + sys.float_info.epsilon)
    mcc = float(tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + sys.float_info.epsilon)

    labels = labels.reshape(-1)
    predict_score = predict_score.reshape(-1)
    aps = average_precision_score(labels, predict_score)
    fpr, tpr, _ = roc_curve(labels, predict_score)
    aucResults = auc(fpr, tpr)

    strResults='thresholds ' + str(thresholds[best_f1_score_index]) +'\tbest_f1_scores ' + str(best_f1_score) +'\tprecision ' + str(precision_value) + '\tauc ' + str(aucResults)+ '\tsensitivity ' + str(sensitivity)+'\tspecificity ' + str(specificity)+'\tmcc ' + str(mcc)+'\taps ' + str(aps) +'\taucResults ' + str(aucResults)

    return strResults

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
        dropout=torch.nn.Dropout(0.5)
        out_attention = torch.bmm(dropout(attention_weights), value)
        return out_attention


class CNNnet(torch.nn.Module):

    def __init__(self):
        super(CNNnet, self).__init__()

        self.rnn = torch.nn.Sequential(
            nn.LSTM(input_size=60, hidden_size=60, num_layers=2),
        )

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


if __name__ == '__main__':
    for divide in ['1','2','3','4','5']:
        checkpoint='checkpoint_ensemble'+divide+'.pt'
        domain=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        root_path=r'example-data'
        domain=iter(domain)
        for folds in ['test.fa']:
            domain_divide=next(domain)
            BATCH_SIZE = 16
            top_seq = 60
            sequence2=[]
            label2=[]

            sequence1, label1 = check_data.get_data(
                os.path.join(root_path, folds))

            #------------------The physical and chemical properties---------------------

            val_onehot_X_phy, val_onehot_Y_phy=dp.phy_decode_aaindex(sequence1,label1,top_seq)
            val_onehot_X_phy, val_onehot_Y_phy = dp.reshape(val_onehot_X_phy, val_onehot_Y_phy)

            #------------------------Primary feature coding-----------------------------
            val_onehot_X, val_onehot_Y = dp.decode(sequence1, label1, top_seq)
            val_onehotX, val_onehotY = dp.reshape(val_onehot_X, val_onehot_Y)


            x1_onehot = torch.from_numpy(val_onehotX).to(torch.float32)
            y1_onehot = torch.from_numpy(val_onehotY).to(torch.float32)

            #-------------------The physical and chemical properties--------------------

            x1_phy = torch.from_numpy(val_onehot_X_phy).to(torch.float32)
            y1_phy = torch.from_numpy(val_onehot_Y_phy).to(torch.float32)

            #------------------cat all feature------------------------------------------
            x1 = torch.cat((torch.tensor(x1_onehot), torch.tensor(x1_phy)), dim=3)

            x1=x1[:,:,:60,:]
            #--------------Tensor type from original encoding---------------------------
            val_data = Data.TensorDataset(x1, y1_onehot)

            val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)

            cnn = CNNnet().to(device)
            print(cnn)

            cnn.load_state_dict(torch.load(checkpoint))

            all_pred = []
            all_val = []
            for val_step, (val_x, val_y) in enumerate(val_loader):
                cnn.eval()
                val_output,emb_150 = cnn(val_x.to(device))
                val_output = val_output[:, :top_seq]
                val_output=F.sigmoid(val_output)
                all_pred.extend(val_output.squeeze().cpu().detach().numpy().tolist())

                all_val.extend(val_y.cpu().detach().numpy().tolist())

            strResults = calculate_performance(np.array(all_val), np.array(all_pred),domain_divide, top_seq)
            print(strResults)
