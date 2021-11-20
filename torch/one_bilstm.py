import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_curve, auc, average_precision_score
from torch.nn.functional import binary_cross_entropy_with_logits, softmax
from torch.autograd import Variable
from pytorchtools import EarlyStopping

sys.path.append("..")
import check_data
import dp

torch.cuda.get_device_name(0)
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


class BLSTM(torch.nn.Module):
    def __init__(self, num_embeddings,embedding_dim, input_dim, hidden_dim, num_layers, output_dim, max_len=40, dropout=0.5):
        super(BLSTM, self).__init__()

        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_dim,
                                padding_idx=0)
        # self.emb.weight = nn.Parameter(embeddings)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # sen encoder
        self.sen_len = max_len
        self.sen_rnn = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               dropout=dropout,
                               batch_first=True,
                               bidirectional=True)

        self.output = nn.Linear(2 * self.hidden_dim, output_dim)

    def bi_fetch(self, rnn_outs, seq_lengths, batch_size, max_len):
        rnn_outs = rnn_outs.view(batch_size, max_len, 2, -1)

        # (batch_size, max_len, 1, -1)
        fw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([0])).cuda())
        fw_out = fw_out.view(batch_size * max_len, -1)
        bw_out = torch.index_select(rnn_outs, 2, Variable(torch.LongTensor([1])).cuda())
        bw_out = bw_out.view(batch_size * max_len, -1)

        batch_range = Variable(torch.LongTensor(range(batch_size))).cuda() * max_len
        batch_zeros = Variable(torch.zeros(batch_size).long()).cuda()

        fw_index = batch_range + seq_lengths.view(batch_size) - 1
        fw_out = torch.index_select(fw_out, 0, fw_index)  # (batch_size, hid)

        bw_index = batch_range + batch_zeros
        bw_out = torch.index_select(bw_out, 0, bw_index)

        outs = torch.cat([fw_out, bw_out], dim=1)
        return outs

    def forward(self, sen_batch, sen_lengths=100):
        sen_batch = self.emb(sen_batch)

        batch_size = len(sen_batch)

        sen_outs, _ = self.sen_rnn(sen_batch.view(batch_size, -1, self.input_dim))
        sen_rnn = sen_outs.contiguous().view(batch_size, -1, 2 * self.hidden_dim)  # (batch, sen_len, 2*hid)

        sentence_batch = self.bi_fetch(sen_rnn, sen_lengths, batch_size, self.sen_len)  # (batch_size, 2*hid)

        representation = sentence_batch
        out_prob = self.output(representation)
        return out_prob


for folds in range(5):

    # Hyper Parameters
    EPOCH = 500
    BATCH_SIZE = 16
    LR = 0.001
    top_seq = 100

    sequence2, label2 = check_data.get_data(
        os.path.join('/mnt/PTM/data/uniport/train/fasta/first' + str(top_seq) + '/val', str(folds), 'all.fa'))
    val_onehot_X, val_onehot_Y = dp.decode(sequence2, label2, top_seq)
    val_onehotX, val_onehotY = dp.reshape(val_onehot_X, val_onehot_Y)
    sequence1, label1 = check_data.get_data(
        os.path.join('/mnt/PTM/data/uniport/train/fasta/first' + str(top_seq) + '/train', str(folds), 'all.fa'))
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

    model = BLSTM(num_embeddings=20, embedding_dim=10, input_dim=100, hidden_dim=50, num_layers=1, output_dim=1)
    print(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    patience = 10
    early_stopping = EarlyStopping(patience, verbose=True)

    for epoch in range(EPOCH):
        val_pre_list = []
        val_lab_list = []
        for step, (b_x, b_y) in enumerate(train_loader):
            model.train()
            output = model(b_x.cuda())
            new_output = output[:, :top_seq]
            loss = binary_cross_entropy_with_logits(new_output, b_y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(str.format('step:{:d}, loss:{:.3f}', step, loss))
        del b_x, b_y

        all_pred = []
        all_val = []
        for val_step, (val_x, val_y) in enumerate(val_loader):
            model.eval()
            val_output = model(val_x.cuda())
            val_output = val_output[:, :top_seq]
            all_pred.extend(val_output.cpu().detach().numpy().tolist())
            all_val.extend(val_y.cpu().detach().numpy().tolist())
        del val_x, val_y

        strResults, f1, loss, acc = calculate_performance(np.array(all_val), np.array(all_pred), epoch, top_seq)
        early_stopping(f1, model)

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
