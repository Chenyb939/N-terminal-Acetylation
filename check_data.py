import os
import dp
import numpy as np
import pandas as pd


def complex_name(filepath1, filepath2, outdir):
    df1 = pd.read_csv(filepath1)
    df2 = pd.read_csv(filepath2)
    list = ["gname", "gsets", "gsete", "gann"]
    same_df = pd.DataFrame(columns=list)
    diff_df = df1
    for ind1, row1 in df1.iterrows():
        if int(ind1) % 100 == 0:
            print(str(ind1) + ' items has been proceed.')
        for ind2, row2 in df2.iterrows():
            names1 = row1['gname']
            names2 = row2['gname'].split(',')[0: -1]
            for i in names2:
                if i in names1 and row1['gsets'] == row2['gsets']:
                    same_df = same_df.append(row1, ignore_index=True)
                    try:
                        diff_df = diff_df.drop(ind1)
                        print(ind1)
                    except:
                        print(ind1)
                    break

    diff_df.to_csv(os.path.join(outdir, 'diff.csv'), index=False)
    same_df.to_csv(os.path.join(outdir, 'same.csv'), index=False)


def remove_duplication(filepath, filename, outdir):
    df = pd.read_csv(os.path.join(filepath, filename))
    new_df = df.drop_duplicates(subset=['gname', 'gsets'], keep='first', inplace=False)
    new_df.to_csv(os.path.join(outdir, 'fin_' + filename), index=False)


def to_fasta(filepath, filename, outdir, outname, top):
    df = pd.read_csv(os.path.join(filepath, filename))
    with open(os.path.join(outdir, outname), 'w') as file:
        for _, row in df.iterrows():
            name = str(row['gname'][:-1])
            set = str(row['gsets'])
            seq = str(row['gseq'])
            if int(set) <= top:
                file.write('>' + name + '\t' + set + '\n')
                file.write(seq + '\n')


def sub_data(filepath, filename, outdir):
    i = 0
    df = pd.read_csv(os.path.join(filepath, filename))
    fileA = open(os.path.join(outdir, 'A.fa'), 'w')
    fileC = open(os.path.join(outdir, 'C.fa'), 'w')
    fileD = open(os.path.join(outdir, 'D.fa'), 'w')
    fileG = open(os.path.join(outdir, 'G.fa'), 'w')
    fileM = open(os.path.join(outdir, 'M.fa'), 'w')
    fileP = open(os.path.join(outdir, 'P.fa'), 'w')
    fileS = open(os.path.join(outdir, 'S.fa'), 'w')
    fileT = open(os.path.join(outdir, 'T.fa'), 'w')
    fileV = open(os.path.join(outdir, 'V.fa'), 'w')
    fileK = open(os.path.join(outdir, 'K.fa'), 'w')
    fileR = open(os.path.join(outdir, 'R.fa'), 'w')
    fileOth = open(os.path.join(outdir, 'Other.fa'), 'w')
    for _, row in df.iterrows():
        ann = row['gann']
        name = str(row['gname'][:-1])
        set = str(row['gsets'])
        seq = str(row['gseq'])
        if seq == '':
            print(name)
        if 'acetylalanine' in ann:
            fileA.write('>' + name + '\t' + set + '\n')
            fileA.write(seq + '\n')
        elif 'acetylcysteine' in ann:
            fileC.write('>' + name + '\t' + set + '\n')
            fileC.write(seq + '\n')
        elif 'acetylaspartate' in ann:
            fileD.write('>' + name + '\t' + set + '\n')
            fileD.write(seq + '\n')
        elif 'acetylglycine' in ann:
            fileG.write('>' + name + '\t' + set + '\n')
            fileG.write(seq + '\n')
        elif 'acetylmethionine' in ann:
            fileM.write('>' + name + '\t' + set + '\n')
            fileM.write(seq + '\n')
        elif 'acetylproline' in ann:
            fileP.write('>' + name + '\t' + set + '\n')
            fileP.write(seq + '\n')
        elif 'acetylserine' in ann:
            fileS.write('>' + name + '\t' + set + '\n')
            fileS.write(seq + '\n')
        elif 'acetylthreonine' in ann:
            fileT.write('>' + name + '\t' + set + '\n')
            fileT.write(seq + '\n')
        elif 'acetylvaline' in ann:
            fileV.write('>' + name + '\t' + set + '\n')
            fileV.write(seq + '\n')
        elif 'acetyllysine' in ann:
            fileK.write('>' + name + '\t' + set + '\n')
            fileK.write(seq + '\n')
        elif 'acetylarginine' in ann:
            fileR.write('>' + name + '\t' + set + '\n')
            fileR.write(seq + '\n')
        else:
            print(ann)
            fileOth.write('>' + name + '\t' + set + '\n')
            fileOth.write(seq + '\n')


def first_csv(filepath, filename, outdir, num):
    df = pd.read_csv(os.path.join(filepath, filename))
    new_df = df[df['gsets'] <= num]
    new_df.to_csv(os.path.join(outdir, 'fin_' + str(num) + '_' + filename), index=False)


def merge_test(filepath, outdir):
    str = ""
    list = ['same.fa', 'Other.fa', 'diff.fa']
    filenames = os.listdir(filepath)
    for name in filenames:
        if name not in list:
            with open(os.path.join(filepath, name), 'r') as filer:
                lines = filer.readlines()
                for line in lines:
                    str = str + line
    file = open(os.path.join(outdir, 'test.fa'), 'w')
    file.write(str)


def split_data(filepath, outdir, time, val_rate=0.2):
    files = dp.findfile(filepath)
    files.remove('m_Other.fa')
    files.remove('m_same.fa')
    files.remove('m_test.fa')
    files.remove('train')
    files.remove('val')
    for file in files:
        if file.startswith('m'):
            df = dp.fa_to_df(os.path.join(filepath, file))
            df_new = df.sample(frac=1)
            df_val = df_new[0: int(len(df_new) * val_rate)]
            df_train = df_new[int(len(df_new) * val_rate):]

            dp.df_to_fa(df_val, os.path.join(outdir, 'val', str(time), file))
            dp.df_to_fa(df_train, os.path.join(outdir, 'train', str(time), file))
        else:
            continue


def add_data(filepath):
    files = dp.findfile(filepath)
    for file in files:
        if os.path.exists(os.path.join(filepath, 'all.fa')):
            filew = open(os.path.join(filepath, 'all.fa'), 'a')
        else:
            filew = open(os.path.join(filepath, 'all.fa'), 'w')
        with open(os.path.join(filepath, file), 'r') as filer:
            lines = filer.readlines()
            for line in lines:
                filew.write(line)
        filew.close()


def cut_seq(df, length=100):
    label = []
    sequence = []
    for indexs in df.index:
        name = df.loc[indexs].values[0].replace('\n', '')
        seq = df.loc[indexs].values[1].replace('\n', '')
        label.append(name.split('\t')[1])
        if len(seq) >= 100:
            sequence.append(seq[:100])
        else:
            sequence.append(seq.ljust(length, 'X'))
    return sequence, label


def cut_seq_PC_pseAAC(df, length=100):
    label = []
    sequence = []
    for indexs in df.index:
        name = df.loc[indexs].values[0].replace('\n', '')
        seq = df.loc[indexs].values[1].replace('\n', '')
        label.append(name.split('\t')[1])
        if len(seq) >= 100:
            sequence.append(list(map(float,seq.split())) )
        else:
            sequence.append(list(map(float,seq.split())))
    return sequence, label

def get_data(filepath):
    df = dp.fa_to_df(filepath)
    df = df.sample(frac=1, random_state=1)
    sequence, label = cut_seq(df)
    return sequence, label


def get_data_PC_pseAAC(filepath):
    df = dp.fa_to_df(filepath)
    df = df.sample(frac=1, random_state=1)
    sequence, label = cut_seq_PC_pseAAC(df)
    return sequence, label


def decode(allseq, label, num_len):
    lMatr = np.zeros((len(label), num_len))
    for item in range(len(label)):
        site = int(label[item]) - 1
        lMatr[item][site] = 1
    #['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["X"] = 20  # add -
    AACategoryLen = 21  # add -
    # print(len(allseq[0]))
    probMatr = np.zeros((len(allseq), 1, len(allseq[0]), AACategoryLen))

    # print(sampleSeq3DArr)
    sampleNo = 0
    for sequence in allseq:
        AANo = 0
        for AA in sequence:
            if not AA in letterDict:
                probMatr[sampleNo][0][AANo] = np.full((1, AACategoryLen), 1.0 / AACategoryLen)
            else:
                index = letterDict[AA]
                probMatr[sampleNo][0][AANo][index] = 1
            AANo += 1
        sampleNo += 1
    return probMatr, lMatr


def merge_fa(filepath):
    filenames = os.listdir(filepath)
    for name in filenames:
        if name.startswith('m'):
            continue
        else:
            gen_list = []
            set_list = []
            seq_list = []
            set_list_s = []
            filein = open(os.path.join(filepath, name), 'r')

            lines = filein.readlines()
            for line in lines:
                if line.startswith('>'):
                    gen_list.append(line.replace('>', '').replace('\n', '').split('\t')[0])
                    set_list.append(line.replace('>', '').replace('\n', '').split('\t')[1])
                else:
                    seq_list.append(line.replace('\n', ''))
            gen_list_s = list(set(gen_list))
            if len(gen_list) == len(gen_list_s):
                with open(os.path.join(filepath, 'm_' + name), 'w') as fileout:
                    for line in lines:
                        fileout.write(line)
                print(name + ' no change')
            else:
                for item in gen_list_s:
                    index_list = [a for a, b in enumerate(gen_list) if b == item]
                    set_list_s.append(index_list)
                with open(os.path.join(filepath, 'm_' + name), 'w') as fileout:
                    for newsets in range(len(set_list_s)):
                        newset = ''
                        for news in set_list_s[newsets]:
                            newname = gen_list[news]
                            newset += set_list[news] + ','
                            newseq = seq_list[news]
                        fileout.write('>' + newname + '\t' + newset + '\n')
                        fileout.write(newseq + '\n')


if __name__ == '__main__':
    # all data
    # complex_name('/home/chenyb/PTM/data/uniport/2021/data.csv', '/home/chenyb/PTM/data/uniport/2015/data.csv',
    #              '/home/chenyb/PTM/data/uniport')

    # remove_duplication('/home/chenyb/PTM/data/uniport/', 'diff.csv', '/home/chenyb/PTM/data/uniport')
    # remove_duplication('/home/chenyb/PTM/data/uniport/', 'same.csv', '/home/chenyb/PTM/data/uniport')

    # to_fasta('/home/chenyb/PTM/data/uniport/', 'fin_diff.csv', '/home/chenyb/PTM/data/uniport/fasta', 'diff.fa')
    # to_fasta('/home/chenyb/PTM/data/uniport/', 'fin_same.csv', '/home/chenyb/PTM/data/uniport/fasta', 'same.fa')

    # sub data
    # sub_data('/home/chenyb/PTM/data/uniport/train', 'fin_same.csv', '/home/chenyb/PTM/data/uniport/train/fasta')
    # sub_data('/home/chenyb/PTM/data/uniport/test', 'fin_diff.csv', '/home/chenyb/PTM/data/uniport/test/fasta')

    # # first 100
    # to_fasta('/home/chenyb/PTM/data/uniport/test/', 'fin_diff.csv',
    #          '/home/chenyb/PTM/data/uniport/test/fasta/first50', 'diff.fa', 50)
    # to_fasta('/home/chenyb/PTM/data/uniport/train/', 'fin_same.csv',
    #          '/home/chenyb/PTM/data/uniport/train/fasta/first50', 'same.fa', 50)
    # first_csv('/home/chenyb/PTM/data/uniport/test/', 'fin_diff.csv', '/home/chenyb/PTM/data/uniport/test/', 50)
    # first_csv('/home/chenyb/PTM/data/uniport/train/', 'fin_same.csv', '/home/chenyb/PTM/data/uniport/train/', 50)
    # sub_data('/home/chenyb/PTM/data/uniport/train', 'fin_50_fin_same.csv',
    #          '/home/chenyb/PTM/data/uniport/train/fasta/first50')
    # sub_data('/home/chenyb/PTM/data/uniport/test', 'fin_50_fin_diff.csv',
    #          '/home/chenyb/PTM/data/uniport/test/fasta/first50')

    # # merge test data
    # merge_test('/home/chenyb/PTM/data/uniport/test/fasta/first50', '/home/chenyb/PTM/data/uniport/test/fasta/first50')
    # merge_test('/home/chenyb/PTM/data/uniport/train/fasta/first50', '/home/chenyb/PTM/data/uniport/train/fasta/first50')
    #
    # merge_fa('/home/chenyb/PTM/data/uniport/train/fasta/first50/')
    # merge_fa('/home/chenyb/PTM/data/uniport/test/fasta/first50/')

    for time in range(5):
        split_data('/home/chenyb/PTM/data/uniport/train/fasta/first50', '/home/chenyb/PTM/data/uniport/train/fasta/first50', time, val_rate=0.2)
        add_data(os.path.join('/home/chenyb/PTM/data/uniport/train/fasta/first50/train', str(time)))
        add_data(os.path.join('/home/chenyb/PTM/data/uniport/train/fasta/first50/val', str(time)))
