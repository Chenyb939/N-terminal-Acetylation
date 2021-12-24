import math


# def count_fa_num(filepath):
#     for line in open(filepath, "r"):
#         fa_data.append(line)
#         fa_num += 1
#     return fa_num/2
from matplotlib import pyplot as plt


def all_result(tp_num,fp_num,tn_num,fn_num,P_num):
    accuracy = (tp_num + tn_num) / (tp_num + tn_num + fp_num + fn_num)
    error_rate = (fp_num + fn_num) / (tp_num + tn_num + fp_num + fn_num)
    sensitive = tp_num / (tp_num + fp_num)
    specificity = tn_num / (tn_num + fn_num)
    precison = tp_num / P_num
    recall = tp_num / (tp_num + fn_num)
    f1 = (2 * precison * recall) / (precison + recall)
    MCC = (tp_num * tn_num - tp_num * fn_num) / math.sqrt(
        (tp_num + fp_num) * (tp_num + fn_num) * (tn_num + fp_num) * (tn_num * fn_num))
    TPR = tp_num / (tp_num + fn_num)  # true positive rate
    spc = tn_num / (tn_num + fp_num)  # specificity
    PPV = tp_num / (tp_num + fp_num)  # positive prediction value

    print("accuracy:%f" % accuracy)
    print("error_rate:%f" % error_rate)
    print("sensitive:%f" % sensitive)
    print("specificity:%f" % specificity)
    print("precision:%f" % precison)
    print("recall:%f" % recall)
    print("f1:%f" % f1)
    print("MCC:%f" % MCC)
    print("TPR:%f" % TPR)
    print("spc:%f" % spc)
    print("PPV:%f" % PPV)

    # Precison = tpsum / (tpsum + fpsum)
    # Recall = tpsum / (tpsum + fnsum)
    # print("lirecall", lirecall)
    # print("liprecision", liprecision)
    # print("sum:",Precison,Recall)
    # plt.plot(lirecall, liprecision, 'o', label='each kind')
    # return accuracy, error_rate, sensitive, specificity, precison, recall, f1
    return precison, recall


def pail_draw(recall, precison):
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.scatter(recall, precison, color='red', label='pall')
    plt.legend()
    plt.show()


# def print_result():
    # print("pre_num:%d" % pre_num)
    # print("p_num:%d" % p_num)
    # print("tp_num:%d" % tp_num)
    # print("fp_num:%d" % fp_num)
    # print("n_num:%d" % (n_num * 3))
    # print("tn_num:%d" % tn_num)
    # print("fn_num:%d" % n_num)


def count_actual_gene_info(filepath):
    K_num=0
    P_num=0
    fa_in=open(filepath,'r')
    for line in fa_in.readlines():
        line=line.rstrip()
        if line[0]=='>':
            ls=line.replace('>','').split('\t')
            name=ls[0]
            str_num=ls[1]
            str_num_list=str_num.split(',')
            P_num+=len(str_num_list)-1
        else:
            str0=line
            K_num+=str0.count('K',0,100)
    return K_num,P_num
def count_prediction_gene_info(file_fa,file_txt):

    tp_num = 0
    fp_num = 0
    K_num , P_num=count_actual_gene_info(file_fa)

    res_in = open(file_txt, 'r')
    lists = res_in.readlines( )
    lenth=len(lists)
    gene_num = set( )

    for lid in range(lenth):
        line=lists[lid].rstrip()
        if line[0] == '>':
            ls = line.replace('>' , '').split('\t')
            gene_num.clear()
            # 把发生乙酰化的位点存进gum
            gnum = ls[1].split(',')
            l1=len(gnum)
            for g in gnum:
                if g=='':
                    break
                gene_num.add(g)
        elif line[1]=='e':
            continue
        else:
            ls=line.split('\t')
            pre_num=ls[1]
            if pre_num in gene_num:
                tp_num=tp_num+1
            else:
                if int(pre_num)>100:
                    continue
                fp_num=fp_num+1
    fn_num=P_num-tp_num
    N_num=K_num-P_num
    tn_num=N_num-fp_num
    return tp_num,tn_num,fp_num,fn_num,K_num,P_num,N_num



# if __name__ == '__main__':
def pall_eva():
    file_fa=r'D:\Users\omen\PycharmProjects\N-terminal-Acetylation\data\uniport\test\fasta\first100\\m_K.fa'
    file_res=r'D:\Users\omen\PycharmProjects\N-terminal-Acetylation\data\uniport\result\pall\K.txt'

    # count_actual_gene_position('E:\\hefei\\PTM\\data\\test_uniport\\train\\fasta\\first3\\A.fa')
    tp_num,tn_num,fp_num,fn_num,K_num,P_num,N_num=count_prediction_gene_info(file_fa,file_res)

    print("K_num:%d" % K_num)
    print("P_num:%d" % P_num)
    print("tp_num:%d" % tp_num)
    print("fp_num:%d" % fp_num)
    print("N_num:%d" % N_num)
    print("tn_num:%d" % tn_num)
    print("fn_num:%d" % fn_num)

    Precision, Recall = all_result(tp_num,fp_num,tn_num,fn_num,P_num)
    return Precision, Recall
    # pail_draw(Recall, Precision)

    # fa_num=count_fa_num(file_fa)
    # accuracy , error_rate , sensitive , specificity , precison , recall , f1=all_result(tp_num,fp_num,tn_num,fn_num)
