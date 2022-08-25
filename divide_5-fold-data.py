import random
import os
def divide_train_val():
    root_path= r'example-data'
    target_path = r'5-fold-data'
    # file_list=os.listdir(root_path)
    file_list=[]
    list1=[]
    list2=[]
    list3 = []
    list4 = []
    list5 = []
    old_num=0
    divide_num=0

    for file_name in ['train']:
        with open(os.path.join(root_path,file_name+'.fa'),'r')as f:
            lines=f.readlines()
            old_num=old_num+len(lines)
            f.close()
            list_temp=[]
            flag=0
            for i in range(len(lines)):
                if i%2==0:
                    dict={lines[i]:lines[i+1]}
                    list_temp.append(dict)
            if len(lines)/2 != len(list_temp):
                print(file_name)
            if len(list_temp)<5:
                list_sample=random.sample(range(1, 6), len(list_temp))
                for sample_i in range(len(list_sample)):
                    if list_sample[sample_i]==1:
                        list1.append(list_temp[sample_i])
                    elif list_sample[sample_i]==2:
                        list2.append(list_temp[sample_i])
                    elif list_sample[sample_i]==3:
                        list3.append(list_temp[sample_i])
                    elif list_sample[sample_i]==4:
                        list4.append(list_temp[sample_i])
                    elif list_sample[sample_i]==5:
                        list5.append(list_temp[sample_i])
            else:
                # list_divide=split_list(list_temp)
                list_sample = random.sample(range(0, len(list_temp)), len(list_temp))
                num=len(list_sample)/5
                for sample_i in range(len(list_temp)):
                    if sample_i<num:
                        list1.append(list_temp[list_sample[sample_i]])
                    elif num<=sample_i<num*2:
                        list2.append(list_temp[list_sample[sample_i]])
                    elif num*2<=sample_i<num*3:
                        list3.append(list_temp[list_sample[sample_i]])
                    elif num*3<=sample_i<num*4:
                        list4.append(list_temp[list_sample[sample_i]])
                    else:
                        list5.append(list_temp[list_sample[sample_i]])
                # print()
            divide_num=len(list1)+len(list2)+len(list3)+len(list4)+len(list5)
            print('divide num:'+str(divide_num*2))
            print('old_num:'+str(old_num))

    strr1=''

    for i in range(len(list1)):
        (key,value),=list1[i].items()
        strr1=strr1+key+value
    with open(os.path.join(target_path,'1.fa'),'w')as f:
        f.write(strr1)

    strr2 = ''
    for i in range(len(list2)):
        (key, value), = list2[i].items()
        strr2 = strr2 + key + value
    with open(os.path.join(target_path,'2.fa'),'w')as f:
        f.write(strr2)
    strr3 = ''
    for i in range(len(list3)):
        (key, value), = list3[i].items()
        strr3 = strr3 + key + value
    with open(os.path.join(target_path,'3.fa'),'w')as f:
        f.write(strr3)
    strr4 = ''
    for i in range(len(list4)):
        (key, value), = list4[i].items()
        strr4 = strr4 + key + value
    with open(os.path.join(target_path,'4.fa'),'w')as f:
        f.write(strr4)
    strr5 = ''
    for i in range(len(list5)):
        (key, value), = list5[i].items()
        strr5 = strr5 + key + value
    with open(os.path.join(target_path,'5.fa'),'w')as f:
        f.write(strr5)
divide_train_val()