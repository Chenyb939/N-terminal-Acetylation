import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
with open('aaindex_feature.txt','r')as f:
    lines=f.readlines()
lines=lines[1:]

feature_all=[]
residue_names=[]
for line in lines:
    feature_all.append([float(i) for i in line.split()[1:]])
    residue_names.append(line.split()[0])
feature_all=np.array(feature_all)
feature_all=mm.fit_transform(feature_all)
strr=''
for i in range(len(residue_names)):
    strr=strr+'letterDict[\''+residue_names[i]+'\']=['+','.join(list(map(str,feature_all[i])))+']'+'\n'
with open('aaindex_feature_to_dp.txt','w')as f:
    f.write(strr)
