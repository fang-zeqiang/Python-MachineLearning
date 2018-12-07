import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random

with open(r'/Users/fangzeqiang/Documents/GitHub/pyhton_to_learn/k-means_for_iris/iris.csv') as file: 
    txt=file.read().split('\n')                                 
datas=[]
dname=txt[0].split(',')

for i in txt[1:]:
    datas.append(dict(zip(dname,i.split(','))))

IniSet=np.array([[x['sepal_length'],x['sepal_width'],x['petal_length'],x['petal_width']] for x in datas],dtype=float)

#类蔟为2，生成两个随机数

n=random.randint(0,149)
m=random.randint(0,149)
while m==n:
    m=random.randint(1,149)
#获得初始均值向量{μ1,μ2}
miu1=IniSet[n]
miu2=IniSet[m]
#令Ci={}
dis_j1=IniSet
dis_j2=IniSet
for i in range(0,150):
    dis_j1[i]=IniSet[i]-miu1
    dis_j2[i]=IniSet[i]-miu2
p=np.dot(dis_j1[1],dis_j1[1])
print(p)