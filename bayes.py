#the simplest bayes
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

with open(r'C:\Users\FZQ\Desktop\python_to_learn\marryhim.csv',encoding='utf-8-sig') as file:
    txt=file.read().split('\n')
datas=[]
dname=txt[0].split(',')
#print(dname)

for i in txt[1:]:
    datas.append(dict(zip(dname,i.split(','))))
#print(datas)

datas_array=np.array([[x['帅'],x['性格'],x['身高'],x['上进'],x['嫁否']] for x in datas],dtype=int)
#print(datas_array)




data_size=np.shape(datas_array)[0]
marry_1=0
marry_0=0
j=0
while j<data_size:
    if datas_array[j][4]==1:
       marry_1+=1 
    j+=1
marry_0 = data_size - marry_1

p_m_1=marry_1/data_size
p_m_0=1-p_m_1  

j=0
handsome_0=0
handsome_1=0
while j<data_size:
    if datas_array[j][4]==1 and datas_array[j][0]==0:
        handsome_0+=1
    if datas_array[j][4]==0 and datas_array[j][0]==0:
        handsome_1+=1
    j+=1

j=0
handsome_0=0
handsome_1=0
while j<data_size:
    if datas_array[j][4]==1 and datas_array[j][1]==0:
        handsome_0+=1
    if datas_array[j][4]==0 and datas_array[j][0]==0:
        handsome_1+=1
    j+=1

array_1={}
array_2={}
for i in range(12):
    array_1[i]=0
    array_2[i]=0
print(array_1)

'''
for i in range(0,4):
    for j in range(0,11):
        if datas_array[j][4]==1 and datas_array[j][i]==0:
            array_1[]+=1
        if datas_array[j][4]==0 and datas_array[j][i]==0:
            array_2=[]+=1
'''
'''
for i in f:
    tmp=i.split(',')                #传数据给x,y轴
    x.append(float(tmp[0]))
    y.append(float(tmp[1]))

plt.scatter(x, y,  color='red')     #绘制散点图

Xsum=0.0
X2sum=0.0
Ysum=0.0
XY=0.0
n=len(x)
for i in range(n):
    Xsum+=x[i]
    Ysum+=y[i]
    XY+=x[i]*y[i]
    X2sum+=x[i]**2
k=(Xsum*Ysum/n-XY)/(Xsum**2/n-X2sum)#斜率
b=(Ysum-k*Xsum)/n                   #截距
print('the line is y=%f*x+%f' % (k,b) )
                                    #打印回归模型函数
for j in range(n):
    y[j]=x[j]*k+b
plt.plot(x,y)                       #绘制回归模型图
#plt.show()                          #函数呈现
'''

