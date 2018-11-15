import matplotlib.pyplot as plt 
import pandas as pd
import os

os.chdir(r"F:\大学\大三上\商务智能\软件1601-2120163203-奚宇星-第三次作业\iris")

with open("iris.csv") as file:
        txt=file.read().split('\n')

datas=[]
dname=txt[0].split(',')

for i in txt[1:]:
     datas.append(dict(zip(dname,i.split(','))))

csf={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
for i in datas:
    i["种类"]=csf[i["种类"]]



