import numpy as np
import matplotlib.pyplot as plt
import matplotlib

'''
打开桌面的iris.csv数据集
建立一个文件txt逐行读取iris内容
'''
with open(r'/Users/fangzeqiang/Desktop/iris/iris.csv') as file: 
    txt=file.read().split('\n')                                 
datas=[]
dname=txt[0].split(',')

'''
每行数据以字典类存放，字典在存放于datas数组中
从第一行开始循环到最后一行
每行和第0行的属性名称用zip方法打包添加进数组
'''
for i in txt[1:]:
    datas.append(dict(zip(dname,i.split(','))))

'''
将数组里的class属性下类别名称用0,1,2替换方便处理
'''
csf={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
for i in datas:
    i["class"]=csf[i["class"]]

'''
将data中Iris-setosa前30条和Iris-versicolor的前30条
添加到trains数组中作为训练集
将这两类余下的各20条数据作为测试集
{'sepal_length': '5.1', 'sepal_width': '3.5', 'petal_length': '1.4', 'petal_width': '0.2', 'class': 0}
'''
trains=datas[0:30]+datas[50:80]
verifys=datas[30:50]+datas[80:100]

"""
将每行的sepal_length,sepal_width,class属性提取出来
打包成一组赋给trains_array值
"""
trains_array=np.array([[x['sepal_length'],x['sepal_width'],x['class']] for x in trains],dtype=float)

"""
实现花的特征数据和类名分开存放
并在特征数据的每行数据前添加一个常数项x0=1.0,方便逻辑回归计算
"""
dataMatIn = trains_array[:, 0:-1]
classLabels = trains_array[:, -1]
dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  

"""
随机梯度上升计算权值
"""
numIter=150        
m, n = np.shape(dataMatIn)                     #m represent the size of dataMatIn
weights = np.ones(n)                           #n represent the num of attributes
for j in range(numIter):
    dataIndex = list(range(m))
    for i in range(m):
        alpha = 4 / (1 + i + j) + 0.01         #保证多次迭代后新数据仍然有影响力
        randIndex = int(np.random.uniform(0, len(dataIndex)))
        h= 1 / (1 + np.exp(-sum(dataMatIn[i] * weights)))#sigmoid function
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatIn[i]
        del(dataIndex[randIndex])
weights=weights.tolist()
print(weights)

'''
绘制两个特征属性散点图，并绘制逻辑回归线性函数
'''
n = np.shape(dataMatIn)[0]
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []
for i in range(n):
    if classLabels[i] == 1:
        xcord1.append(dataMatIn[i][1])
        ycord1.append(dataMatIn[i][2])
    else:
        xcord2.append(dataMatIn[i][1])
        ycord2.append(dataMatIn[i][2])
plt.scatter(xcord1, ycord1,s=30, c='red',marker='x')
plt.scatter(xcord2, ycord2, s=30, c='blue',marker='x')
x = np.arange(2, 8, 0.1)                         #x轴坐标数值从2到8每0.1一格
y = (-weights[0] - weights[1] * x) / weights[2]  #matix
plt.plot(x, y,c='black')# draw the plot line to classify the two kind of flower
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')

font = matplotlib.font_manager.FontProperties(fname=r'/Library/Fonts/Songti.ttc') #调用中文字体，防止乱码
plt.title("训练数据分类",fontproperties=font)
plt.show()


"""
----------------------------------------------------------------------------------------------------------
以下同上训练集方法对函数进行测试
"""

verifys_array=np.array([[x['sepal_length'],x['sepal_width'],x['class']] for x in verifys],dtype=float)

dataMatIn = verifys_array[:, 0:-1]
classLabels = verifys_array[:, -1]
dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  

n = np.shape(dataMatIn)[0]
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []
for i in range(n):
    if classLabels[i] == 1:
        xcord1.append(dataMatIn[i][1])
        ycord1.append(dataMatIn[i][2])
    else:
        xcord2.append(dataMatIn[i][1])
        ycord2.append(dataMatIn[i][2])
plt.scatter(xcord1, ycord1,s=30, c='red',marker='x')
plt.scatter(xcord2, ycord2, s=30, c='blue',marker='x')
x = np.arange(2, 8, 0.1)
y = (-weights[0] - weights[1] * x) / weights[2]  
plt.plot(x, y,c='black')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
font = matplotlib.font_manager.FontProperties(fname=r'/Library/Fonts/Songti.ttc') #调用中文字体，防止乱码
plt.title("验证数据分类",fontproperties=font)
plt.show()
print('斜率k=',-weights[1]/weights[2])
print('截距b=',-weights[0]/weights[2])