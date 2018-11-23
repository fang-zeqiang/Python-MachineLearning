import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = matplotlib.font_manager.FontProperties(fname=r'/Library/Fonts/Songti.ttc') #调用中文字体，防止乱码

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def convert_data(datas):
    """
    转换读入数据为numpy.ndarray
    """
    return np.array([[x['sepal_length'],x['sepal_width'],x['class']] for x in datas],dtype=float)

def process_data(data):
    """
    处理数据
    """
    dataMatIn = data[:, 0:-1]
    classLabels = data[:, -1]
    dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  #特征数据集，添加1是构造常数项x0
    return dataMatIn, classLabels

def stoc_grad_ascent(dataMatIn, classLabels, numIter=150):
    """
    随机梯度上升
    """
    m, n = np.shape(dataMatIn)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1 + i + j) + 0.01 #保证多次迭代后新数据仍然有影响力
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatIn[i] * weights))  # 数值计算
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatIn[i]
            del(dataIndex[randIndex])
    return weights.tolist()

def plotBestFIt(data,weights,title="None"):
    """
    结果绘图
    """
    dataMatIn, classLabels = process_data(data)
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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(2, 8, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]  #matix
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title,fontproperties=font)
    plt.show()

def read(file):
    """
    读取和处理数据,每条数据以dict存放
    """
    with open(file) as file:
        txt=file.read().split('\n')

    datas=[]
    dname=txt[0].split(',')

    for i in txt[1:]:
        datas.append(dict(zip(dname,i.split(','))))

    csf={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
    for i in datas:
        i["class"]=csf[i["class"]]
    return datas

if __name__ == '__main__':
    datas=read(r'/Users/fangzeqiang/Desktop/iris/iris.csv')
    
    trains=datas[:30]+datas[50:80] #用于训练的数据
    left=datas[31:51]+datas[81:101] #用于验证用的数据

    trains_array=convert_data(trains)
    left_array=convert_data(left)
    
    dataMatIn, classLabels=process_data(trains_array)
    weights=stoc_grad_ascent(dataMatIn, classLabels, numIter=150)

    plotBestFIt(trains_array,weights,"训练数据")
    plotBestFIt(left_array,weights,"验证数据")
    



