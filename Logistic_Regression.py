import numpy as np
from matplotlib import pyplot as plt

def loadData(txt):    
    iris_data = np.loadtxt(txt)
    class_label = iris_data[0:iris_data.shape[0],4]
    data_mat = iris_data[0:iris_data.shape[0],[0,1,2,3]]
    return data_mat,class_label

def gradAscent(data_mat,class_label):
    m,n = data_mat.shape
    numIter = 300
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (i + j + 1) + 0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(data_mat[randIndex] * weights))
            error = class_label[randIndex] - h
            weights = weights + alpha * error * data_mat[randIndex]
            del(dataIndex[randIndex])
    return weights

def sigmoid(z):
    return 1.0 / (1 + c(-z))

def plotBestFit(weights,txt):  
    data_mat,class_label = loadData(txt)
    dataArr = np.array(data_mat)
    n = dataArr.shape[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(class_label[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
            
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(1.0, 5.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

#main
data_mat1,class_label1 = loadData("C:/Users/jkjhhgg/Desktop/机器学习/商务智能作业/iris.txt")
data_mat2,class_label2 = loadData("C:/Users/jkjhhgg/Desktop/机器学习/商务智能作业/iris_test.txt")
weights = gradAscent(data_mat1,class_label1)
plotBestFit(weights,"C:/Users/jkjhhgg/Desktop/机器学习/商务智能作业/iris.txt")
plotBestFit(weights,"C:/Users/jkjhhgg/Desktop/机器学习/商务智能作业/iris_test.txt")
