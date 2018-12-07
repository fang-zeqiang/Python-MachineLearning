from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

#读取csv文件
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(r'/Users/fangzeqiang/Documents/GitHub/pyhton_to_learn/k-means_for_iris/iris.csv', names=names,encoding='utf-8')

#对类别进行编码，3个类别分别赋值0，1，2
dataset['class'][dataset['class']=='Iris-setosa']=0
dataset['class'][dataset['class']=='Iris-versicolor']=1
dataset['class'][dataset['class']=='Iris-virginica']=2


#算距离
def distEclud(vecA, vecB):                  #两个向量间欧式距离
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randChosenCent(dataSet,k):
    # 样本数
    m=shape(dataSet)[0]
    # 初始化列表
    centroidsIndex=[]
    #生成类似于样本索引的列表
    dataIndex=list(range(m))
    for i in range(k):
        #生成随机数
        randIndex=random.randint(0,len(dataIndex))
        #将随机产生的样本的索引放入centroidsIndex
        centroidsIndex.append(dataIndex[randIndex])
        #删除已经被抽中的样本
        del dataIndex[randIndex]
    #根据索引获取样本
    centroids = dataSet.iloc[centroidsIndex]
    return mat(centroids)

def kMeans(dataSet, k):
    # 样本总数
    m = shape(dataSet)[0]
    #分配样本到最近的簇：存[簇序号,距离的平方]
    # m行  2 列
    clusterAssment = mat(zeros((m,2)))
 
    #step1:
    #通过随机产生的样本点初始化聚类中心
    centroids = randChosenCent(dataSet, k)
    print('初始的聚类中心分别为：','\n','类簇1:',centroids[0],'\n','类簇2:',centroids[1],'\n','类簇3:',centroids[2])
 
    #标志位，如果迭代前后样本分类发生变化值为Tree，否则为False
    clusterChanged = True
    #查看迭代次数
    iterTime=0
    #所有样本分配结果不再改变，迭代终止
    while clusterChanged:   
        clusterChanged = False        
        #step2:分配到最近的聚类中心对应的簇中
        for i in range(m):
            #初始定义距离为无穷大
            minDist = inf;
            #初始化索引值
            minIndex = -1
            # 计算每个样本与k个中心点距离
            for j in range(k):
                #计算第i个样本到第j个中心点的距离
                distJI = distEclud(centroids[j,:],dataSet.values[i,:])
                #判断距离是否为最小
                if distJI < minDist:
                    #更新获取到最小距离
                    minDist = distJI
                    #获取对应的簇序号
                    minIndex = j
            #样本上次分配结果跟本次不一样，标志位clusterChanged置True
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #分配样本到最近的簇
        iterTime+=1
        sse=sum(clusterAssment[:,1])
        print('迭代 %d'%iterTime+' 次后，所有类簇误差平方和为 %f'%sse)
        #step3:更新聚类中心
        for cent in range(k):#样本分配结束后，重新计算聚类中心
            #获取该簇所有的样本点
            ptsInClust = dataSet.iloc[nonzero(clusterAssment[:,0].A==cent)[0]]
            #更新聚类中心：axis=0沿列方向求均值。
            centroids[cent,:] = mean(ptsInClust, axis=0) 
    return centroids, clusterAssment

#2维数据聚类效果显示
def datashow(dataSet,k,centroids,clusterAssment):  #二维空间显示聚类结果
    from matplotlib import pyplot as plt
    import matplotlib
    font = matplotlib.font_manager.FontProperties(fname=r'/Library/Fonts/Songti.ttc') #调用中文字体，防止乱码
    num,dim=shape(dataSet)  #样本数num ,维数dim
    
    if dim!=2:
        print('sorry,the dimension of your dataset is not 2!')
        return 1
    marksamples=['or','ob','og','ok','^r','^b','<g'] #样本图形标记
    if k>len(marksamples):
        print('sorry,your k is too large,please add length of the marksample!')
        return 1
        #绘所有样本
    for i in range(num):
        markindex=int(clusterAssment[i,0])#矩阵形式转为int值, 簇序号
        #特征维对应坐标轴x,y；样本图形标记及大小
        plt.plot(dataSet.iat[i,0],dataSet.iat[i,1],marksamples[markindex],markersize=6)
 
    #绘中心点            
    markcentroids=['^','^','^']#聚类中心图形标记
    label=['cluster_1','cluster2','cluster3']
    c=['red','blue','green']
    for i in range(k):
        plt.plot(centroids[i,0],centroids[i,1],markcentroids[i],markersize=15,label=label[i],c=c[i])
        plt.legend(loc = 'upper left')
    plt.xlabel('萼片长度',fontproperties=font)  
    plt.ylabel('萼片宽度',fontproperties=font) 
   
    plt.title('k-means 类簇中心分布',fontproperties=font)      
    plt.show()

#聚类前，绘制原始的样本点
def originalDatashow(dataSet):
        #样本的个数和特征维数
    import matplotlib
    num,dim=shape(dataSet)
    marksamples=['ob'] #样本图形标记
    for i in range(num):
        plt.plot(datamat.iat[i,0],datamat.iat[i,1],marksamples[0],markersize=5,c='gray')
    font = matplotlib.font_manager.FontProperties(fname=r'/Library/Fonts/Songti.ttc') #调用中文字体，防止乱码
    plt.title('初始数据集',fontproperties=font)
    plt.xlabel('萼片长度',fontproperties=font)  
    plt.ylabel('萼片宽度',fontproperties=font) #标题
    plt.show()

if __name__=='__main__':
#=====kmeans聚类
    # # #获取样本数据
    datamat=dataset.loc[:, ['sepal-length','sepal-width']]
    #真实的标签
    labels=dataset.loc[:, ['class']]
    # #原始数据显示
    originalDatashow(datamat)
 
    # #*****kmeans聚类
    k=3 #用户定义聚类数
    mycentroids,clusterAssment=kMeans(datamat,k)
 
    #绘图显示
    datashow(datamat,k,mycentroids,clusterAssment)
    #trgartshow(datamat,3,labels)
