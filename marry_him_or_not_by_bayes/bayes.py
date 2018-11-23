#the naive bayes
import numpy as np

with open(r'/Users/fangzeqiang/Documents/GitHub/pyhton_to_learn/marryhim.csv',encoding='utf-8-sig') as file:
    txt=file.read().split('\n')#
'''
读取文档并预处理存入txt
打印txt，结果如下：
['帅,性格,身高,上进,嫁否', '1,0,0,0,0',...,'1,1,0,0,0']
'''
datas=[]
dname=txt[0].split(',')
for i in txt[1:]:
    datas.append(dict(zip(dname,i.split(','))))
'''
对数据进一步处理，结果存入datas数组中
打印当前datas，结果如下：
[{'帅': '1', '性格': '0', '身高': '0', '上进': '0', '嫁否': '0'},
    ...,
    {'帅': '1', '性格': '1', '身高': '0', '上进': '1', '嫁否': '0'}]
'''
datas_array=np.array([[x['帅'],x['性格'],x['身高'],x['上进'],x['嫁否']] for x in datas],dtype=int)
'''
对数据用numpy中的方法再加工，存入data_array
打印当前datas_array,结果如下：
[[1 0 0 0 0]
 [0 1 0 1 0]
 ...
 [1 1 0 0 0]]
'''
n=np.shape(datas_array)[0]    #n为数组行数12
m=np.shape(datas_array)[1]    #m为数组列数5
marry_1=0                     #嫁人的个数
marry_0=0                     #不嫁的个数
for i in range(n):
    if datas_array[i][m-1]==1:#计算嫁人数
       marry_1+=1 
marry_0 = n - marry_1         #不嫁人数=总数-嫁人数
p_m_1=marry_1/ n              #p(c=1)
p_m_0=1-p_m_1                 #p(c=0)

array_1={}   #存放各个属性为0的频数的数组
array_2={}   #存放各个属性为0且女方又嫁的频数的数组
array_3={}   #存放各个属性为0且女方不嫁的频数的数组
for i in range(4):
    array_1[i]=0
    array_2[i]=0
    array_3[i]=0    #为每个数组赋初值

for i in range(m-1):
    for j in range(0,11):
            if datas_array[j][i]==0:
                array_1[i]+=1           #计算各个属性为0的频数
                if datas_array[j][m-1]==1 :
                    array_2[i]+=1       #计算各个属性为0且女方又嫁的频数   
                if datas_array[j][m-1]==0 :
                    array_3[i]+=1       #计算各个属性为0且女方不嫁的频数
p_x=1
for i in array_1:
    p_x*=array_1[i]/m           #计算全部属性并存的频率
p_x_m_1=1
for i in array_2:
    p_x_m_1*=array_2[i]/marry_1 #计算在女方嫁的条件下全部属性并存的频率p(x|c=1)
p_x_m_0=1
for i in array_3:
    p_x_m_0*=array_3[i]/marry_0 #计算在女方不嫁的条件下全部属性并存的频率p(x|c=0)        
p_c_1_x=p_x_m_1*p_m_1/p_x       #p(c=1|x)=p(x|c=1)p(c=1)/p(x)
p_c_0_x=p_x_m_0*p_m_0/p_x       #p(c=0|x)=p(x|c=0)p(c=0)/p(x)

print("p(c=1|x)= ",p_c_1_x)
print("p(c=0|x)= ",p_c_0_x)

if(p_c_1_x>p_c_0_x):
    print("cause p(c=1|x)>p(c=0|x)\nso she will marry")
if(p_c_1_x<p_c_0_x):
    print("cause p(c=1|x)<p(c=0|x)\nso she won't marry")
else:
    print("she will be diffcult to make decison")


