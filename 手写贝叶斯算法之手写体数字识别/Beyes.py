import numpy as np
import os
import operator
from PIL import Image
import time
class Bayes(object):
    def __init__(self):
        self.length = -1
        self.labelweight = dict() # 权重：类别占总类别总数的比例
        self.vectorcount = dict() # 训练数据向量

    def fit(self, dataSet:list, labels:list):
        '''
        训练数据
        
        Args:
            dataSet:训练数据
            labels:训练数据对应的类别
        Return: None
        '''
        print("-"*40,"开始训练数据","-"*35)
        if(len(dataSet) != len(labels)):
            raise ValueError("测试组与类别组的长度不一样")
        self.length = len(dataSet[0])  # 测试数据特征值的长度
        labelsnum = len(labels)  #  类别所有的数量
        norlabels = set(labels)  #  不重复类别的数量
        for item in norlabels:
            thislabel = item
            self.labelweight[thislabel] = labels.count(thislabel)/labelsnum  # 权重:每个类别的占比
        for vector, label in zip(dataSet, labels):
            if(label not in self.vectorcount):
                self.vectorcount[label] = []
            self.vectorcount[label].append(vector)
        for i in range(0,10):            
            print("类别%s所占比例%s"%(i,self.labelweight[i]))
        print("-"*40,"数据训练结束","-"*35)
        print("")
        return self

    def btest(self, TestData, labelSet):
        '''
        Beyes算法测试

        Args:
            TestData:测试数据
            labelSet:数据类别
        Return: None
        '''
        if(self.length == -1):
            raise ValueError("没有进行训练，先训练")
        # 计算testdata分别为各个类别的概率
        lbDict = dict()
        for thislb in labelSet:
            p = 1
            alllabel = self.labelweight[thislb]   # 类别的权重
            allvector = self.vectorcount[thislb]  # 类别的向量
            vnum = len(allvector)  # 类别的训练样本个数
            allvector = np.array(allvector).T # 数组转置
            for index in range(0, len(TestData)): 
                vector = list(allvector[index])
                p *= vector.count(TestData[index])/vnum   # 测试样本与训练集的特征值的匹配数
            lbDict[thislb] = p * alllabel  #  算出概率
        thislabel = sorted(lbDict, key=lambda x:lbDict[x], reverse=True)[0]  # 取出概率最大的预测类别
        return thislabel

    def testdata(self):
        '''
        测试所有数据
    
        Return: None
        '''
        
        print("-"*40,"开始测试数据","-"*35)
        start = time.time() # 开始时间
        #trainarr, labels = traindata()
        errors = [] # 所有出错的数字类别
        errors_num = [] #  每个数字类别出错的次数
        errors_rating = [] # 每个数字类别的错误率
        errors_maxdigital = [] # 每个数字类别错判次数最多的数字
        maxdigital = [] # 每个数字类别所有错判的数字
        testlist = os.listdir("test_digital_txt/" ) # 获取手写体图片名
        num = len(testlist)
        print("文件存储相对路径：test_digital_txt/") # 转换数据目录
        print("所有测试txt文本数量:",num) #  num：所有txt文本数量 
        # 长度784列，每一行存储一个文件
        # 以一个数组存储所有训练数据，行：文件总数，列：784
        testarr = np.zeros((num,784))
        ten = 0
        print("")
        print("-"*40,"类别",ten,"-"*40)
        print("")                
        for i in range(0,num):
            thisfname = testlist[i]
            thisdata = datatoarray("test_digital_txt/%s"  %(thisfname))
            labelsall = [0,1,2,3,4,5,6,7,8,9]
            rknn = self.btest(thisdata,labelsall)
            
            if str(rknn) != thisfname.split("_")[0]: # 预测错误
                print("%s样本预测出错：" %(thisfname),rknn)
                errors.append(int(thisfname.split("_")[0])) # 记录出错数字类别
                maxdigital.append(rknn) # 判错数字
            
            if (i+1)%220 == 0:  # 某个数字类别预测结束
                errors_num.append(errors.count(ten)) # 计算某数字类别的出错次数
                errors_rating.append(errors_num[ten]/2200) # 计算某数字类别出错率
                if maxdigital: # 某数字类别判错次数最大的数字
                    errors_maxdigital.append(max(maxdigital, key=maxdigital.count))
                else:
                    errors_maxdigital.append("None") # 没有，存入None
                print("数字%s出错情况：出错次数%s、出错率%.1f%s、错判次数最多的数字%s" %(ten,errors_num[ten],(errors_rating[ten]*100),"%",errors_maxdigital[ten]))
                maxdigital = [] # 清空，记录下一个数字类别
                ten += 1
                
                print("")
                print("-"*40,"类别",ten,"-"*40)
                print("")   
        accuracy = (2200 - len(errors)) # 预测正确的所有次数
        accuracy_rating = accuracy/2200  # 正确率
        elapsed = (time.time() - start)  # 开始时间与结束时间之差
        print("-"*40,"数据测试结束","-"*35)
        print("准确率：%.1f%s、正确次数：%s" %((accuracy_rating*100),"%",accuracy))
        for k in range(0,10):
            print("数字%s出错情况：出错次数%s、出错率%.1f%s、错判次数最多的数字%s" %(k,errors_num[k],(errors_rating[k]*100),"%",errors_maxdigital[k]))
        print("程序运行消耗时间：%.1f分" %(elapsed/60))

            #print(thisfname,":",rknn)
                
def pictureTotxt(data_name): 
    '''
    处理bmp图片转为txt文本
    
    Args:
      data_name: 数据类型名称
    Return: None
    '''
    if not os.path.exists("%s_digital_txt" %(data_name)): # 创建txt文本文件夹
        os.mkdir("%s_digital_txt" %(data_name))
    lists = os.listdir("%s_digital_pictures" %(data_name)) # 获取所有手写体图片名称
    for list in lists: # 图片转换文本
        im = Image.open("%s_digital_pictures/%s" % (data_name,list)) # 打开手写体图片
        fh = open("%s_digital_txt/%s.txt" % (data_name,list.split(".")[0]),"a") # 打开txt文本
        width = im.size[0]  # 图片宽
        height = im.size[1]  # 图片高
        for k in range(0,width):  #  存入txt文本
            for j in range(0,height):
                cl = im.getpixel((k,j))  # bmp图片的像素只有一个数
                if(cl == 0):  # 0：黑色
                    fh.write("0") # 黑色为0
                else:
                    fh.write("1") # 白色为1
            fh.write("\n")        
        fh.close() # 关闭文件
        
def datatoarray(fname):
    '''
    加载txt文本数据
    
    Args:
      fname: 文本名称
    Return: None
    '''
    arr = []
    fh = open(fname) # 打开txt文本
    for i in range(0,28):
        thisline = fh.readline() # 读取文本内容
        for j in range(0,28):
            arr.append(int(thisline[j]))
    fh.close()
    return arr

def seplabel(fname):
    '''
    获取文本类别
    
    Args:
      fname: 文本名称、名称格式例子：0_0.txt
    Return: None
    '''
    filestr = fname.split(".")[0]
    label = int(filestr.split("_")[0]) 
    return label
    
def traindata():
    '''
    加载训练数据

    Return: None
    '''
    print("-"*40,"开始加载数据","-"*35)
    labels = []
    trainfile = os.listdir("train_digital_txt/") # 获取手写体图片名
    num = len(trainfile)
    # 长度784列，每一行存储一个文本
    trainarr = np.zeros((num ,784)) # 数组存储训练数据，行：文件总数，列：784=28*28    
    print("文件存储相对路径：train_digital_pictures/") # 转换数据目录
    print("所有训练txt文本数量:",num) #  num：所有txt文本数量
    for i in range(0,num):
        thisfname = trainfile[i]
        thislabel = seplabel(thisfname)
        labels.append(thislabel)  # 记录txt文本对应的数字编号
        trainarr[i,:] = datatoarray("train_digital_txt/%s"  %(thisfname))# 加载txt文本数据    
    num = 0
    for arrs in trainarr: # 统计有用数据数量 
        for arr in arrs:
            if arr == 1:
                num += 1
                break
    print("统计有用数据数量 :",num)
    print("-"*40,"结束加载数据","-"*35)
    print("")
    return trainarr, labels

def run():
    bys = Bayes()
    # 训练数据
    train_data, labels = traindata()
    bys.fit(train_data,labels)
    # 测试
    bys.testdata()

def test_one(trainarr,labels,testfile):
    '''
    抽某一个文件进行测试
    
    Args:
      trainarr: 训练集
      labels: 训练集类别
      testfile: 文本名称
    Return: None
    '''
    labelsall = [0,1,2,3,4,5,6,7,8,9]
    testarr=datatoarray("test_digital_txt/" + testfile)  # 获取测试样本
    bys.fit(train_data,labels)
    rst = bys.btest(thisdata,labelsall)
    print("%s的测试结果:%s"%(testfile,rst))
#test_one()
