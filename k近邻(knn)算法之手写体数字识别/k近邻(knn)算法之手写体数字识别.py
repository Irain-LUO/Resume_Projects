#-*-coding:utf-8-*-
import numpy as np
import operator
import os
from PIL import Image
import time
# 训练集合和试集图片处理
# 先将所有图片转为固定框架，比如：28*28，然后转为文本
from PIL import Image
import os
def pictureTotxt(data_name): #  data_name:数据类型名称
    if not os.path.exists("%s_digital_txt" %(data_name)):
        os.mkdir("%s_digital_txt" %(data_name))
    lists = os.listdir("%s_digital_pictures" %(data_name)) # 获取手写体图片名
    for list in lists: # 遍历所有图片
        # 打开手写体图片
        im = Image.open("%s_digital_pictures/%s" % (data_name,list))
        # 打开txt文本
        fh = open("%s_digital_txt/%s.txt" % (data_name,list.split(".")[0]),"a")
        width = im.size[0]  # 图片宽
        height = im.size[1]  #图片高
        for k in range(0,width):  #  存入txt文本
            for j in range(0,height):
                cl = im.getpixel((k,j))  # bmp图片的像素只有一个数
                if(cl == 0):  # 0：黑色
                    # 黑色为0
                    fh.write("0")
                else:
                    # 白色为1
                    fh.write("1")
            fh.write("\n")        
        fh.close()


# 获取文件夹所有文件
# 从列方向扩张  tile(字典,(列数,1))
# 从行方向扩张  tile(字典,(行数,0))
def knn(k, testdata, traindata, labels):
    traindatasize = traindata.shape[0]  #  扩展行数，测试集行数与训练集行数一样
    # print(traindatasize)
    dif = np.tile(testdata, (traindatasize, 1)) - traindata# why 行数一致，是否确保所有训练集都与测试集计算
    # print(len(dif))
    sqdif = dif**2 #  差值平方
    sumsqdif = sqdif.sum(axis=1) #  平方求和，求出距离的平方
    distance = sumsqdif**0.5 #  开方，求出距离
    sortdistance = distance.argsort() # 距离升序
    sortdistance
    # print(len(sortdistance))
    # print(sortdistance)
    count = {}
    for i in range(0,k):
        # print(sortdistance[i])
        vote = labels[sortdistance[i]]
        count[vote] = count.get(vote,0) + 1  #  类别次数+1
    sortcount = sorted(count.items(),key=operator.itemgetter(1),reverse=True) # reverse降序,不懂
    return sortcount[0][0]  # 返回分类结果


# 加载txt文本数据
def datatoarray(fname):
    arr = []
    fh = open(fname)
    for i in range(0,28):
        thisline = fh.readline()
        for j in range(0,28):
            arr.append(int(thisline[j]))
    fh.close()
    return arr

# 建立取文件的类别前缀函数
def seplabel(fname):
    filestr = fname.split(".")[0]
    label = int(filestr.split("_")[0])
    return label
    
# 建立训练数据
def traindata():
    labels = []
    trainfile = os.listdir("train_digital_txt/") # 获取手写体图片名
    num = len(trainfile)
    # 长度784列，每一行存储一个文件
    # 以一个数组存储所有训练数据，行：文件总数，列：784
    trainarr = np.zeros((num ,784))
    print("相对路径：train_digital_pictures/") # 转换数据目录
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
    return trainarr, labels


# 用测试数据调用knn算法取测试，看是否能够准确识别
def testdata():
    start = time.time()
    errors = [] # 所有出错的数字类别
    errors_num = [] #  每个数字类别出错的次数
    errors_rating = [] # 每个数字类别的错误率
    errors_maxdigital = [] # 每个数字类别错判次数最多的数字
    maxdigital = [] # 每个数字类别所有错判的数字
    trainarr, labels = traindata()
    print("类别：",set(labels))
    print("相对路径：test_digital_txt/") # 转换数据目录
    testlist = os.listdir("test_digital_txt/") # 获取测试样本名    
    num = len(testlist)
    print("所有测试txt文本数量:",num) #  num：所有txt文本数量 
    ten = 0
    for i in range(0,num):
        thisfname = testlist[i]
        testarr = datatoarray("test_digital_txt/%s"  %(thisfname))  # 获取测试样本
        rknn = knn(21,testarr,trainarr,labels)  # knn算法进行测试
        
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
            print("数字%s的出错次数\出错率\错判次数最多的数字:%s\%s\%s" %(ten,errors_num[ten],errors_rating[ten],errors_maxdigital[ten]))
            maxdigital = [] # 清空，记录下一个数字类别
            ten += 1
    accuracy = (2200 - len(errors)) # 预测正确的所有次数
    accuracy_rating = accuracy/2200  # 正确率
    elapsed = (time.time() - start)
    print("准确率：%s、正确次数：%s" %(accuracy_rating,accuracy))
    for k in range(0,10):
        print("数字%s出错情况：出错次数%s、出错率%s、错判次数最多的数字%s" %(k,errors_num[k],errors_rating[k],errors_maxdigital[k]))
    print("程序运行消耗时间：%.3f" %elapsed)

testdata()
