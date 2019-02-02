# 参考博客：https://cuijiahua.com/blog/2017/11/ml_1_knn.html
'''
KNneighborsClassifier参数说明：

n_neighbors：默认为5，就是k-NN的k的值，选取最近的k个点。
weights：默认是uniform，参数可以是uniform、distance，也可以是用
         户自己定义的函数。uniform是均等的权重，就说所有的邻近点
         的权重都是相等的。distance是不均等的权重，距离近的点比距
         离远的点的影响大。用户自定义的函数，接收距离的数组，返回一组维数相同的权重。
algorithm：快速k近邻搜索算法，默认参数为auto，可以理解为算法自己
          决定合适的搜索算法。除此之外，用户也可以自己指定搜索算法ball_tree、
          kd_tree、brute方法进行搜索，brute是蛮力搜索，也就是线性扫描，当训练
          集很大时，计算非常耗时。kd_tree，构造kd树存储数据以便对其进行快速检索
          的树形数据结构，kd树也就是数据结构中的二叉树。以中值切分构造的树，每个
          结点是一个超矩形，在维数小于20时效率高。ball tree是为了克服kd树高纬失
          效而发明的，其构造过程是以质心C和半径r分割样本空间，每个节点是一个超球体。
leaf_size：默认是30，这个是构造的kd树和ball树的大小。这个值的设置会影响树构建的
          速度和搜索速度，同样也影响着存储树所需的内存大小。需要根据问题的性质选
          择最优的大小。
metric：用于距离度量，默认度量是minkowski，也就是p=2的欧氏距离(欧几里德度量)。
p：距离度量公式。在上小结，我们使用欧氏距离公式进行距离度量。除此之外，还有
    其他的度量方法，例如曼哈顿距离。这个参数默认为2，也就是默认使用欧式距离公式进
    行距离度量。也可以设置为1，使用曼哈顿距离公式进行距离度量。
metric_params：距离公式的其他关键参数，这个可以不管，使用默认的None即可。
n_jobs：并行处理设置。默认为1，临近点搜索并行工作数。如果为-1，那么CPU的
         所有cores都用于并行工作。

'''

import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

def img2vector(filename):
    # 创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    # 打开文件
    fr = open(filename)
    # 按行读取
    for i in range(32):
        lineStr = fr.readline()
        # 每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    # 返回转换后的1x1024向量
    return returnVect


"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无

Modify:
    2019-02-02
"""


def handwritingClassTest():
    # 测试集的Labels
    hwLabels = []
    # 返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    # 返回文件夹下文件的个数
    m = len(trainingFileList)
    # 初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m, 1024))
    # 从文件名中解析出训练集的类别
    for i in range(m):
        # 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNameStr))
    # 构建kNN分类器
    neigh = kNN(n_neighbors=3, algorithm='auto')
    # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    # 返回testDigits目录下的文件列表
    testFileList = listdir('testDigits')
    # 错误检测计数
    errorCount = 0.0
    # 测试数据的数量
    mTest = len(testFileList)
    # 从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        # 获得文件的名字
        fileNameStr = testFileList[i]
        # 获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        # 获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


"""
函数说明:main函数

Parameters:
    无
Returns:
    无

Modify:
    2019-02-02
"""
if __name__ == '__main__':
    handwritingClassTest()


