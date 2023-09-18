import time
import pickle
import numpy as np
from PIL import Image
import scipy.stats as stats
from joblib import dump, load


def BayesianClassifier(x, ms, gs, p, weight):
    '''
    根据参数构成权重贝叶斯分类器,并进行分类
    :param x:待预测数据
    :param ms:均值
    :param gs:方差
    :param p:先验概率
    :param weight:权重
    :return: 分类结果
    '''
    prob = np.zeros([len(x), lableSize])
    for i in range(0, lableSize):
        pu = stats.norm.pdf(x, ms[i, :], gs[i, :])  # 根据当前类别对应的某个属性的均值和标准差,来构建高斯分布,并根绝待分类数据在此属性的值,得到概率
        pu = pu ** weight  # 对参数进行加权
        pp = np.prod(pu, axis=1)  # 连乘
        prob[:, i] = pp * p  # 乘以先验概率
    cls = np.argmax(prob, axis=1)  # 返回预测结果中最大元素的索引值
    return cls  # 返回识别结果


def Prediction(filename, dataBaseName):
    '''
    构建权重贝叶斯分类器并进行分类
    :param filename:文件路径
    :param dataBaseName:对应数据库名称
    :return: 预测结果以及耗时
    '''
    global lableSize, dataSize, perSize
    # 读取训练信息库中的数据库信息
    with open('./TrainingInfo/' + dataBaseName + '/DataBaseInfo.pkl', 'rb') as pickle_load:
        DataBaseInfo = pickle.load(pickle_load)
        pass
    # 将数据库信息进行保存
    lableSize = DataBaseInfo['lableSize']
    dataSize = DataBaseInfo['dataSize']
    perSize = DataBaseInfo['perSize']
    dataBaseNameTarget = DataBaseInfo['dataBaseName']
    # 获取程序运行开始时间
    start = time.perf_counter()
    # 读取PCA模型
    pca = load('./TrainingInfo/' + dataBaseName + '/PCA.joblib')
    # 读取待分类图片
    xPre = []
    # 打开图片(灰度)
    raw_image = Image.open(filename).convert('L')
    xPre.append(list(raw_image.getdata()))
    # 利用读取到的分类器模型对图片降维
    pcax = pca.transform(xPre)
    # 读取均值,方差,权重,先验概率
    with open('./TrainingInfo/' + dataBaseName + '/ms.pkl', 'rb') as pickle_load:  # 读ms
        ms = pickle.load(pickle_load)
    with open('./TrainingInfo/' + dataBaseName + '/sg.pkl', 'rb') as pickle_load:  # 读sg
        sg = pickle.load(pickle_load)
    with open('./TrainingInfo/' + dataBaseName + '/Xbest.pkl', 'rb') as pickle_load:  # 读最佳权重
        weight = pickle.load(pickle_load)
    p = np.loadtxt('./TrainingInfo/' + dataBaseName + '/p.txt', delimiter=' ')
    # 利用得到的参数信息构建权重贝叶斯分类器并进行分类
    pred_cls3 = BayesianClassifier(pcax, ms, sg, p, weight)
    # 获得结束时间
    end = time.perf_counter()
    return pred_cls3[0] + 1, end - start  # pred_cls3得到的是标签，但是他得加1才能对应上规定的标签，这里因为check的下标本来就是-1的，所以不用动了


if __name__ == '__main__':
    pass
