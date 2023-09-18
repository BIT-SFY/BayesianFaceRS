import os
import time
import pickle
import numpy as np
from PIL import Image
import scipy.stats as stats
from joblib import dump, load
from sklearn.decomposition import PCA

'''
人脸训练模块
'''
HEIGHT = 320  # 112  # 照片高
WIDTH = 243  # 92  # 照片宽
picSize = WIDTH * HEIGHT  # 照片大小
dataSize = 400  # 数据集大小
lableSize = 40  # 类的数目
perSize = 11  # 10  # 每一类有多少个（张）数据
n_components = 80  # PCA维度

MAX_GENERATION = 20  # 最大迭代次数
N = 20  # 种群规模
GENE = 100  # 基因个数
F = 10  # 变异因子
CR = 0.5  # 交叉概率
MINERROR = 0.01005025  # 1/99.5
BOUND = [0, 1]  # 上下界o
meanFitness = 0  # 当代平均准确率


# ----------------------------------------------------------------------------------
# 数据预处理相关函数
# ----------------------------------------------------------------------------------
def DataSplit(data, label, test_size=0.3):
    train_size = 1 - test_size
    X_train = list()  # 训练集
    y_train = list()  # 训练集标签
    X_test = list()  # 测试集
    y_test = list()  # 测试集标签
    for i in range(0, dataSize, perSize):
        for k in range(0, int(perSize * train_size)):
            X_train.append(data[i + k])
            y_train.append(label[i])
        for k in range(int(perSize * train_size), int(perSize)):
            X_test.append(data[i + k])
            y_test.append(label[i])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def LoadFaceData(rootDir, dataBaseName):
    '''
    读取人脸数据库+PCA降维+数据分割
    :param rootDir: 人脸信息库的位置
    :param dataBaseName:人脸信息库的名称
    :return: 降维后的人脸信息以及对应类别和训练信息库位置
    '''
    global dataSize, lableSize, perSize
    list1 = os.listdir(rootDir)  # 列出文件夹下所有的目录与文件
    # 根据文件夹的树形结构，读取图像
    Y = list()
    X = list()
    for i in range(0, len(list1)):
        path1 = os.path.join(rootDir, list1[i])
        list2 = os.listdir(path1)
        for k in range(0, len(list2)):
            path2 = os.path.join(path1, list2[k])
            raw_image = Image.open(path2).convert('L')
            X.append(list(raw_image.getdata()))
            Y.append(i + 1)
            pass
        pass
    # 确认人脸信息库的基本信息
    lableSize = len(list1)
    dataSize = len(Y)
    perSize = int(dataSize / lableSize)
    # 创建人脸数据库对应的训练信息文件夹
    newDir = './TrainingInfo/' + rootDir[57:]
    isExists = os.path.exists(newDir)
    if not isExists:
        os.makedirs(newDir)
        pass
    # 存储人脸数据和类别为pkl数据
    with open(newDir + '/X.pkl', 'wb') as pickle_file:
        pickle.dump(X, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    with open(newDir + '/Y.pkl', 'wb') as pickle_file:
        pickle.dump(Y, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    # 对人脸数据进行降维
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X)
    X = pca.transform(X)
    # 保存PCA模型信息
    dump(pca, newDir + '/PCA.joblib')  # 保存PCA模型
    # 保存降维后的人脸信息
    with open(newDir + '/PCAX.pkl', 'wb') as pickle_file:
        pickle.dump(X, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    # 保存人脸数据库基本信息：
    DataBaseInfo = {'dataSize': dataSize, 'lableSize': lableSize, 'perSize': perSize, 'dataBaseName': dataBaseName}
    with open(newDir + '/DataBaseInfo.pkl', 'wb') as pickle_file:
        pickle.dump(DataBaseInfo, pickle_file)
    # 返回降维后的人脸信息以及对应类别和训练信息库位置
    return X, Y, newDir


# ----------------------------------------------------------------------------------
# 贝叶斯相关函数
# ----------------------------------------------------------------------------------
def Evaluate(pred, gt):
    '''
    准确率评估函数
    :param pred:预测的类别
    :param gt:实际类别
    :return:准确率
    '''
    correct = 0
    for i in range(len(pred)):
        if pred[i] + 1 == gt[i]:
            correct += 1
    return (correct / len(pred)) * 100  # 预测正确的数目/一共预测的数目,再乘100


def BayesLearn(Xf, label):
    '''
    贝叶斯分类器参数学习
    :param Xf: 待训练数据
    :param label: 类别
    :return: 均值,方差,先验概率
    '''
    ms = []
    sg = []
    for i in range(1, lableSize + 1):  # 对于所有类别
        idx = np.where(label == i)  # 从label中找到属于第i个属性的图片的下标
        class_i = Xf[idx]  # 根据idx得到的下标,获取xf中的对应下标的图片
        mu = np.mean(class_i, axis=0)  # 求当前类别的所有测试集在该属性下的均值
        ms.append(mu)
        su = np.std(class_i, axis=0)  # 求当前类别的标准差
        sg.append(su)
    p = len(class_i) / len(Xf)  # 得到先验概率,这里所有类别的图片所占比重都是相同的所以采用同一个p
    return [np.array(ms), np.array(sg), p]


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


# ----------------------------------------------------------------------------------
# 演化算法相关函数
# ----------------------------------------------------------------------------------
def Initialization():
    '''
    种群初始化
    :return:初始化结果
    '''
    data = np.random.random((N, GENE))
    return data


def mutate(G, mG):
    '''
    变异操作
    :param G: 父代个体集
    :param mG: 变异个体集
    :return:None
    '''
    # 对于每个个体
    for i in range(N):
        r = np.random.randint(1, N, 2)  # 生成两个1~N的整数
        # 变异个体的基使用每次适应度值最好的个体geneBest
        # F过大 无法收敛
        # F过小 早熟
        mG[i, :] = geneBest + F * (G[r[0], :] - G[r[1], :])  # 差分体现在此
        # 边界检测
        for j in range(GENE):
            if mG[i][j] > BOUND[1] or mG[i][j] < BOUND[0]:
                mG[i][j] = np.random.random()
                pass
            pass
        pass
    return None


def crossover(G, mG, cG):
    '''
    交叉操作
    :param G:父代个体集
    :param mG:变异个体集
    :param cG:交叉个体集
    :return:
    '''
    for i in range(N):
        # jRand随机从一个个体中挑选一个基因，其必然传给交叉代
        jRand = np.floor(np.random.random() * GENE)  # floor返回不大于(<=)输入参数的最大整数
        for j in range(GENE):
            # CR调小 有利于往geneBest的方向进化
            if np.random.random() < CR or j == jRand:
                cG[i, j] = mG[i, j]  # 如果小于交叉率，则让这个基因遗传变异代
                pass
            else:
                cG[i, j] = G[i, j]  # 如果大于交叉率，则让这个基因遗传父代
                pass
            pass
        pass
    return None


def select(G, cG):
    '''
    选择操作
    :param G:父代个体集
    :param cG:交叉个体集
    :return: None
    '''
    global geneBest, meanFitness
    fitOld = np.zeros(N)  # 父代适应度集
    fitNew = np.zeros(N)  # 新代适应度集
    # 对于每个个体
    for i in range(N):
        fitOld[i] = CalcFitness(G[i])
        fitNew[i] = CalcFitness(cG[i])
        # 变异交叉后的个体表现更好  CalcFitness的值越小约好
        if fitOld[i] >= fitNew[i]:
            # 如果父代个体集适应度更高[这里越高越不好,懒得修改了],则选择交叉代进行遗传,否则继续遗传父代
            G[i, :] = cG[i, :]
            fitOld[i] = fitNew[i]
            # 是否比geneBest适应度值更小
            if fitNew[i] < CalcFitness(geneBest):
                geneBest = np.copy(cG[i, :])
                pass
            pass
        pass
    # 计算种群的平均准确率
    meanFitness = (np.sum(1 / fitOld)) / N
    return None


def CalcFitness(weight):
    '''
    适应度函数
    :param weight:权重
    :return:适应度
    '''
    y_pre = BayesianClassifier(X_test, ms, sg, p, weight)  # 根据参数构成权重贝叶斯分类器,并进行预测
    accuracy_result = Evaluate(y_pre, y_test)  # 根绝分结果和真实类别进行对比得到准确率
    return 1 / accuracy_result  # 对准确率 /1 得到适应度


# ----------------------------------------------------------------------------------
# 训练调度函数
# ----------------------------------------------------------------------------------
def Fit(dirPath, n_components_, MAX_GENERATION_, N_, F_, CR_, test_size_):
    '''
    根绝获取的人脸信息库信息和参数进行分类器训练
    :param dirPath:人脸信息库的位置
    :param n_components_:PCA特征向量维数
    :param MAX_GENERATION_:最大演化代数
    :param N_:种群大小
    :param F_:变异因子
    :param CR_:交叉概率
    :param test_size_:测试集比例
    :return:分类器训练后得到的部分信息，如耗时准确率等
    '''
    global X_test, y_test, ms, sg, p  # 设置为全局变量方便调用修改
    global n_components, MAX_GENERATION, N, F, CR, GENE, geneBest, test_size
    # 对参数赋值
    n_components = n_components_
    MAX_GENERATION = MAX_GENERATION_
    N = N_
    F = F_
    CR = CR_
    test_size = test_size_
    # 获取程序运行开始时间
    start = time.perf_counter()
    # 根据人脸信息库的地址，地址信息57个字符之后表示的就是人脸信息库的名称
    'C:/Users/BIT_0306/PycharmProjects/BayesianFaceRS/FaceLib/ORL'
    '123456789012345678901234567890123456789012345678901234567890'
    X, Y, newDir = LoadFaceData(dirPath, dirPath[57:])
    # 对人脸数据进行分割
    X_train, y_train, X_test, y_test = DataSplit(X, Y, test_size)
    # 学习均值方差先验概率
    ms, sg, p = BayesLearn(X_train, y_train)
    # 存储均值方差先验概率
    with open(newDir + '/ms.pkl', 'wb') as pickle_file:
        pickle.dump(ms, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    with open(newDir + '/sg.pkl', 'wb') as pickle_file:
        pickle.dump(sg, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    with open(newDir + '/p.txt', 'w') as fp:
        fp.write(str(p))
        fp.close()
        pass
    # 保存分类器训练的参数设置信息
    DEBayesInfo = {'n_components': n_components, 'MAX_GENERATION': MAX_GENERATION, 'N': N, 'F': F, 'CR': CR,
                   'test_size': test_size}
    with open(newDir + '/DEBayesInfo.pkl', 'wb') as pickle_file:
        pickle.dump(DEBayesInfo, pickle_file)
    # 进入差分演化部分
    GENE = n_components  # 基因长度就是特征值个数
    numGeneration = 0  # 当前代数
    Generation = Initialization()  # 初始化当前种群
    mutateGeneration = np.zeros((N, GENE))  # 设置变异代
    crossoverGeneration = np.zeros((N, GENE))  # 设置交叉代
    geneBest = np.zeros(GENE)  # 最佳个体
    fitOld = np.zeros(N)  # 父代适应度集
    # 找出随机生成的最好的种子
    for i in range(N):
        fitOld[i] = CalcFitness(Generation[i])
    # 找出最好的个体 使适应度值最小
    geneBest = Generation[np.argmin(fitOld)]  # 当代最佳个体
    # 打开日志文件,将分类器相关参数写入
    fobj = open(newDir + '/TrainingLog.txt', 'w', encoding='utf-8')
    fobj.write( '-' * 38 + '\nPCA维度: {}   测试集比例: {}\nMAX_G: {}  NP: {}  F: {}  Cr: {}  \n'.format(n_components,test_size, MAX_GENERATION, N, F,CR) + '-' * 38 + '\n')
    # 进入迭代
    # 如果最棒的个体已经超过阈值，则结束进化
    while numGeneration < MAX_GENERATION:
        # 对种群最优个体进行判断,准确率阈值则停止演化
        if CalcFitness(geneBest) < MINERROR:
            print(numGeneration)
            break
        # 变异mutate
        mutate(Generation, mutateGeneration)
        # 交叉crossover
        crossover(Generation, mutateGeneration, crossoverGeneration)
        # 选择select
        select(Generation, crossoverGeneration)
        # 代数+1
        numGeneration += 1
        # 计算最佳个体准确率
        ans = CalcFitness(geneBest)
        # 将迭代过程写入日志
        fobj.write('第{}代 最优准确率:{} 平均准确率:{}\n'.format(numGeneration, 1 / ans, meanFitness))
        print('第{}代 最优准确率:{} 平均准确率:{}'.format(numGeneration, 1 / ans, meanFitness))
        # 保存全部权重与最佳权重
        with open(newDir + '/weight.pkl', 'wb') as pickle_file:
            pickle.dump(Generation, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(newDir + '/Xbest.pkl', 'wb') as pickle_file:
            pickle.dump(geneBest, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    end = time.perf_counter()  # 获取程序结束时间
    fobj.write('耗时: {}s\n'.format(round(end - start, 2)))
    fobj.close()
    return dataSize, lableSize, str(end - start), 1 / CalcFitness(geneBest)


if __name__ == '__main__':
    pass
