# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import warnings
import datetime
import os

warnings.filterwarnings('ignore')
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

from sklearn.cluster import MiniBatchKMeans

import math
import sklearn.metrics as sm


def cluster():
    print('minikmeans')
    path = '2021-11-15-20-52-03'
    X_train = np.loadtxt(path + '/X_train.txt', dtype=float)
    X_test = np.loadtxt(path + '/X_test.txt', dtype=float)
    # np.loadtxt(path+'/y_train.txt',y_train)
    y_test = np.loadtxt(path + '/y_test.txt', dtype=float)

    preAll = np.loadtxt(path + '/preAll.txt', dtype=float)
    with open(path + '/ClusterPrediction.txt', 'w') as f:
        for i in range(len(X_test)):
            for j in range(7):
                f.write(str(i + 1) + ' ' + str(j + 1) + ' ' + str(preAll[i, j] + 1))
                f.write('\n')

    with open(path + '/truth_cluster.txt', 'w') as f:
        for i in range(len(X_test)):
            f.write(str(i + 1) + ' ' + str(y_test[i] + 1))
            f.write('\n')

    with open(path + '/ground_truth.txt', 'w') as f:
        f.write('[')
        for i in range(len(X_test)):
            f.write(str(y_test[i]) + ',')
        f.write(']')

    Narr = np.zeros((len(X_test), 2))
    Xitaarr = np.zeros((len(X_test), 3))
    for i in range(len(X_test)):
        for j in range(7):
            if (preAll[i, j] == 0):
                Narr[i, 0] += 1
            if (preAll[i, j] == 1):
                Narr[i, 1] += 1
    # print(Narr)
    for i in range(len(X_test)):
        Xitaarr[i, 0] = Narr[i, 0] / (Narr[i, 0] + Narr[i, 1])
        Xitaarr[i, 1] = Narr[i, 1] / (Narr[i, 0] + Narr[i, 1])
        Xitaarr[i, 2] = (Xitaarr[i, 1] - Xitaarr[i, 0]) / 2
    # print(Xitaarr)
    plt.scatter(Xitaarr[:, 0], Xitaarr[:, 1], c='red', marker='o', label='see')
    # 绘制数据分布图

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()

    nmf = NMF(2)
    preAllnmf = nmf.fit_transform(preAll)
    # print('preAllnmf')
    # print(preAllnmf)

    #
    # 概念层（众包特征聚类）
    #
    concept_estimator = MiniBatchKMeans(n_clusters=2)  # 构造聚类器
    # concept_estimator.fit(Xitaarr)  # 聚类
    concept_estimator.fit(preAll)  # 聚类
    # concept_estimator.fit(preAllnmf)  # 聚类
    conceptLayer_label_pred = concept_estimator.labels_  # 获取聚类标签
    concept_cluster_centers = concept_estimator.cluster_centers_  # 获取聚类标签
    print("cluster_label_pred")
    print(conceptLayer_label_pred)
    print("conceptLayer_cluster_centers")
    print(concept_cluster_centers)
    with open(path + '/cluster_label_pred.txt', 'w') as f:
        f.write('[')
        for i in range(len(X_test)):
            f.write(str(conceptLayer_label_pred[i]) + ',')
        f.write(']')
    x0 = preAll[conceptLayer_label_pred == 0]
    x1 = preAll[conceptLayer_label_pred == 1]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    x_values = concept_cluster_centers[0][0]
    y_values = concept_cluster_centers[0][1]
    x_values2 = concept_cluster_centers[1][0]
    y_values2 = concept_cluster_centers[1][1]

    '''
    scatter() 
    x:横坐标 y:纵坐标 s:点的尺寸
    '''
    # plt.scatter(x_values, y_values, c="pink")
    # plt.scatter(x_values2, y_values2, c="blue")
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()

    f = open(path + "/cluster_label_pred.txt", 'r')
    cluster_label = eval(f.read())

    f = open(path + "/Clusterpred.txt", 'r')
    list = eval(f.read())
    f = open(path + "/ground_truth.txt", 'r')
    real = eval(f.read())
    f.close()

    print('F1:%.6f' % sm.f1_score(y_true=real, y_pred=list))
    print('ACC:%.6f' % sm.accuracy_score(real, list))

    # for i in range(concept_estimator.n_clusters):
    # print(np.where(concept_estimator.labels_ == i)[0])
    # print(Xitaarr[np.where(estimator.labels_ == i)])

    #
    # 物理层（测试集特征聚类）
    #
    physical_estimator = MiniBatchKMeans(n_clusters=2)  # 构造聚类器
    physical_estimator.fit(X_test)  # 聚类
    physical_label_pred = physical_estimator.labels_  # 获取聚类标签
    physical_cluster_centers = physical_estimator.cluster_centers_  # 获取聚类标签
    print("physical_label_pred")
    print(physical_label_pred)
    # print("physical_cluster_centers")
    # print(physical_cluster_centers)

    matrix = np.zeros((2, 2))
    for i in np.where(physical_label_pred == 0)[0]:

        if (real[i] == 0):
            matrix[0, 0] += 1;

        elif (real[i] == 1):
            matrix[0, 1] += 1;
    for i in np.where(physical_label_pred == 1)[0]:

        if (real[i] == 0):
            matrix[1, 0] += 1;

        elif (real[i] == 1):
            matrix[1, 1] += 1;
    if (matrix[0, 0] < matrix[1, 0]):
        for i in range(len(X_test)):
            physical_label_pred[i] = 1 - physical_label_pred[i]
    matrix2 = np.zeros((2, 2))
    for i in np.where(conceptLayer_label_pred == 0)[0]:
        if (real[i] == 0):
            matrix2[0, 0] += 1;

        elif (real[i] == 1):
            matrix2[0, 1] += 1;
    for i in np.where(conceptLayer_label_pred == 1)[0]:
        if (real[i] == 0):
            matrix2[1, 0] += 1;

        elif (real[i] == 1):
            matrix2[1, 1] += 1;
    if (matrix2[0, 0] < matrix2[1, 0]):
        for i in range(len(X_test)):
            conceptLayer_label_pred[i] = 1 - conceptLayer_label_pred[i]

    x0 = X_test[physical_label_pred == 0]
    x1 = X_test[physical_label_pred == 1]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    # plt.show()

    #
    Dp0 = 0
    for i in np.where(physical_estimator.labels_ == 0)[0]:
        for j in np.where(physical_estimator.labels_ == 0)[0]:
            Dp0 += np.linalg.norm(X_test[i] - X_test[j])
    Dp0 /= len(np.where(physical_estimator.labels_ == 0)[0])
    Dp0 /= len(np.where(physical_estimator.labels_ == 0)[0])
    print('Dp0:' + str(Dp0))
    Dp1 = 0
    for i in np.where(physical_estimator.labels_ == 1)[0]:
        for j in np.where(physical_estimator.labels_ == 1)[0]:
            Dp1 += np.linalg.norm(X_test[i] - X_test[j])
    Dp1 /= len(np.where(physical_estimator.labels_ == 1)[0])
    Dp1 /= len(np.where(physical_estimator.labels_ == 1)[0])
    print('Dp1:' + str(Dp1))
    # Dp0 = 40.33791500257799
    # Dp1 = 16.66171958208334

    UPArr = np.zeros((len(X_test), 1))
    for i in np.where(physical_estimator.labels_ == 0)[0]:
        di00 = np.linalg.norm(X_test[i] - physical_cluster_centers[0]) / Dp0
        di01 = np.linalg.norm(X_test[i] - physical_cluster_centers[1]) / Dp1
        UPArr[i] = 0 - (di00 / (di00 + di01) * math.log10(di00 / (di00 + di01)) + di01 / (di00 + di01) * math.log10(
            di01 / (di00 + di01)))
        # print(di00)
        # print(di01)
    for i in np.where(physical_estimator.labels_ == 1)[0]:
        di10 = np.linalg.norm(X_test[i] - physical_cluster_centers[0]) / Dp0
        di11 = np.linalg.norm(X_test[i] - physical_cluster_centers[1]) / Dp1
        UPArr[i] = 0 - (di10 / (di10 + di11) * math.log10(di10 / (di10 + di11)) + di11 / (di10 + di11) * math.log10(
            di11 / (di10 + di11)))
        # print(di10)
        # print(di11)
    UPT = 100000
    PT = 0
    for index, element in enumerate(UPArr):
        # print(i, element)
        smallpart = np.where(UPArr < element)
        bigpart = np.where(UPArr > element)
        small = 0
        big = 0
        left = 0
        right = 0
        for i in smallpart[0]:
            small += UPArr[i]
        if (len(smallpart[0]) > 0):
            small /= len(smallpart[0])
        for i in bigpart[0]:
            big += UPArr[i]
        if (len(bigpart[0]) > 0):
            big /= len(bigpart[0])
        for i in range(0, index):
            left += math.fabs(UPArr[i] - small)
        for i in range(index, len(UPArr)):
            right += math.fabs(UPArr[i] - big)

        if ((left + right) < UPT):
            # print(left + right)
            UPT = (left + right)
            PT = index
    print('PT:{},{}'.format(str(PT), str(UPT)))
    # T = 14784

    ####################################

    Dc0 = 0

    for i in np.where(concept_estimator.labels_ == 0)[0]:
        for j in np.where(concept_estimator.labels_ == 0)[0]:
            Dc0 += np.linalg.norm(preAll[i] - preAll[j])
    Dc0 /= len(np.where(concept_estimator.labels_ == 0)[0])
    Dc0 /= len(np.where(concept_estimator.labels_ == 0)[0])
    print('Dc0:' + str(Dc0))
    Dc1 = 0
    # print('here')
    # print(np.where(concept_estimator.labels_ == 1)[0])
    for i in np.where(concept_estimator.labels_ == 1)[0]:
        for j in np.where(concept_estimator.labels_ == 1)[0]:
            Dc1 += np.linalg.norm(preAll[i] - preAll[j])
    Dc1 /= len(np.where(concept_estimator.labels_ == 1)[0])
    Dc1 /= len(np.where(concept_estimator.labels_ == 1)[0])
    print('Dc1:' + str(Dc1))
    # Dc0 = 0.29344444696489586
    # Dc1 = 0.2643519896961697
    UCArr = np.zeros((len(X_test), 1))
    for i in np.where(concept_estimator.labels_ == 0)[0]:
        dci00 = np.linalg.norm(preAll[i] - concept_cluster_centers[0]) / Dc0
        dci01 = np.linalg.norm(preAll[i] - concept_cluster_centers[1]) / Dc1
        UCArr[i] = 0 - (dci00 / (dci00 + dci01) * math.log10(dci00 / (dci00 + dci01)) + dci01 / (
                    dci00 + dci01) * math.log10(dci01 / (dci00 + dci01)))
        # print(di00)
        # print(di01)
    for i in np.where(concept_estimator.labels_ == 1)[0]:
        dci10 = np.linalg.norm(preAll[i] - concept_cluster_centers[0]) / Dc0
        dci11 = np.linalg.norm(preAll[i] - concept_cluster_centers[1]) / Dc1
        UCArr[i] = 0 - (dci10 / (dci10 + dci11) * math.log10(dci10 / (dci10 + dci11)) + dci11 / (
                    dci10 + dci11) * math.log10(dci11 / (dci10 + dci11)))
        # print(di10)
        # print(di11)
    UCT = 100000
    CT = 0
    for index, element in enumerate(UCArr):
        # print(i, element)
        smallpart = np.where(UCArr < element)
        bigpart = np.where(UCArr > element)
        small = 0
        big = 0
        left = 0
        right = 0
        for i in smallpart[0]:
            small += UCArr[i]
        if (len(smallpart[0]) > 0):
            small /= len(smallpart[0])
        for i in bigpart[0]:
            big += UCArr[i]
        if (len(bigpart[0]) > 0):
            big /= len(bigpart[0])
        for i in range(0, index):
            left += math.fabs(UCArr[i] - small)
        for i in range(index, len(UCArr)):
            right += math.fabs(UCArr[i] - big)
        # print(left + right)
        if ((left + right) < UCT):
            UCT = (left + right)
            CT = index
    print('CT:{},{}'.format(str(CT), str(UCT)))
    # CT = 2
    np.savetxt(path + '/physicalU.txt', UPArr)
    np.savetxt(path + '/conceptU.txt', UCArr)
    # PT = 5
    # CT = 10
    UPArr = np.loadtxt(path + '/physicalU.txt', dtype=float)
    UCArr = np.loadtxt(path + '/conceptU.txt', dtype=float)
    UCArrSort = np.loadtxt(path + '/conceptU.txt', dtype=float)
    UCArrSort.sort()
    UP = UPArr[PT]
    UC = UCArr[CT]

    UPBig = np.where(UPArr < UP)[0]
    UCSmall = np.where(UCArr > UC)[0]

    uncertain = []
    for i in UPBig:
        for j in UCSmall:
            if (i == j):
                uncertain.append(j)
    print("LEN uncertain：{}".format(len(uncertain)))
    #
    # UCSmall = []
    # count = len(UPBig)
    # # for index, element in enumerate(UCArr):
    # #     if (element < UCArrSort[count]):
    # #         UCSmall.append(index)
    # # print(UCSmall)
    correct = 0
    bubian = 0
    wrong = 0
    for i in uncertain:
        if ((conceptLayer_label_pred[i] == real[i]) & (real[i] != list[i])):
            correct += 1
        if ((conceptLayer_label_pred[i] == list[i]) & (real[i] != list[i])):
            bubian += 1
        if ((conceptLayer_label_pred[i] != real[i]) & (real[i] == list[i])):
            wrong += 1
    print("correct,bubian,wrong:{},{},{}".format(correct, bubian, wrong))
    for i in uncertain:
        list[i] = conceptLayer_label_pred[i]
    print('修正后')
    print('F1:%.6f' % sm.f1_score(y_true=real, y_pred=list))
    print('ACC:%.6f' % sm.accuracy_score(real, list))
    num2 = 0
    for i in range(len(X_test)):
        if (real[i] == list[i]):
            num2 += 1
    #print("num2:{}".format(num2))

    a = 0
    b = 0
    c = 0
    d = 0
    a1 = 0
    b1 = 0
    c1 = 0
    d1 = 0
    print(1111)
    #print(conceptLayer_label_pred)
    for i in range(len(X_test)):
        # for i in range(len(uncertain)):
        # list[i] = physical_label_pred[i]
        real[i] = int(real[i])
        physical_label_pred[i] = int(physical_label_pred[i])
        list[i] = int(list[i])
        #conceptLayer_label_pred[i] = 1 - conceptLayer_label_pred[i]
        if ((physical_label_pred[i] == conceptLayer_label_pred[i]) & (real[i] == conceptLayer_label_pred[i])):
            a += 1
        if ((physical_label_pred[i] == real[i]) & (real[i] != conceptLayer_label_pred[i])):
            b += 1
        if ((conceptLayer_label_pred[i] == real[i]) & (real[i] != physical_label_pred[i])):
            c += 1
        if ((conceptLayer_label_pred[i] == physical_label_pred[i]) & (real[i] != physical_label_pred[i])):
            d += 1
        # if (((1 - conceptLayer_label_pred[i]) == physical_label_pred[i]) & (real[i] == (1-conceptLayer_label_pred[i]))):
        #     a1 += 1
        # if ((physical_label_pred[i] == real[i]) & (real[i] != (1 - conceptLayer_label_pred[i]))):
        #     b1 += 1
        # if (((1 - conceptLayer_label_pred[i]) == real[i]) & (real[i] != physical_label_pred[i])):
        #     c1 += 1
        # if (((1 - conceptLayer_label_pred[i]) == physical_label_pred[i]) & (real[i] != physical_label_pred[i])):
        #     d1 += 1
        #list[i] = conceptLayer_label_pred[i]
    print(a, b, c, d)
    print(a1, b1, c1, d1)

    print('F1:%.6f' % sm.f1_score(y_true=real, y_pred=list))
    print('ACC:%.6f' % sm.accuracy_score(real, list))
    # for i in range(physical_estimator.n_clusters):
    # print('np.where(physical_estimator.labels_ == i)[0]')
    # print(np.where(physical_estimator.labels_ == i)[0])
    # print(Xitaarr[np.where(estimator.labels_ == i)])
    matrix = np.zeros((2, 2))
    for i in np.where(physical_label_pred == 0)[0]:
        if (real[i] == 0):
            matrix[0, 0] += 1;

        elif (real[i] == 1):
            matrix[0, 1] += 1;
    for i in np.where(physical_label_pred == 1)[0]:
        if (real[i] == 0):
            matrix[1, 0] += 1;

        elif (real[i] == 1):
            matrix[1, 1] += 1;
    print('p matrix')
    print(matrix)

    matrix2 = np.zeros((2, 2))
    for i in np.where(conceptLayer_label_pred == 0)[0]:
        if (real[i] == 0):
            matrix2[0, 0] += 1;

        elif (real[i] == 1):
            matrix2[0, 1] += 1;
    for i in np.where(conceptLayer_label_pred == 1)[0]:
        if (real[i] == 0):
            matrix2[1, 0] += 1;

        elif (real[i] == 1):
            matrix2[1, 1] += 1;
    print('c matrix')
    print(matrix2)
    return physical_label_pred, conceptLayer_label_pred


if __name__ == '__main__':
    cluster()
