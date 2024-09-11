import numpy as np
import pandas as pd
from Other_fuc import find_ture_labs

# from MEL_ECA_lab import N,D,K

#################        计算context_based_distance              ###################
def probability_entropy(dataset, feature_d):
    "用于求特征d下，所有特征值的概率"
    feature_data = dataset[:, feature_d]
    a, counts = np.unique(feature_data, return_counts=True)  # a：有哪些值 ；counts：这些值在数据集中出现的次数
    probabilities = counts / len(feature_data)  # 计算概率分布
    return a, probabilities


def entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def joint_entropy(dataset):
    " 计算每个特征的熵"
    N, D = np.shape(dataset)
    feature_entropies = []  # 数组用于记录每个特征的熵
    for d in range(D):  # 计算每一个特征的熵
        a, probabilities = probability_entropy(dataset, d)  # 计算每个特征值与其对应的概率
        feature_entropy = entropy(probabilities)  # 计算  求和（P*log2（P））
        feature_entropies.append(feature_entropy)  # 将值放入每个数组中
    return feature_entropies


def conditional_entropy(dataset, feature_x, feature_y):
    "已知特征Y与特征X，求H（X|Y）；先将特征Y分解获得其每个特征值y和概率p；在针对每个特征值y，求y条件下x的条件概率"
    feature_y_value, probabilities_y = probability_entropy(dataset, feature_y)  # 求Y特征的特征值，与对应的概率
    conditional_entropy = 0  # 初始化 H（X|Y）的值
    for value_y in feature_y_value:  # 计算Y的每个特征值下，与每个X的特征值的条件概率
        index_row_y = np.where(dataset[:, feature_y] == value_y)[0]  # y其中一个特征值value_y在原始数据集的行位置
        subset_data = dataset[index_row_y, :]  # 提取对应行位置的数据，构成子集

        a, probabilities_x_y = probability_entropy(subset_data, feature_x)  # P( feature_x | value_y )->获得y条件下X的条件概率
        subset_conditional_entropy = entropy(probabilities_x_y)  # P * log2(P)->计算每个x下的条件熵，一维矩阵
        conditional_entropy += probabilities_y[feature_y_value == value_y][0] * subset_conditional_entropy

    return conditional_entropy


def conditional_entropy_matrix(dataset):
    N, D = np.shape(dataset)
    conditional_entropy_matrix = np.zeros([D, D])
    for i in range(D):  # i:X        其中要求的是H（X|Y）
        for j in range(D):  # j:Y
            if i == j:  # 表示本特征的熵
                conditional_entropy_matrix[i, j] = 0
            else:
                conditional_entropy_matrix[i, j] = conditional_entropy(dataset, i, j)
    return conditional_entropy_matrix


def information_gain(dataset):
    "计算信息增益矩阵IG"
    N, D = np.shape(dataset)
    conditional_entropy = conditional_entropy_matrix(dataset)  # 条件熵矩阵
    information_gain_matrix = np.zeros([D, D])
    feature_entropies = joint_entropy(dataset)  # 特征的熵
    for i in range(D):
        for j in range(D):
            if i == j:
                information_gain_matrix[i, j] = 0
            else:
                information_gain_matrix[i, j] = (feature_entropies[i] - conditional_entropy[i, j])
    return information_gain_matrix


def symmetrical_uncertaninty(dataset):
    "计算对称不确定性SU"
    N, D = np.shape(dataset)
    feature_entropies = joint_entropy(dataset)  # 每个特征的熵
    information_gain_matrix = information_gain(dataset)  # 信息增益
    symmetrical_uncertaninty_matrix = np.zeros([D, D])  # 空SU矩阵
    for i in range(D):
        for j in range(D):
            if i == j:
                symmetrical_uncertaninty_matrix[i, j] = 0
            else:
                symmetrical_uncertaninty_matrix[i, j] = 2 * (
                        information_gain_matrix[i, j] / (feature_entropies[i] + feature_entropies[j]))
    return symmetrical_uncertaninty_matrix


def avg_su(dataset):
    "计算平均SU"
    N, D = np.shape(dataset)
    feature_avg_su = np.zeros(D)  # 用于存放每个特征的平均SU
    symmetrical_uncertaninty_matrix = symmetrical_uncertaninty(dataset)
    feature_su_sum = np.sum(symmetrical_uncertaninty_matrix, axis=1)
    for i in range(D):
        feature_avg_su = feature_su_sum / (D - 1)
    return feature_avg_su


def context(dataset):
    "计算特征 与其他特征间的相关性"
    N, D = np.shape(dataset)
    context_matrix = np.zeros([D, D])
    symmetrical_uncertaninty_matrix = symmetrical_uncertaninty(dataset)
    feature_avg_su = avg_su(dataset)
    for i in range(D):
        for j in range(D):
            if (symmetrical_uncertaninty_matrix[i, j] >= feature_avg_su[i]):
                context_matrix[i, j] = 1
            else:
                context_matrix[i, j] = 0
    return context_matrix


def context_based_dis_X(dataset, feature_x):
    "计算X特征下的，每个值之间的距离"
    x_value, _ = probability_entropy(dataset, feature_x)
    dis_x = np.zeros([len(x_value), len(x_value)])  # 创建x值矩阵
    context_matrix = context(dataset)
    context_feature_row = context_matrix[feature_x]  # 取出x所在特征，那一行的context值【0 0 1 0 1...】
    indices_context = np.where(context_feature_row == 1)[0]  # 找到当前特征的context==1时的位置
    for Y in indices_context:  # 循环每一个context特征 Y
        y_value = probability_entropy(dataset, Y)[0]  # 找到当前Y特征下的，所有特征值
        for y in y_value:  # 循环每个值
            index_row_y = np.where(dataset[:, Y] == y)[0]  # Y其中一个特征值y在原始数据集的行位置
            subset_data = dataset[index_row_y, :]  # 提取对应行位置的数据，构成子集
            subset_x_value_t, probabilities_x_y_t = probability_entropy(subset_data,
                                                                        feature_x)  # 获得暂时p（x|y）【p1,p2,p3...】，后期有判断

            if len(subset_x_value_t) != len(x_value):  # 子集中的特征值数量与全集中特征值的数量不一样
                subset_x_value_t = np.resize(subset_x_value_t, x_value.shape)  # 将子集数组扩大到与全集数组等大的长度
                index = 0
                while index < (len(x_value)):
                    if np.equal(x_value[index], subset_x_value_t[index]):
                        index += 1
                    else:  # 如果不相等时填充完子集后，索引后移
                        subset_x_value_t = np.insert(subset_x_value_t, index, 'null')  # 在子集中对应位置添加缺少的值
                        probabilities_x_y_t = np.insert(probabilities_x_y_t, index, 0)  # 在概率中用0补充缺少特征值的，位置的值
                        index += 1
                subset_x_value_t = subset_x_value_t[:len(x_value)]  # 修剪长度
            # else
            # 此时，p的维度随着子集的维度 ，扩充到正常大小，可以计算距离并写入矩阵中
            for i in range(len(x_value)):
                for j in range(len(x_value)):
                    dis_x[i, j] += (probabilities_x_y_t[i] - probabilities_x_y_t[j]) ** 2
        # 此时，在Y特征下，x特征值之间距离计算完毕
    # 此时，所有X的context特征下，x特征值值之间距离计算完毕
    dis = np.sqrt(dis_x)
    dis = dis.tolist()
    return dis

def CBDMPre(dataset):
    N, D = np.shape(dataset)
    AttMtx = []
    for d in range(D):
        context_dis = context_based_dis_X(dataset, d)  # 得到d特征的值之间的距离
        # print("context", context_dis)
        # print("context",type(context_dis))
        AttMtx.append(context_dis)
    return AttMtx

# def context_based_dis(dataset, centers_p, att_mtx):
#     N, D = np.shape(dataset)
#     # K,_,_= np.shape(centers_p)
#     K = len(centers_p)
#     distance = np.zeros([N, K])
#     for i in range(N):
#         for j in range(K):
#             SubDist = np.zeros(D)  # 每个属性下的距离
#             for r in range(D):  # 第r个属性
#                 v = np.unique(dataset[:,r])
#                 index_i = int( np.where(np.array(v) == dataset[i,r])[0] )
#                 # print(index_i)
#                 row_v = att_mtx[r][index_i]
#                 # print("row v",row_v)
#                 row_c = centers_p[j][r]
#                 if (len(row_v) != len(row_c)):
#                     print("row v", row_v)
#                     print("row c", row_c)
#                     print("CBDM：簇心和数据集属性下长度不同","点",i,"簇",centers_p[j])
#                 else:
#                     SubDist[r] = np.sum(row_v * row_c)
#             distance[i,j] = np.sum(SubDist)
#
#     lab = np.argmin(distance, axis=1)
#     distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))  # 归一化
#
#     return distance, lab


def context_based_dis(dataset, centers, att_mtx):
    N, D = np.shape(dataset)
    K,_= np.shape(centers)
    distance = np.zeros([N, K])
    for d in range(D):  # 每个特征都抽取出来，挨个计算此特征下两个数据点之间的距离并加和

        d_value = np.unique(dataset[:, d])
        subset_data = dataset[:, d]  # 得到d特征那一列
        subset_centers = centers[:, d]
        # print("subset_centers",subset_centers)
        for i in range(N):  # 行坐标
            for j in range(K):
                # print("j",j)
                index_i = np.where(d_value == subset_data[i])[0][0]  # 数据点的d特征下的x值
                index_j = np.where(d_value == subset_centers[j])[0][0]  # 中心的d特征下的x值


                distance[i, j] += att_mtx[d][index_i][index_j]
    lab = np.argmin(distance, axis=1)
    distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))  # 归一化
    # print(distance)
    return distance, lab



