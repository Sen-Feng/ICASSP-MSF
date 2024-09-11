import numpy as np
# from MEL_ECA_lab import N,D,K

#################        计算entropy_based_distance              ###################

def att_value_entropy(dataset, feature_d):
    "用于求特征d下，所有特征值的概率"
    feature_data = dataset[:, feature_d]
    a, counts = np.unique(feature_data, return_counts=True)  # a：有哪些值 ；counts：这些值在数据集中出现的次数

    probabilities = counts / len(feature_data)  # 计算概率分布
    entropy_d = -probabilities * np.log2(probabilities)  # 获得当前特征，每个特征值的E
    entropy_d = list(entropy_d)
    return entropy_d

def entropy_dis(entropy_pv_dis):
    nv = len(entropy_pv_dis)
    entropy_dis = np.zeros((nv,nv))
    for i in range(nv):
        for j in range(nv):
            if i == j:
                entropy_dis[i,j] = 0
            else:
                entropy_dis[i, j] = entropy_pv_dis[i] + entropy_pv_dis[j]
    return entropy_dis.tolist()
def EBDMPre(dataset):
    N, D = np.shape(dataset)
    AttMtx = []
    for d in range(D):
        entropy_pv_dis = att_value_entropy(dataset , d)
        entropy_dis1 = entropy_dis(entropy_pv_dis)
        # print("11",type(entropy_dis1))
        AttMtx.append(entropy_dis1)
    return AttMtx

# def entropy_based_dis(dataset, centers_p, att_mtx):
#     N, D = np.shape(dataset)
#     # K, _, _ = np.shape(centers_p)
#     K = len(centers_p)
#     distance = np.zeros([N, K])
#     for i in range(N):
#         for j in range(K):
#             SubDist = np.zeros(D)  # 每个属性下的距离
#             for r in range(D):  # 第r个属性
#                 v = np.unique(dataset[:, r])
#                 index_i = int(np.where(np.array(v) == dataset[i, r])[0])
#                 # print("index i",index_i)
#                 row_v = att_mtx[r][index_i] #第r个属性下，index i的值所在地的行，v与本属性下其他v之间的距离
#                 # print("row v", row_v)
#                 row_c = centers_p[j][r]
#                 if (len(row_v) != len(row_c)):
#                     print("row v", row_v)
#                     print("row c", row_c)
#                     print("EBDM：簇心和数据集属性下长度不同", "点", i, "簇", centers_p[j])
#                 else:
#                     SubDist[r] = np.sum(row_v * row_c)
#             distance[i, j] = np.sum(SubDist)
#
#     lab = np.argmin(distance, axis=1)
#     distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))  # 归一化
#
#     return distance, lab


def entropy_based_dis(dataset, centers, att_mtx):
    N, D = np.shape(dataset)
    K, _ = np.shape(centers)
    distance = np.zeros([N, K])

    for d in range(D):
        d_value = np.unique(dataset[:,d])
        subset_data = dataset[:, d]  # 提取第d列
        subset_centers = centers[:, d]
        for i in range(N):  # 行坐标
            for j in range(K):  # 列坐标
                index_i = np.where(d_value == subset_data[i])[0]
                index_j = np.where(d_value == subset_centers[j])[0]
                if index_j == index_i:
                    distance[i, j] += 0
                else:
                    # distance[i, j] += (entropy_d[index_j][0] + entropy_d[index_i][0]) ** 2
                    distance[i, j] += (att_mtx[d][index_i[0]][index_j[0]])**2
    distance = np.sqrt(distance)
    # print(distance)
    lab = np.argmin(distance, axis=1)
    distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))  # 归一化

    return distance, lab