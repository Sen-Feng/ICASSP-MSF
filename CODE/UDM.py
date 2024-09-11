#################        计算UDM距离              ########################
import numpy as np
# from MEL_ECA_lab import N,D,K

def Entropy(X, XX, Vmax):
    # print("X",X)
    # print("XX",XX)
    # print("Vmax",Vmax)
    Xlth = len(XX)
    RangeMax = np.unique(X)
    SubE = np.zeros(np.shape(RangeMax)[0])
    j = 0
    for i in RangeMax:
        p = len(np.where(X == i)[0]) / Xlth
        if p == 0:
            SubE[j] = 0
        else:
            SubE[j] = -p * np.log2(p)
        j += 1

    EntropyX = np.sum(SubE)
    SubS = np.ones(np.shape(Vmax)[0]) * (-(1 / len(Vmax)) * np.log2(1 / len(Vmax)))
    StandEntropy = np.sum(SubS)
    # print( "SubS",SubS )
    # print(StandEntropy)

    Dist = EntropyX / StandEntropy

    return Dist

def UDMPre(X, Pm, ValNum):

    X = X[:, :Pm['Xwd']]
    for i in range(Pm['Owd'], Pm['Xwd']):
        new_row = np.unique(X[:, i])
        ValNum[i] = new_row
    WeightMtx = np.zeros((Pm['Xwd'], Pm['Xwd']))  # 不同属性之间的权重 R
    # 先为属性计算权重
    for i in range(Pm['Xwd']):  # 第i个属性
        for j in range(Pm['Xwd']):  # 第j个属性
            # noinspection PyTypeChecker
            ConMtx = np.zeros((len(ValNum[i]), len(ValNum[i])))  # 创建两属性中，值与值之间的条件矩阵
            Mid = 0
            for m in range(len(ValNum[i])):  # 第i个属性的第m个值
                MValLct = np.where(X[:, i] == ValNum[i][m])[0]  # 找到i属性中第m个属性值对应的行#eg. [3,4]
                CorValM = X[MValLct, j]  # 切出这些行，所对应的第j列的值  eg.【usa，china】
                for h in range(len(ValNum[i])):  # 第i个属性的第h个值   此时x(i,m)<x(i，h)
                    if h > m:
                        HValLct = np.where(X[:, i] == ValNum[i][h])[0]  # 找到i属性中第h个属性值对应的行   eg.[2]
                        CorValH = X[HValLct, j]  # 切出这些行，所对应的第j列的值    eg.[uk]
                        if i < Pm['Owd'] and j < Pm['Owd']:
                            for r in range(len(MValLct)):
                                for g in range(len(HValLct)):
                                    if CorValM[r] < CorValH[g]:  # m所对应的，第j个属性下的属性值<h所对应的，第j个属性下的属性值
                                        ConMtx[m, h] += 1  # positive_concordant
                                    if CorValM[r] > CorValH[g]:
                                        ConMtx[h, m] += 1  # negative_concordant
                        if i >= Pm['Owd'] or j >= Pm['Owd']:  # 有一属性是nominal
                            ConMtxMH = np.zeros((len(ValNum[j]), len(ValNum[j])))
                            for r in range(len(MValLct)):
                                for g in range(len(HValLct)):
                                    if CorValM[r] != CorValH[g]:  # o(s,m)≠o(s,h) ->
                                        ConMtxMH[np.where(np.atleast_1d(ValNum[j]) == CorValM[r])[0],
                                        np.where(np.atleast_1d(ValNum[j]) == CorValH[g])[0]] += 1
                            SubCon = 0
                            SubDisCon = 0
                            for r in range(len(ValNum[j])):
                                for g in range(len(ValNum[j])):
                                    if g > r:
                                        if ConMtxMH[r, g] > ConMtxMH[g, r]:
                                            SubCon += ConMtxMH[r, g]
                                            SubDisCon += ConMtxMH[g, r]
                                        if ConMtxMH[r, g] < ConMtxMH[g, r]:
                                            SubCon += ConMtxMH[g, r]
                                            SubDisCon += ConMtxMH[r, g]
                            ConMtx[m, h] = np.sum(np.sum(SubCon))
                            ConMtx[h, m] = np.sum(np.sum(SubDisCon))
                CorValMCandi = np.unique(CorValM)
                for u in range(len(CorValMCandi)):  # 求C=
                    SubMidLth = len(np.where(CorValM == CorValMCandi[u])[0])
                    if SubMidLth > 1:
                        Mid += SubMidLth * (SubMidLth - 1) / 2
                Con = 0
                DisCon = 0
                for r in range(len(ValNum[i])):
                    for g in range(len(ValNum[i])):
                        if g > r:
                            Con += ConMtx[r, g]
                        if g < r:
                            DisCon += ConMtx[r, g]

                WeightMtx[i, j] = 2 * (np.abs(Con - DisCon) + Mid) / (Pm['Xlth'] * (Pm['Xlth'] - 1))
    AttMtx = [None] * Pm['Xwd']
    for i in range(Pm['Xwd']):  # 每个属性
        if i + 1 <= Pm['Owd']:  # 属于 ordinal
            DistMtx = np.zeros((len(ValNum[i]), len(ValNum[i])))  # 创建i属性值大小*i属性值大小的矩阵
            for m in range(len(ValNum[i])):  # 循环访问每个值
                for h in range(len(ValNum[i])):
                    if h > m:  # 后一个值大于前一个  A>B ->此时就要计算B到A之间的熵距离和
                        SubDist = np.zeros(Pm['Xwd'])
                        for j in range(Pm['Xwd']):  # 循环每个属性
                            SubSubDist = np.zeros(h - m)  # 存放B到A之间 走过的 熵距离
                            for g in range(m, h):  # 从m到h->就是从B到A
                                ValListM = X[(np.where(np.atleast_1d(X[:, i]) == ValNum[i][g])[0]), j]
                                ValListH = X[(np.where(np.atleast_1d(X[:, i]) == ValNum[i][g + 1])[0]), j]
                                ValList = np.concatenate([ValListM, ValListH])
                                SubSubDist[g - m] = Entropy(ValList, X[:, j], ValNum[j])  # g到g+1的距离
                            SubDist[j] = np.sum(SubSubDist)
                        DistMtx[m, h] = np.mean(SubDist * WeightMtx[i, :])
                        DistMtx[h, m] = DistMtx[m, h]
            # print("DostMtx",DistMtx.tolist())
            # print(type(DistMtx))
            AttMtx[i] = DistMtx.tolist()
        if i + 1 > Pm['Owd']:
            DistMtx = np.zeros((len(ValNum[i]), len(ValNum[i])))

            for m in range(len(ValNum[i])):
                for h in range(len(ValNum[i])):
                    if h > m:
                        SubDist = np.zeros(Pm['Xwd'])
                        for j in range(Pm['Xwd']):
                            ValListM = X[(np.where(X[:, i] == ValNum[i][m])[0]), j]
                            ValListH = X[(np.where(X[:, i] == ValNum[i][h])[0]), j]
                            ValList = np.concatenate([ValListM, ValListH])
                            SubDist[j] = Entropy(ValList, X[:, j], ValNum[j])
                        DistMtx[m, h] = np.mean(SubDist * WeightMtx[i, :])
                        DistMtx[h, m] = DistMtx[m, h]
            AttMtx[i] = DistMtx.tolist()  # min为标签

    return AttMtx  # 此为每个属性值之间的距离


# def udm_dis(dataset, centers_p, Pm, AttMtx, ValNum):
#     N, D = np.shape(dataset)
#     # K, _, _ = np.shape(centers_p)
#     K = len(centers_p)
#     distance = np.zeros([N, K])
#     for i in range(N):
#         for j in range(K):
#             SubDist = np.zeros(D)  # 每个属性下的距离
#             for r in range(D):  # 第r个属性
#                 index_i = int(np.where(np.array(ValNum[r]) == (dataset[i, r]))[0])
#
#                 # row_v = AttMtx[r][index_i][0]
#                 row_v = AttMtx[r][index_i]
#                 # print("row v",row_v)
#                 row_c = centers_p[j][r]
#                 if (len(row_v) != len(row_c)):
#                     print("row v", len(row_v))
#                     print("row v", row_v)
#                     print("row c", len(row_c))
#                     print("row c", row_c)
#                     print("UDM：簇心和数据集属性下长度不同")
#                 else:
#                     SubDist[r] = np.sum(row_v * row_c)
#             distance[i, j] = np.sum(SubDist)
#     lab = np.argmin(distance, axis=1)
#     distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))  # 归一化
#
#     return distance, lab

def mddm_dis(I, J, att_mtx, pm, ValNum):
    SubDist = np.zeros(pm['Xwd'])  # 每个属性下的距离
    for r in range(pm['Xwd']):  # 第r个属性
        index_i = np.where(np.array(ValNum[r]) == str(I[r]))[0]
        index_j = np.where(np.array(ValNum[r]) == str(J[r]))[0]
        if len(index_i) > 0 and len(index_j) > 0:
            # print()
            a = int(index_i[0])
            b = int(index_j[0])
            SubDist[r] = att_mtx[r][a][b]
            # SubDist[r] = (att_mtx[r][index_i[0]][index_j[0]])
    # print("SubDist",SubDist)
    DistVct = np.sqrt(np.sum(np.power(SubDist, 2)))
    return DistVct
def udm_dis(dataset, center, Pm, AttMtx, ValNum):
    N,D = np.shape(dataset)
    K,_= np.shape(center)
    # print(AttMtx)

    distance = np.zeros((N, K))
    for i in range(Pm['Xlth']):  # 循环每一个样本
        DistVec = np.zeros(Pm['k'])  # 创建一个长度为Pm['k']（类别数）的零向量DistVec，用于保存样本i与每个类别的距离。
        for j in range(Pm['k']):  # 对于每个簇心
            DistVec[j] = mddm_dis(dataset[i, :], center[j, :], AttMtx, Pm, ValNum)

            distance[i, j] = DistVec[j]

    lab = np.argmin(distance, axis=1)  # 找到DistVec中最小值所对应的类别标签Winner。

    distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))  # 归一化
    # sim1 = sim_calc(distance)

    return distance, lab


