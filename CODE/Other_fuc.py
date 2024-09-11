import numpy as np
from scipy.optimize import linear_sum_assignment



def adam(weight, e, mt_1, vt_1):
    b1 = 0.9
    b2 = 0.999
    c = 0.00000001
    a = 0.01
    mean = np.average(e)
    d = e - mean
    mt = b1 * mt_1 + (1 - b1) * d
    vt = b1 * vt_1 + (1 - b2) * np.square(d)
    mt = mt / (1 - b1)
    vt = vt / (1 - b2)
    weight = weight - a * mt / (np.sqrt(vt) + c)

    return weight, mt, vt

def find_ture_labs(dataset):
    N, D = np.shape(dataset)
    dataset_class = dataset[:, -1]
    class_vlue = np.unique(dataset_class)
    labs = np.zeros(N)
    for n in range(N):
        index = np.where(class_vlue == dataset_class[n])[0]
        labs[n] = index[0]
    return labs, len(class_vlue)


def find_ture_labs_first(dataset):
    N, D = np.shape(dataset)
    dataset_class = dataset[:, 0]
    class_vlue = np.unique(dataset_class)
    labs = np.zeros(N)
    for n in range(N):
        index = np.where(class_vlue == dataset_class[n])[0]
        labs[n] = index[0]
    return labs, len(class_vlue)


def Eva_CA(dataCluster, dataLabel):
    nData = len(dataLabel)
    nC = int(max(dataLabel)) + 1
    E = np.zeros((nC, nC))
    for m in range(nData):
        i1 = int(dataCluster[m])
        i2 = int(dataLabel[m])
        E[i1, i2] += 1
    E = -E
    row_ind, col_ind = linear_sum_assignment(E)
    nMatch = -E[row_ind, col_ind].sum()
    accuracy = nMatch / nData
    return accuracy

def update_centers(dataset, lab, center):
    N, D = np.shape(dataset)
    K,_ = np.shape(center)
    centers = np.zeros_like(center)
    np.copyto(centers, center)
    for i in range(K):
        index = np.where(lab == i)[0]
        if len(index) != 0:
            subset_data = dataset[index]
            for d in range(D):
                subset_data_d = subset_data[:, d]
                unique_vals, counts = np.unique(subset_data_d, return_counts=True)
                index = np.argmax(counts)
                centers[i, d] = (unique_vals[index])
    return centers


def OCIL_init(dataset,K):
    N,D = np.shape(dataset)
    sim_x_X = np.zeros((N))
    for i in range(N):
        sim = 0
        for d in range(D):
            sub_data = dataset[:,d]
            value, number = np.unique(sub_data,return_counts=True)
            index = np.where(value == dataset[i][d])[0]
            sim += number[index[0]]/N
        sim_x_X[i] = sim/D
    index = np.argmax(sim_x_X)
    u=[]
    u.append(dataset[index])
    for i in range(1,K):
        sim_x_U = np.zeros((N))
        for j in  range (N):
            sim = 0
            for d in range(D):
                sub_data = [row[d] for row in u]
                value, number = np.unique(sub_data, return_counts=True)
                index = np.where(value == dataset[j][d])[0]
                if len(index) !=0:
                    sim += number[index[0]] / i
                else :
                    sim += 0
            sim_x_U[j] = sim / D
        d_sim_x_U = 1- sim_x_U
        pry = d_sim_x_U + sim_x_X
        index_p = np.argmax(pry)
        u.append(dataset[index_p])
    center = np.array(u)
    return center

def juli_gailvhua(lab,dataset,sim):
    N, D = np.shape(dataset)
    K = len( np.unique(lab))
    distance = np.zeros_like(sim)
    for i in range(N):
        for k in range(K):          #对于每个簇 计算当前点与簇的关系
            index_k = np.where(lab == k)[0]  #找到这个簇内的所有点
            for d in range(D):      #对于每个属性
                attribute_value, value_number = np.unique(dataset[index_k, d], return_counts=True)
                index = np.where(attribute_value == dataset[i,d])[0]        #找到当前点d属性的值所处的位置
                if len(index)!=0:
                    distance[i,k] +=  value_number[index[0]]/len(index_k)
                else:               #当前点d属性的值在这个簇里没有
                    distance[i, k] += 0
    return distance/D

def creat_centers(dataset, k):
    error_message = True
    while error_message:
        indices = np.random.choice(len(dataset), k, replace=False)
        initial_clusters = dataset[indices]
        has_equal_rows = any(
            (initial_clusters[i] == initial_clusters[j]).all() for i in range(len(initial_clusters)) for j in
            range(i + 1, len(initial_clusters)))
        if has_equal_rows:
            error_message = True
        else:
            error_message = False
    return initial_clusters