import arff
import numpy as np
import pandas as pd
from scipy.io import loadmat
import time

from UDM import UDMPre
from EBDM import EBDMPre
from CBDM import CBDMPre
from Other_fuc import modes_init,find_ture_labs,find_ture_labs_first,Eva_CA,update_centers,OCIL_init,creat_centers
from Metric_func import compute_full_metric,weight_metric
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score


D=None
N=None
K = None
M = 3
Pm = None
ValNum = None
AttMtx = None
T = 3




def MEl(dataset, center, Pm, ValNum, AttMtx):
    old_lab = np.zeros((N))
    old_weight = np.full((M,), 1/M)
    old_center = np.copy(center)
    labs_iter0 = None
    mt_1 = np.array([0.0, 0.0, 0.0])
    vt_1 = np.array([0.0, 0.0, 0.0])
    for e in range(100):
        weight, labs, mt, vt = compute_full_metric(dataset,old_center,AttMtx,Pm,ValNum,old_weight, mt_1, vt_1)
        if e == 0:
            labs_iter0 = labs
        print("weight",weight)
        lable = weight_metric(dataset,old_center,weight,AttMtx,Pm,ValNum)
        centers = update_centers(dataset, lable, old_center)
        if np.array_equal(lable, old_lab) == True or np.array_equal(centers, old_center) == True:
            break
        else:
            old_lab = lable
            old_center = centers
            old_weight = weight
            mt_1 = mt
            vt_1 = vt

    return lable,labs_iter0

def text(dataset,center,Pm,ValNum, true_labs,K):

    AttMtx_u = UDMPre(dataset, Pm, ValNum)
    AttMtx_e = EBDMPre(dataset)
    AttMtx_c = CBDMPre(dataset)
    AttMtx = [AttMtx_u, AttMtx_c, AttMtx_e]

    lab,_ = MEl(dataset, center, Pm, ValNum, AttMtx)
    ca = Eva_CA(lab, true_labs)
    ari = adjusted_rand_score(true_labs, lab)
    nmi = normalized_mutual_info_score(true_labs, lab)
    return ca, ari, nmi

def circle(dataset, center, Pm, ValNum, AttMtx,K):
    CA = []
    ARI = []
    NMI = []
    for i in range(10):
        ca, ari, nmi = text(dataset, center, Pm, ValNum, AttMtx,K)
        CA.append(ca)
        ARI.append(ari)
        NMI.append(nmi)
    # file_name = 'E:/result.xlsx'
    # data = {'CA': CA, 'ARI': ARI, 'NMI': ARI}
    # df = pd.DataFrame(data)
    # df.to_excel(file_name, index=False)
    # print(f"Data saved to {file_name}")
    return

if __name__ == "__main__":

    data = loadmat("E:/ECAI-zmj-dataset/data/TT.mat")
    print(data)
    datas = data['TT']
    dataset = np.array(datas, dtype=object)
    true_labs, K = find_ture_labs(dataset)
    dataset = np.delete(dataset, -1, axis=1)
    dataset = dataset.astype(str)
    N, D = np.shape(dataset)
    ValNum = [None] * D
    for d in range(D):
        ValNum[d] = np.unique(dataset[:, d])
    Owd = 0

    v,n = np.unique(true_labs, return_counts=True)
    print(v,n)

    # data = pd.read_csv("E:/ECAI-zmj-dataset/Car Evaluation/car.data", delimiter=",",
    #                    dtype='object')
    # dataset = np.array(data.dropna())
    # true_labs, K = find_ture_labs(dataset)
    # dataset = np.delete(dataset, -1, axis=1)
    # N, D = np.shape(dataset)
    # ValNum = [None] * 6
    # ValNum[0] = ['vhigh', 'high', 'med', 'low']
    # ValNum[1] = ['vhigh', 'high', 'med', 'low']
    # ValNum[2] = ['2', '3', '4', '5more']
    # ValNum[3] = ['2', '4', 'more']
    # ValNum[4] = ['small', 'med', 'big']
    # ValNum[5] = ['low', 'med', 'high']
    # Owd = 6


    center = OCIL_init(dataset, K)
    # center = creat_centers(dataset, K)
    Pm = {'Xlth': 0, 'Xwd': 0, 'Owd': 0}
    Pm['Xlth'], Pm['Xwd'] = dataset.shape  # 获得行数和列数
    Pm['k'] = K  # 获取总共几个标签->k
    Pm['Owd'] = Owd  # Ordinal所在的列数
    text(dataset, center, Pm, ValNum, true_labs,K)
