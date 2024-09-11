import numpy as np
from Other_fuc import adam
# from MEL_ECA_lab import N,D,K
M = 3
def entropy_att(dataset):
    n, d = np.shape(dataset)
    entropy_att = np.zeros((d))
    for i in range(d):
        value,number = np.unique(dataset[:,i], return_counts=True)
        # print("value,number",value,number)
        pro = number/n
        pro = pro[pro > 0]
        entropy = -np.sum(pro*np.log2(pro))
        entropy_att[i] = entropy
    return entropy_att

def weight_calc(dataset, labs, weight, mt_1, vt_1):
    K = len(np.unique(labs[0]))
    entropy = np.zeros((M))
    full_entropy_att = entropy_att(dataset) #全空间每个属性的熵值
    # print(full_entropy_att)
    for m in range(M):
        # print("m",m)
        lab = labs[m]   #取第m个metric下的结果
        for k in range (K): #第k个簇
            index = np.where(lab == k)[0]
            sub_dataset = dataset[index,:]
            sub_entropy_att = entropy_att(sub_dataset)  #子空间下每个属性的熵值
            # print("m",m,"k",k,"sub_entropy_att",sub_entropy_att)
            entropy[m] += np.sum( sub_entropy_att / full_entropy_att )
    # avg = np.mean(entropy)
    weight,mt,vt = adam(weight, entropy, mt_1, vt_1)

    weight = weight / np.sum(weight)

    return weight,entropy, mt, vt



