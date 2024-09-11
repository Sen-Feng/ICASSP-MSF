import numpy as np
from Weight import weight_calc
from UDM import  udm_dis        #0
from EBDM import entropy_based_dis      #2
from CBDM import context_based_dis      #1

from Other_fuc import juli_gailvhua,update_centers,Hamming_dis
M = 3

def compute_full_metric(dataset, center, AttMtx, Pm, ValNum, weight, mt_1, vt_1):
    N, D = np.shape(dataset)
    K = len(center)
    labs = []
    for m in range(M):
        centers = center
        old_lab = np.zeros([N, K])
        lab = []
        for e in range(100):
            if m == 0:
                distance, lab = udm_dis(dataset, centers, Pm, AttMtx[0], ValNum)
            elif m == 1:
                distance, lab = context_based_dis(dataset,centers,AttMtx[1])
            elif m == 2:
                distance, lab = entropy_based_dis(dataset,centers,AttMtx[2])
            if np.array_equal(lab, old_lab):
                break
            else:
                if e != 0:
                    different_indices = np.where(lab != old_lab)[0]
                    print(len(different_indices))
                old_lab = lab
                centers = update_centers(dataset,lab,centers)
        labs.append(lab)
    labs = np.array(labs)
    weight,entropy,mt,vt = weight_calc(dataset,labs,weight,mt_1,vt_1)
    print("entropy",entropy)

    return weight, labs, mt, vt

def weight_metric(dataset,center,weight,AttMtx,Pm,ValNum):
    N, D = np.shape(dataset)
    K = len(center)
    old_lab = np.zeros((N))
    centers = center
    oj1 = 0.0
    oj2 = 0.0
    for e in range(100):
        sim_full = np.zeros(( N, K ))
        for m in range(M):
            if m == 0:
                distance, lab = udm_dis(dataset, centers, Pm, AttMtx[0], ValNum)
                sim_full += juli_gailvhua(lab, dataset, distance)*weight[m]
            elif m == 1:
                distance, lab = context_based_dis(dataset, centers, AttMtx[1])
                sim_full += juli_gailvhua(lab, dataset, distance) * weight[m]
            else:
                distance, lab = entropy_based_dis(dataset, centers, AttMtx[2])
                sim_full += juli_gailvhua(lab, dataset, distance) * weight[m]
        row_max_values = np.max(sim_full, axis=1)
        sum_of_max_values = np.sum(row_max_values)
        full_lab = np.argmax(sim_full,axis=1)
        if (np.array_equal(full_lab, old_lab) == True )or(oj1 == sum_of_max_values)or(oj2 == sum_of_max_values) :
            break
        else:
            centers = update_centers(dataset,full_lab,centers)
            old_lab = full_lab
            oj2 = oj1
            oj1 = sum_of_max_values
    return full_lab




