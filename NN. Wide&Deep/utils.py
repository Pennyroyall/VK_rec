import io
from dateutil.parser import parse
import six
import datetime
from time import time
import numpy as np
import pickle
import scipy.sparse as sparse
import logging
import implicit
import sys
import random

def pickle_dump(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

    
def pickle_load(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def convert_to(R_train, load=False):
    if load==True:
        G_train = pickle_load('z_G_train.pckl') 
        Y_train = pickle_load('z_Y_train.pckl')
    if load==False:
        
        U, I = np.shape(R_train)
        C = R_train.nnz
        G = sparse.lil_matrix((C, U+I), dtype='int')
        Y = np.zeros((C, 1))

        N = 0
        for u in range(0, U):
            if u%10000==0:
                print(u)
            for i in R_train[u, :].indices:
                G[N, u] = 1 #user embedding
                G[N, U + i] = 1 #item embedding
                Y[N, 0] = R_train[u, i]
                N += 1
        G_train = G.tocsr()
        Y_train = Y
        pickle_dump(G_train, 'z_G_train.pckl')
        pickle_dump(Y_train, 'z_Y_train.pckl')
    return G_train, Y_train   



def to_test(R, R_train, P, Q, N_users=1000, N=100, load=False):    
    if load==True:
        test_users = pickle_load('z_test_users.pckl')
        users_cases = pickle_load('z_users_cases.pckl')
        
    if load==False:    
        R_test = R - R_train
        U, I = np.shape(R)
        test_users = random.sample(range(U), N_users)
        users_cases = []
        to_remove = []
        for u in test_users:
            if (R[u,:].nnz==R_train[u,:].nnz):
                to_remove.append(u)
                continue
                
            temp = P[u, :].dot(Q)
            train_indices = R_train[u,:].indices
            temp[train_indices] = -np.inf
            users_cases.append(np.argpartition(temp, -N)[-N:])

        for i in range(0, len(to_remove)):
            test_users.remove(to_remove[i])
            
        pickle_dump(test_users, 'z_test_users.pckl')
        pickle_dump(users_cases,'z_users_cases.pckl') 
    return test_users, users_cases
    




def to_test_simple(R, R_train, N_users=1000, load=False): 
    U, I = np.shape(R)
    test_users = random.sample(range(U), N_users)
    to_remove = []
    for u in test_users:
        if (R[u,:].nnz==R_train[u,:].nnz) or (R_train[u,:].nnz==0) or (R[u,:].nnz==0):
            to_remove.append(u)
            continue

    for i in range(0, len(to_remove)):
        test_users.remove(to_remove[i])   

    return test_users    

    
def test_to_onehot(U, I, test_user, users_case):
    A = sparse.lil_matrix((len(users_case), U + I))
    c=0
    for i in users_case:
        A[c, test_user] = 1
        A[c, U+i] = 1
        c += 1

    return A.tocsr()   


def test_to_dense(P, Q, test_user, users_case):
    current_list = []
    list_2 = []
    for i in users_case:

        temp1 = np.array([test_user, i])
        add = np.asarray([np.linalg.norm(P[temp1[0], :]), np.linalg.norm(Q[:, temp1[1]]), P[temp1[0], :].dot(Q[:, temp1[1]])])
        temp2 = np.concatenate((P[temp1[0], :], Q[:, temp1[1]], add))
        current_list.append(temp2)
        list_2.append(temp1)

    current_matrix = np.matrix(current_list)
    matrix_2 = np.matrix(list_2)
    return matrix_2, current_matrix   



def test_siamese(P, Q, test_user, users_case):
    list_1 = []
    list_2 = []
    list_3 = []
    for i in users_case:

        temp1 = np.array([test_user, i])#, P[test_user, :].dot(Q[:, i])])
        temp2 = np.concatenate((P[test_user, :], [np.linalg.norm(P[test_user, :])]))
        temp3 = np.concatenate((Q[:, i], [np.linalg.norm(Q[:, i])]))

        list_1.append(temp1)
        list_2.append(temp2)
        list_3.append(temp3)

    matrix_1 = np.matrix(list_1)
    matrix_2 = np.matrix(list_2)
    matrix_3 = np.matrix(list_3)
    return matrix_1, matrix_2, matrix_3 


def test_nn(P, Q, Q_lda, test_user, users_case):

    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    for i in users_case:

        temp1 = np.concatenate((P[test_user, :], [np.linalg.norm(P[test_user, :])]))
        temp2 = np.concatenate((Q[:, i], [np.linalg.norm(Q[:, i])]))
        temp3 = np.concatenate((P[test_user, :], [np.linalg.norm(P[test_user, :])]))
        temp4 = np.concatenate((Q[:, i], [np.linalg.norm(Q[:, i])], Q_lda[i, :]))

        list_1.append(temp1)
        list_2.append(temp2)
        list_3.append(temp3)
        list_4.append(temp4)

    matrix_1 = np.matrix(list_1)
    matrix_2 = np.matrix(list_2)
    matrix_3 = np.matrix(list_3)
    matrix_4 = np.matrix(list_4)
    return matrix_1, matrix_2, matrix_3, matrix_4


def test_wd(P, Q, Q_lda, test_user, users_case):

    list_1 = []
    list_2 = []

    for i in users_case:

        temp1 = np.array([test_user, i])
        temp2 = np.concatenate((P[test_user, :], Q[:, i], [np.linalg.norm(P[test_user, :])], [np.linalg.norm(Q[:, i])], [P[temp1[0], :].dot(Q[:, temp1[1]])], Q_lda[i, :]))

        list_1.append(temp1)
        list_2.append(temp2)

    matrix_1 = np.matrix(list_1)
    matrix_2 = np.matrix(list_2)

    return matrix_1, matrix_2  


def grid_bin(R_train, grid):
    R_b = (R_train>0).astype('int')

    for i in grid:
        R_b += (R_train>=i).astype('int')

    return R_b    




