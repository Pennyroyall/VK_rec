from utils import *

from sklearn.metrics import mean_squared_error
import logging
import keras
from keras import regularizers
from keras import Sequential
from keras.layers import *
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json





if (sys.argv[1]).lower()=='create_test':
    N_USERS = int(sys.argv[2])
    SIZE = int(sys.argv[3])

    tt = time()
    P = pickle_load(path_data + 'z_P.pckl')
    Q = pickle_load(path_data + 'z_Q.pckl')
    R_test = pickle_load(path_data + 'z_R_test_1.pckl')
    R_train = pickle_load(path_data + 'z_R_train_1.pckl')

    R = R_test + R_train

    test_users, users_cases = to_test(R, R_train, P, Q, N_users=N_USERS, N=SIZE, load=False)
    # test_users, users_cases = publics_to_test(R, R_train, P, Q, N_users=N_USERS, N=SIZE, load=False)
    print('getting test users, time:', (time()-tt))



if (sys.argv[1]).lower()=='ndcg':

    NAME = str(sys.argv[2])
    N_USERS = int(sys.argv[3])
    SIZE = int(sys.argv[4])    
    EP = str(sys.argv[5])   


    with open('MODELS/' + NAME + '_architecture.json', 'r') as f:
        model = model_from_json(f.read())

    model.load_weights('CHECKPOINTS/' + NAME + 'weights.'+ EP + '.hdf5')

    print('\n\nModel loaded ok')




    tt = time()
    P = pickle_load(path_data + 'z_P.pckl')
    Q = pickle_load(path_data + 'z_Q.pckl')
    R_test = pickle_load(path_data + 'z_R_test_1.pckl')
    R_train = pickle_load(path_data + 'z_R_train_1.pckl')

    R = R_test + R_train
    U, I = np.shape(R_train)


    test_users, users_cases = to_test([], [], [], [], N_users=N_USERS, N=SIZE, load=True)

    ndcg = 0
    for u in range(0, len(test_users)):
        A = (test_nn(P, Q, test_users[u], users_cases[u]))
        A_scores = model.predict(A).flatten()
        pred_indices = np.argpartition(A_scores, -10)[-10:]
        pred_indices_sorted = np.argsort(A_scores[pred_indices])[::-1]
        pred = users_cases[u][pred_indices[pred_indices_sorted]]

        l = min(10, len(R_test[test_users[u], :].indices))
        vector = np.zeros(10)
        for idx in range(0, 10):
            if R[test_users[u], pred[idx]]>0:
                vector[idx] = 1
        score = dcg_score(vector[0:10])
        ideal = dcg_score(np.ones(l))
        ndcg += score/ideal
    ndcg = ndcg/len(test_users)

    print('NN_ndcg, time:', (time()-tt))
    print('NN_ndcg is', ndcg)


    # tt = time()
    # test_users, users_cases = publics_to_test([], [], [], [], N_users=N_USERS, N=SIZE, load=True)
    # ndcg = 0
    # for u in range(0, len(test_users)):
    #     A = (test_nn(P, Q, test_users[u], users_cases[u]))
    #     A_scores = model.predict(A).flatten()
    #     pred_indices = np.argpartition(A_scores, -10)[-10:]
    #     pred_indices_sorted = np.argsort(A_scores[pred_indices])[::-1]
    #     pred = users_cases[u][pred_indices[pred_indices_sorted]]

    #     l = min(10, len(R_test[test_users[u], :].indices))
    #     vector = np.zeros(10)
    #     for idx in range(0, 10):
    #         if R[test_users[u], pred[idx]]>0:
    #             vector[idx] = 1
    #     score = dcg_score(vector[0:10])
    #     ideal = dcg_score(np.ones(l))
    #     ndcg += score/ideal
    # ndcg = ndcg/len(test_users)

    # print('publics_NN_ndcg, time:', (time()-tt))
    # print('publics_NN_ndcg is', ndcg)    



    with open('results_publics_nn.txt', 'a') as the_file:
        temp = str(sys.argv)  + '; '  + str(ndcg) + '\n'
        the_file.write(temp)


    tt = time()
    ndcg_wrmf = 0

    for u in test_users:
        temp = np.asarray(P[u, :].dot(Q)).flatten()
        train_indices = R_train[u,:].indices
        temp[train_indices] = -np.inf
        indices_top = np.argpartition(temp, -10)[-10:]
        pred_indices = np.argsort(temp[indices_top])[::-1]
        pred = [indices_top[pred_indices]][0]

        l = min(10, len(R_test[u, :].indices))
        vector = np.zeros(10)
        for idx in range(0, 10):
            if R[u, pred[idx]]>0:
                vector[idx] = 1
        score = dcg_score(vector[0:10])
        ideal = dcg_score(np.ones(l))
        ndcg_wrmf += score/ideal

    print('WRMF_ndcg, time', (time()-tt))
    print('WRMF_ndcg is', ndcg_wrmf/len(test_users))




