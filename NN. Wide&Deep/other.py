from utils import *
from factorizations import *

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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


logging.basicConfig(filename='log_main.txt', format='%(asctime)s %(levelname)-8s %(message)s',
                   filemode='w', level=logging.DEBUG)



if (sys.argv[1]).lower()=='create_test':
    N_USERS = int(sys.argv[2])
    tt = time()
    P = pickle_load('z_P.pckl')
    Q = pickle_load('z_Q.pckl')
    R = pickle_load('z_R.pckl')
    R_train = pickle_load('z_R_train.pckl')
    test_users, users_cases = to_test(R, R_train, P, Q, N_users=N_USERS, N=100, load=False)
    print('getting test users, time:', (time()-tt))






if (sys.argv[1]).lower()=='wd':
    EPOCHS = int(sys.argv[2])
    from sklearn.preprocessing import StandardScaler

    class MySequence(Sequence):
        def __init__(self, paths):
            self.grid = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
            self.paths = paths

        def __len__(self):
            return len(paths)

        def __getitem__(self, idx):
            data = np.array(pickle_load(self.paths[idx]))
            # targets_scaled = [(i+1)*(i+1) for i in data[:, -1]]
            targets = data[:, 2]
            targets_scaled = np.ones(len(targets))
            for i in self.grid:
                targets_scaled += (targets >= i).astype('int')
            targets_scaled **= 2
            # targets_scaled = targets


            # user_ind = np.concatenate((np.arange(3,33), [63])) #[P[u, :] | normP] --- 31
            # grp_ind = np.concatenate((np.arange(33,63), [64])) #[Q[:, i] | normQ] --- 31
            # grp_lda_ind = np.concatenate((np.arange(33,63), [64], np.arange(66,366))) #[Q[:, i] | normQ | lda topic vector] --- 331
            
            # uids = data[:, 0].astype('int') 
            # oids = data[:, 1].astype('int')

            to_wide = (data[:, 0:2])
            to_deep = StandardScaler().fit_transform(data[:, 3:])
            return ([to_wide, to_deep], targets_scaled)



    # wide
    wide_input = Input(shape=(2,))
    wide = Dense(1, activation='relu')(wide_input) 

    # deep
    deep_input = Input(shape=(363,))
    deep = Dense(input_dim=363, output_dim=155, activation='relu', kernel_initializer='normal')(deep_input)
    deep = Dropout(0.5)(deep)
    deep = Dense(75, activation='relu', kernel_initializer='normal')(deep)
    deep = Dropout(0.5)(deep)
    deep = Dense(32, activation='relu', kernel_initializer='normal')(deep)
    deep = Dropout(0.5)(deep)
    deep = Dense(1, activation='relu', kernel_initializer='normal')(deep)
    deep = Dropout(0.5)(deep)


    # target
    target_input = concatenate([wide, deep])
    target = Dense(1)(target_input)
    model = keras.Model(inputs=[wide_input, deep_input], outputs=target)
    model.compile(optimizer='Nadam', loss='mse')



    # print(model.summary())
    # plot_model(model, to_file='model_2.png', show_shapes=True, show_layer_names=True)


    path = 'LDA_train/z_X_lda_train_'
    paths = [path + str(i) + '.pckl' for i in range(500)]


    test_users, users_cases = to_test([], [], [], [], N_users=10000, N=100, load=True)



    for k in range(0, EPOCHS):

        model.fit_generator(generator=MySequence(paths), use_multiprocessing=True, workers=24, epochs=1)

        if ((k+1)%1==0) or (k==EPOCHS-1):
                tt = time()
                P = pickle_load('z_P.pckl')
                Q = pickle_load('z_Q.pckl')
                R = pickle_load('z_R.pckl')
                Q_lda = pickle_load('z_Q_lda.pckl')
                R_train = pickle_load('z_R_train.pckl')
                R_test = R - R_train
                U, I = np.shape(R_train)
                ndcg = 0
                for u in range(0, len(test_users)):
                    A, B = (test_wd(P, Q, Q_lda, test_users[u], users_cases[u]))
                    A_scores = model.predict([A, B]).flatten()

                    pred_indices = np.argpartition(A_scores, -10)[-10:]
                    pred_indices_sorted = np.argsort(A_scores[pred_indices])[::-1]
                    pred = users_cases[u][pred_indices[pred_indices_sorted]]

                    l = min(10, len(R_test[test_users[u], :].indices))
                    vector = np.zeros(10)
                    for idx in range(0, l):
                        if R_test[test_users[u], pred[idx]]>0:
                            vector[idx] = 1
                    score = dcg_score(vector[0:10])
                    ideal = dcg_score(np.ones(l))
                    ndcg += score/ideal
                ndcg = ndcg/len(test_users)

                print('EPOCH is', k+1)
                print('WD_ndcg counting, time:', (time()-tt))
                print('WD_ndcg is', ndcg)
                print()

                with open('results_wd.txt', 'a') as the_file:
                    temp = str(sys.argv)  + '; ' + str(k) + '; '  + str(ndcg) + '\n'
                    the_file.write(temp)
                            
                P = None
                Q = None
                R = None
                R_train = None  
                Q_lda = None  


    print('Train time ended')



    P = pickle_load('z_P.pckl')
    Q = pickle_load('z_Q.pckl')
    R = pickle_load('z_R.pckl')
    R_train = pickle_load('z_R_train.pckl')



    tt = time()
    R_test = R - R_train
    U, I = np.shape(R_train)
    ndcg_wrmf = 0
    for u in test_users:
        temp = P[u, :].dot(Q)
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

