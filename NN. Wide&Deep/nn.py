from keras.models import Sequential
from keras import layers
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
import numpy as np
from utils import *
from time import time
from keras import regularizers



def dcg_score(y_score):
    gain = y_score
    discounts = np.log2(np.arange(len(y_score)) + 2)
    return np.sum(gain / discounts)


model = Sequential()

model.add(layers.Dense(units=512, activation='relu', kernel_initializer='random_normal', use_bias=True, input_dim=63, 
                       kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(units=256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])


path = 'z_X_train_'
paths = [path + str(i) + '.pckl' for i in range(150)]

class MySequence(Sequence):

    def __init__(self, paths):
        self.grid = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
        self.paths = paths

    def __len__(self):
        return len(paths)

    def __getitem__(self, idx):
        data = np.array(pickle_load(self.paths[idx]))
        targets = data[:, -1]
        targets_scaled = np.ones(len(targets))
        for i in self.grid:
            targets_scaled += (targets >= i).astype('int')
        targets_scaled **= 2
        features = np.delete(data, [0, 1, data.shape[1] - 1], axis=1)
        return (features, targets_scaled)

checkpoint = ModelCheckpoint('checkpoints_square/weights.{epoch:02d}.hdf5')
#for epoch in range(20):
model.fit_generator(generator=MySequence(paths), use_multiprocessing=True, workers=24, callbacks=[checkpoint], epochs=2)
model.save('checkpoints_square/deep_model_epochs_2')


P = pickle_load('z_P.pckl')
Q = pickle_load('z_Q.pckl')
R = pickle_load('z_R.pckl')
R_train = pickle_load('z_R_train.pckl')


tt = time()
test_users, users_cases = to_test(R, R_train, P, Q, N_users=10000, N=100, load=False)
print('getting test users, time:', (time()-tt))


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



R_test = R - R_train
U, I = np.shape(R_train)
ndcg = 0
for u in range(0, len(test_users)):
    A = sparse.csr_matrix(test_to_dense(P, Q, test_users[u], users_cases[u]))
    A_scores = model.predict([A]).flatten()
#     A_scores = np.random.rand((100))
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

print('NN_ndcg counting, time:', (time()-tt))
print('NN_ndcg is', ndcg)


print()

with open('results_nn.txt', 'a') as the_file:
    temp = str(sys.argv)  + '; '  + str(ndcg) + '\n'
    the_file.write(temp)


