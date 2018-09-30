from utils import *



def WRMF(R_train, factors=30, confidence=8, lmbda=1, iter_wrmf=10, load=False):
    if load==True:
        P = pickle_load('z_P.pckl')
        Q = pickle_load('z_Q.pckl')
        
    if load==False:
        model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=lmbda, 
                                                     iterations=iter_wrmf, 
                                                     calculate_training_loss=False, num_threads=0)
        R_train_a = (confidence * R_train).astype('double')
        model.fit(R_train_a.T)
        P = model.user_factors
        Q = (model.item_factors).T 
        pickle_dump(P, 'z_P.pckl')
        pickle_dump(Q, 'z_Q.pckl') 
    return np.asarray(P), np.asarray(Q) 


def iter(R_train, S, P, Q, lmbda, alpha, learning_rate, CURR, verbose=False):   
    """One iteration of non-batched gradient descent for SBMF. Vectirized"""

    grad_P = P.dot(Q.dot(Q.T)) - R_train.dot(Q.T) 
    grad_P += lmbda*P

    grad_Q = (P.T.dot(P)).dot(Q) - (R_train.T.dot(P)).T
    grad_Q += alpha*(Q.dot(Q.T)).dot(Q) - alpha*(sparse.csr_matrix(Q).dot(S)) 
    grad_Q += lmbda*Q

    if (CURR>12) and (abs(np.mean(grad_Q))>0.7):
        learning_rate *= 1/2


    P = P - learning_rate*grad_P
    Q = Q - learning_rate*grad_Q

    if verbose in [True, 'true', '1', 't', 'y']:
        print('P:::', '% 6.8f' % np.min(grad_P), '% 6.8f' % np.max(grad_P), '% 6.8f' % np.mean(grad_P), '% 6.8f' % np.mean(P))
        print('Q:::', '% 6.8f' % np.min(grad_Q), '% 6.8f' % np.max(grad_Q), '% 6.8f' % np.mean(grad_Q), '% 6.8f' % np.mean(Q))
    return P, Q


def SBMF(R_train, S, factors=30, confidence=8, lmbda=1, alpha=1, learning_rate=0.00001, iter_sbmf=10, verbose=True, load=False):
    if load==True:
        P = pickle_load('z_P.pckl')
        Q = pickle_load('z_Q.pckl')

    if load==False:
        if verbose in [True, 'true', '1', 't', 'y']:
            print('parameters:::', sys.argv[1:])

        R_train_a = (confidence * R_train).astype('double')
        S_a = (confidence * S).astype('double')

        U, I = np.shape(R_train)
        P = np.random.rand(U, factors) *0.1
        Q = np.random.rand(factors, I) *0.1

        decay = 1
        for k in range(0, iter_sbmf): 
            tt = time()
            if verbose in [True, 'true', '1', 't', 'y']:
                print ('iteration %d of %d' % (k+1, iter_sbmf))
            if k<15:
                decay = 24   
            if k>150:
                decay = np.log(k)  #np.sqrt(k-740)

            if k>300:
                decay = np.log(k-100)    

            try:
                P, Q = iter(R_train_a, S_a, P, Q, lmbda, alpha, learning_rate/decay, k, verbose)
                logging.debug('iteration %d done \n' %(k))
            except KeyboardInterrupt:
                break


            if verbose in [True, 'true', '1', 't', 'y']:
                print('time for iteration',  '% 6.2f' % (time()-tt))

        pickle_dump(P, 'z_P.pckl')
        pickle_dump(Q, 'z_Q.pckl') 
    return np.asarray(P), np.asarray(Q)


def iter_b(R_train, S, P, Q, B, C, oU, oI, mu, lmbda, alpha, learning_rate, verbose=False):   
    """One iteration of non-batched gradient descent for biased SBMF. Vectirized"""
    U, I = np.shape(R_train)

    grad_P = P.dot(Q.dot(Q.T)) - R_train.dot(Q.T) 
    grad_P += B.dot(oI.dot(Q.T)) + oU.dot(C.dot(Q.T)) + mu*oU.dot(oI.dot(Q.T))
    grad_P += lmbda*P

    grad_Q = (P.T.dot(P)).dot(Q) - (R_train.T.dot(P)).T 
    grad_Q += (P.T.dot(B)).dot(oI) + (P.T.dot(oU)).dot(C) + mu*(P.T.dot(oU)).dot(oI)
    grad_Q += alpha*(Q.dot(Q.T)).dot(Q) - alpha*(sparse.csr_matrix(Q).dot(S)) 
    grad_Q += lmbda*Q

    grad_B = P.dot(Q.dot(oI.T)) + I*B + oU.dot(C.dot(oI.T)) + I*mu - R_train.dot(oI.T) + lmbda*B
    grad_C = (oU.T.dot(P)).dot(Q) + (oU.T.dot(B)).dot(oI) + U*C + U*mu - (R_train.T.dot(oU)).T + lmbda*C

    P = P - learning_rate*grad_P
    Q = Q - learning_rate*grad_Q
    B = B - learning_rate*grad_B
    C = C - learning_rate*grad_C
       
    if verbose in [True, 'true', '1', 't', 'y']:
        print('P:::', '% 6.8f' % np.min(grad_P), '% 6.8f' % np.max(grad_P), '% 6.8f' % np.mean(grad_P), '% 6.8f' % np.mean(P))
        print('Q:::', '% 6.8f' % np.min(grad_Q), '% 6.8f' % np.max(grad_Q), '% 6.8f' % np.mean(grad_Q), '% 6.8f' % np.mean(Q))
        print('B:::', '% 6.8f' % np.min(grad_B), '% 6.8f' % np.max(grad_B), '% 6.8f' % np.mean(grad_B), '% 6.8f' % np.mean(B))
        print('C:::', '% 6.8f' % np.min(grad_C), '% 6.8f' % np.max(grad_C), '% 6.8f' % np.mean(grad_C), '% 6.8f' % np.mean(C))

    grad_P = None
    grad_Q = None
    grad_B = None
    grad_C = None
    
    return P, Q, B, C


def SBMF_b(R_train, S, factors=30, confidence=8, lmbda=1, alpha=1, learning_rate=0.00001, iter_sbmf=10, verbose=True, load=False):

    if load==True:
        P = pickle_load('z_P.pckl')
        Q = pickle_load('z_Q.pckl')
        B = pickle_load('z_B.pckl')
        C = pickle_load('z_C.pckl')

    if load==False:
        if verbose in [True, 'true', '1', 't', 'y']:
            print('parameters:::', sys.argv[1:])

        R_train_a = (confidence * R_train).astype('double')
        S_a = (1 * S).astype('double')

        U, I = np.shape(R_train)
        P = np.random.rand(U, factors) *0.01
        Q = np.random.rand(factors, I) *0.01
        B = np.random.rand(U, 1) *0.005
        C = np.random.rand(1, I) *0.005
        mu = np.mean(R_train)
        oU = np.ones((U, 1))
        oI = np.ones((1, I))

        decay = 1
        for k in range(0, iter_sbmf): 
            tt = time()

            if verbose in [True, 'true', '1', 't', 'y']:
                print ('iteration %d of %d' % (k+1, iter_sbmf))

            if k>70:
                decay = np.sqrt(k-60)

            P, Q, B, C = iter_b(R_train, S, P, Q, B, C, oU, oI, mu, lmbda, alpha, learning_rate/decay, verbose)
            logging.debug('iteration %d done \n' %(k))

            if verbose in [True, 'true', '1', 't', 'y']:
                print('time for iteration',  '% 6.2f' % (time()-tt))

        pickle_dump(P, 'z_P.pckl')
        pickle_dump(Q, 'z_Q.pckl') 
        pickle_dump(B, 'z_B.pckl')
        pickle_dump(C, 'z_C.pckl') 

    return np.asarray(P), np.asarray(Q), np.asarray(B), np.asarray(C)



def dcg_score(y_score):
    gain = y_score
    discounts = np.log2(np.arange(len(y_score)) + 2)
    return np.sum(gain / discounts)


def ndcg(R, R_train, P, Q, test_users, at=10):
    U, I = np.shape(R_train)    
    R_test = R - R_train

    ndcg = 0
    for u in test_users:
        temp = P[u, :].dot(Q)
        train_indices = R_train[u,:].indices
        temp[train_indices] = -np.inf
        indices_top = np.argpartition(temp, -at)[-at:]
        pred_indices = np.argsort(temp[indices_top])[::-1]
        pred = [indices_top[pred_indices]][0]   
        
        l = min(at, len(R_test[u, :].indices))
        vector = np.zeros(at)
        for idx in range(0, at):
            if R[u, pred[idx]]>0:
                vector[idx] = 1
        score = dcg_score(vector[0:at])
        ideal = dcg_score(np.ones(l))
        ndcg += score/ideal

    return ndcg/len(test_users)


def ndcg_b(R, R_train, P, Q, C, test_users, at=10):
    U, I = np.shape(R_train)    
    R_test = R - R_train

    ndcg = 0
    for u in test_users:
        temp = P[u, :].dot(Q).flatten() + C.flatten()
        train_indices = R_train[u,:].indices
        temp[train_indices] = -np.inf
        indices_top = np.argpartition(temp, -at)[-at:]
        pred_indices = np.argsort(temp[indices_top])[::-1]
        pred = [indices_top[pred_indices]][0]   
        
        l = min(at, len(R_test[u, :].indices))
        vector = np.zeros(at)
        for idx in range(0, at):
            if R[u, pred[idx]]>0:
                vector[idx] = 1
        score = dcg_score(vector[0:at])
        ideal = dcg_score(np.ones(l))
        ndcg += score/ideal

    return ndcg/len(test_users)


def acc(R, R_train, P, Q, test_users, at=10, N=100):
    U, I = np.shape(R_train)
    R_test = R - R_train

    acc = 0
    for u in test_users:
        temp = P[u, :].dot(Q)
        train_indices = R_train[u,:].indices
        temp[train_indices] = -np.inf
        indices_top = np.argpartition(temp, -N)[-N:]
        
        count = 0
        for idx in indices_top:
            if R[u, idx]>0:
                count += 1
        acc += count/len(R_test[u, :].indices)        

    return acc/len(test_users)



def acc_b(R, R_train, P, Q, C, test_users, at=10, N=100):
    U, I = np.shape(R_train)
    R_test = R - R_train

    acc = 0
    for u in test_users:
        temp = P[u, :].dot(Q).flatten() + C.flatten()
        train_indices = R_train[u,:].indices
        temp[train_indices] = -np.inf
        indices_top = np.argpartition(temp, -N)[-N:]
        
        count = 0
        for idx in indices_top:
            if R[u, idx]>0:
                count += 1
        acc += count/len(R_test[u, :].indices)        

    return acc/len(test_users)





def ndcg_ideal(R, R_train, P, Q, test_users, N=100):
    U, I = np.shape(R_train)
    R_test = R - R_train
    
    ndcg = 0
    for u in test_users:
        temp = P[u, :].dot(Q)
        train_indices = R_train[u,:].indices
        temp[train_indices] = -np.inf
        indices_top = np.argpartition(temp, -N)[-N:]
        pred_indices = np.argsort(temp[indices_top])[::-1]
        pred = [indices_top[pred_indices]][0]

        test_100 = R[u, indices_top].nnz
        test_all = min(R[u].nnz, 10)
        score = dcg_score(np.ones(test_100))
        ideal = dcg_score(np.ones(test_all))
        ndcg += score/ideal
    
    return ndcg/len(test_users)



def ndcg_ideal_b(R, R_train, P, Q, C, test_users, N=100):
    U, I = np.shape(R_train)
    R_test = R - R_train
    
    ndcg = 0
    for u in test_users:
        temp = P[u, :].dot(Q).flatten() + C.flatten()
        train_indices = R_train[u,:].indices
        temp[train_indices] = -np.inf
        indices_top = np.argpartition(temp, -N)[-N:]
        pred_indices = np.argsort(temp[indices_top])[::-1]
        pred = [indices_top[pred_indices]][0]

        test_100 = R[u, indices_top].nnz
        test_all = min(R[u].nnz, 10)
        score = dcg_score(np.ones(test_100))
        ideal = dcg_score(np.ones(test_all))
        ndcg += score/ideal
    
    return ndcg/len(test_users) 



