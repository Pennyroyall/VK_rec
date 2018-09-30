from utils import *
from factorizations import *
from fastFM import als
from fastFM import sgd
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(filename='log_main.txt', format='%(asctime)s %(levelname)-8s %(message)s',
				   filemode='w', level=logging.DEBUG)


path_R='z_R.pckl'
path_R_train='z_R_train.pckl'
path_S = 'z_S.pckl'

tt = time()
R_train = pickle_load(path_R_train)
logging.info( u'R_train ok \n')
R = pickle_load(path_R)
S = pickle_load(path_S)
print('Loading, time:', (time()-tt))



if (sys.argv[1]).lower()=='wrmf':

	FACTORS = int(sys.argv[2])
	CONFIDENCE = int(sys.argv[3])
	LMBDA = float(sys.argv[4])
	ITER = int(sys.argv[5])

	# R = R[:, 0:150000]
	# R_train = R_train[:, 0:150000]
	gr = [0.05, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
	R_train_d = grid_bin(R_train, gr)

	P, Q = WRMF(R_train_d, factors=FACTORS, confidence=CONFIDENCE, lmbda=LMBDA, iter_wrmf=ITER, load=False)

	tt = time()
	# test_users, users_cases = to_test(R, R_train, P, Q, N_users=1000, N=100, load=False)
	test_users = to_test_simple(R, R_train, N_users=100000, load=False)  
	print('Getting test users, time:', (time()-tt))

	tt = time()
	ndcg = ndcg(R, R_train, P, Q, test_users, at=10)
	print('WRMF_ndcg, time', (time()-tt))
	print('WRMF_ndcg is', ndcg)



	# tt = time()
	# ndcg_i = ndcg_ideal(R, R_train, P, Q, test_users, N=NN)
	# print('WRMF_ndcg_i, time', (time()-tt))
	# print('WRMF_ndcg_i is', ndcg_i)


	# for NN in [50, 100, 200, 500, 1000]:
	# 	tt = time()
	# 	ndcg_i = ndcg_ideal(R, R_train, P, Q, test_users, N=NN)
	# 	print('WRMF_ndcg_i, time', (time()-tt))
	# 	print('WRMF_ndcg_i is', ndcg_i)
	# 	with open('results_N_vs_ideal_wrmf.txt', 'a') as the_file:
	# 		temp = str(NN) + '; '  + str(ndcg_i)  + '\n'
	# 		the_file.write(temp)

	# tt = time()
	# acc = acc(R, R_train, P, Q, test_users, at=10, N=100)
	# print('WRMF accuracy at time', (time()-tt))
	# print('WRMF accuracy is', acc)
	# print()

	with open('results_wrmf.txt', 'a') as the_file:
		temp = str(sys.argv) + '; '  + str(ndcg) +  '\n' # '; ' + str(ndcg_i) + '; ' + str(acc)  + '\n'
		the_file.write(temp)




if (sys.argv[1]).lower()=='sbmf':

	FACTORS = int(sys.argv[2])
	LMBDA = float(sys.argv[3])
	ALPHA = float(sys.argv[4])
	CONFIDENCE = float(sys.argv[5])
	ITER = int(sys.argv[6])
	LR = float(sys.argv[7])
	VERBOSE = str(sys.argv[8])

	gr = [0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
	R_train_d = grid_bin(R_train, gr)
	
	P, Q = SBMF(R_train_d, S, factors=FACTORS, confidence=CONFIDENCE, lmbda=LMBDA, alpha=ALPHA, learning_rate=LR, iter_sbmf=ITER, verbose=VERBOSE, load=False)

	tt = time()
	# test_users, users_cases = to_test(R, R_train, P, Q, N_users=1000, N=100, load=False)
	test_users = to_test_simple(R, R_train, N_users=100000, load=False)  
	print('Getting test users, time:', (time()-tt))

	tt = time()
	ndcg = ndcg(R, R_train, P, Q, test_users, at=10)
	print('SBMF_ndcg, time', (time()-tt))
	print('SBMF_ndcg is', ndcg)

	# tt = time()
	# ndcg_i = ndcg_ideal(R, R_train, P, Q, test_users, N=100)
	# print('SBMF ndcg_i, time', (time()-tt))
	# print('SBMF ndcg_i is', ndcg_i)

	# tt = time()
	# acc = acc(R, R_train, P, Q, test_users, at=10, N=100)
	# print('SBMF accuracy at time', (time()-tt))
	# print('SBMF accuracy is', acc)
	# print()

	with open('results_sbmf.txt', 'a') as the_file:
		temp = str(sys.argv) + '; '  + str(ndcg) + '\n' # '; ' + str(ndcg_i) + '; ' + str(acc)  + '\n'
		the_file.write(temp)




if (sys.argv[1]).lower()=='sbmf_b':


	FACTORS = int(sys.argv[2])
	LMBDA = float(sys.argv[3])
	ALPHA = float(sys.argv[4])
	CONFIDENCE = float(sys.argv[5])
	ITER = int(sys.argv[6])
	VERBOSE = str(sys.argv[7])

	R = R[:, 0:150000]
	R_train = R_train[:, 0:150000]
	S = S[0:150000, 0:150000]


	# gr = [0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
	# R_train_d = grid_bin(R_train, gr)
	R_train_d = R_train

	P, Q, B, C = SBMF_b(R_train_d, S, factors=FACTORS, confidence=CONFIDENCE, lmbda=LMBDA, alpha=ALPHA, learning_rate=0.0000001, iter_sbmf=ITER, verbose=VERBOSE, load=False)

	tt = time()
	# test_users, users_cases = to_test(R, R_train, P, Q, N_users=1000, N=100, load=False)
	test_users = to_test_simple(R, R_train, N_users=10000, load=False)         
	print('Getting test users, time:', (time()-tt))

	tt = time()
	ndcg = ndcg_b(R, R_train, P, Q, C, test_users, at=10)
	print('SBMF_b ndcg, time', (time()-tt))
	print('SBMF_b ndcg is', ndcg)

	# for NN in [50, 100, 200, 500, 1000]:
	# 	tt = time()
	# 	ndcg_i = ndcg_ideal_b(R, R_train, P, Q, C, test_users, N=NN)
	# 	print('SBMF_ndcg_i, time', (time()-tt))
	# 	print('SBMF_ndcg_i is', ndcg_i)
	# 	with open('results_N_vs_ideal_sbmf_b.txt', 'a') as the_file:
	# 		temp = str(NN) + '; '  + str(ndcg_i)  + '\n'
	# 		the_file.write(temp)


	# tt = time()
	# ndcg_i = ndcg_ideal_b(R, R_train, P, Q, C, test_users, N=100)
	# print('SBMF_b ndcg_i, time', (time()-tt))
	# print('SBMF_b ndcg_i is', ndcg_i)





	# tt = time()
	# acc = acc_b(R, R_train, P, Q, C, test_users, at=10, N=100)
	# print('SBMF_b accuracy at time', (time()-tt))
	# print('SBMF_b accuracy is', acc)
	# print()

	with open('results_sbmf_b.txt', 'a') as the_file:
		temp = str(sys.argv) + '; '  + str(ndcg) +  '\n' #'; ' + str(ndcg_i) + '; ' + str(acc)  + '\n'
		the_file.write(temp)




try:
	S = None
	R_train_d = None
except:
	pass






if (sys.argv[1]).lower()=='fm':

	FACTORS = int(sys.argv[2])
	LMBDA_1 = float(sys.argv[3])
	LMBDA_2 = float(sys.argv[4])
	ITER = int(sys.argv[5])
	TOPN = int(sys.argv[6])

	G_train, Y_train = convert_to(R_train, load=True)
	G_test = pickle_load('z_G_test.pckl')
	Y_test = pickle_load('z_Y_test.pckl')
		
	Y_train_d = (Y_train>=0.01).astype('int') + (Y_train>=0.02).astype('int') + (Y_train>=0.025).astype('int')
	Y_train_d += (Y_train>=0.03).astype('int') + (Y_train>=0.04).astype('int') + (Y_train>=0.05).astype('int')
	Y_train_d += (Y_train>=0.06).astype('int') + (Y_train>0).astype('int')

	# shuf = np.random.permutation(np.shape(G_train)[0])
	
	tt = time()
	print('FM started')
	# fm = sgd.FMRegression(n_iter=ITER*1000000, init_stdev=0.01, rank=FACTORS, step_size=0.001, l2_reg_w=LMBDA_1, l2_reg_V=LMBDA_2)
	fm = als.FMRegression(n_iter=0, init_stdev=0.01, rank=30, l2_reg_w=LMBDA_1, l2_reg_V=LMBDA_2)


	U, I = np.shape(R)
	sample = random.sample(range(U), 100000)
	fm.fit(G_train[sample, :], (Y_train[sample]).flatten())

	for i in range(0, ITER):
		ttt = time()
		sample = random.sample(range(U), 1000000)
		fm.fit(G_train[sample, :], (10 + 5*Y_train[sample]).flatten(), 1)
		

		# sc = np.sqrt(mean_squared_error(fm.predict(G_train[sample, :]), Y_train[sample]))
		# print('rmse_train:', sc)

		# sc = np.sqrt(mean_squared_error(fm.predict(G_test[0:1000000, :]), Y_test[0:1000000]))
		# print('rmse_test:', sc)

		print ('iteration %d of %d, time: %f' % (i+1, ITER, (time()-ttt)))


	# fm.fit(G_train[shuf], (1 + 1*Y_train_d[shuf]).flatten())
	print('FM learning, time:', (time()-tt))


	pickle_dump(fm, 'z_FM.pckl')
	#fm = pickle_load('z_FM.pckl')
	
	G_train = None
	Y_train = None

	# tt = time()
	# test_users = to_test_simple(R, R_train, N_users=100, load=False) 
	# print('Getting test users, time:', (time()-tt))

	# tt = time()
	# R_test = R - R_train
	# U, I = np.shape(R_train)
	# ndcg = 0
	# for u in range(0, len(test_users)):

	# 	user_case = list(np.arange(I))

	# 	for i in range(0, len(R_train[test_users[u], :].indices)):
	# 		user_case.remove(R_train[test_users[u], :].indices[i]) 

	# 	A = test_to_onehot(U, I, test_users[u], user_case)

	# 	# A = test_to_onehot(U, I, test_users[u], users_cases[u])
	# 	A_scores = fm.predict(A).flatten()
	# #     A_scores = np.random.rand((100))
	# 	pred_indices = np.argpartition(A_scores, -10)[-10:]
	# 	pred_indices_sorted = np.argsort(A_scores[pred_indices])[::-1]
	# 	pred = np.asarray(user_case)[pred_indices[pred_indices_sorted]]
	# 	# pred = users_cases[u][pred_indices[pred_indices_sorted]]

	# 	l = min(10, len(R_test[test_users[u], :].indices))
	# 	vector = np.zeros(10)
	# 	for idx in range(0, 10):
	# 		if R[test_users[u], pred[idx]]>0:
	# 			vector[idx] = 1
	# 	score = dcg_score(vector[0:10])
	# 	ideal = dcg_score(np.ones(l))
	# 	ndcg += score/ideal   
	# ndcg = ndcg/len(test_users)

	# print('FM_ndcg counting, time:', (time()-tt))
	# print('FM_ndcg is', ndcg)




	tt = time()
	P = pickle_load('z_P.pckl')
	Q = pickle_load('z_Q.pckl')	
	
	test_users, users_cases = to_test(R, R_train, P, Q, N_users=1000, N=TOPN, load=False)
	print('Getting test users, time:', (time()-tt))


	tt = time()
	ndcg = ndcg(R, R_train, P, Q, test_users, at=10)
	print('WRMF_ndcg, time', (time()-tt))
	print('WRMF_ndcg is', ndcg)


	tt = time()
	R_test = R - R_train
	U, I = np.shape(R_train)
	ndcg = 0
	for u in range(0, len(test_users)):
		A = test_to_onehot(U, I, test_users[u], users_cases[u])
		A_scores = fm.predict(A).flatten()
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

	print('FM_ndcg counting, time:', (time()-tt))
	print('FM_ndcg is', ndcg)


	print()

	with open('results_fm.txt', 'a') as the_file:
		temp = str(sys.argv) + '; '  + str(ndcg)  + '\n'
		the_file.write(temp)


