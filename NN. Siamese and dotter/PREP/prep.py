from utils import *
import fastparquet
import pyarrow.parquet as pq


def count_interactions():
    interactions = 0
    for file in listdir(path_dataset):
        if file[-8:]=='.parquet':
            current_file = fastparquet.ParquetFile(path_dataset + file)
            current_df = current_file.to_pandas(['post_id'])
            interactions += len(current_df.values)
    print('Interactions total:', interactions) 
    return interactions       


def get_val_counts():
    oid2count = {}
    uid2count = {}
    for file in listdir(path_dataset):
        if file[-8:]=='.parquet':
            current_file = fastparquet.ParquetFile(path_dataset + file)
            current_df = current_file.to_pandas(['post_id', 'uid'])
            
            post_id_values = current_df['post_id'].values
            for post_id in post_id_values:
                temp_oid = int(post_id.split('_')[0])     
                try:
                    oid2count[temp_oid] += 1
                except KeyError:
                    oid2count[temp_oid] = 1
                                  
            uid_values = current_df['uid'].values
            for uid in uid_values:
                try:
                    uid2count[uid] += 1
                except KeyError:
                    uid2count[uid] = 1

    print('Uids counted total:', len(uid2count))                
    print('Oids counted total:', len(oid2count)) 
    return oid2count, uid2count


def get_val_vector_counts(path):
    temp = {}
    for file in listdir(path):
        if file[-8:]=='.parquet':
            current_file = fastparquet.ParquetFile(path + file)
            val_values = current_file.to_pandas(['id']).values
            
            for val in val_values:
                temp_val = int(val)
                try:
                    temp[temp_val] += 1
                except KeyError:
                    temp[temp_val] = 1
    return temp


def filter(oid2count, oids_from_items, uid2count, uids_from_users, filter_threshold=5):
    possible_oids = set(oid2count.keys()) & set(oids_from_items.keys())
    possible_uids = set(uid2count.keys()) & set(uids_from_users.keys())

    oid2ind = {}
    uid2ind = {}
    idx = 0

    for d in possible_oids:
        if oid2count[d]>=filter_threshold:
            oid2ind[d] = idx
            idx += 1

    idx = 0
    for d in possible_uids:
        if uid2count[d]>=filter_threshold:
            uid2ind[d] = idx
            idx += 1        
                
    print('Uids remained:', len(uid2ind))
    print('Oids remained:', len(oid2ind))
    return uid2ind, oid2ind


def get_matrix(path, val2ind):
    temp_vector = {}

    for file in listdir(path):
        if file[-8:]=='.parquet':
            current_df = pq.read_table(path + file).to_pandas()
            
            for row in current_df.values:
                current_val = row[0]
                current_vector = row[1]
                temp_vector[current_val] = current_vector
                

    temp_list = []
    for val in val2ind.keys():
        temp_list.append(temp_vector[val])

    M = np.asarray(temp_list)    
    return M    
    

def count_remaining_interactions():
    interactions_remained = 0
    for file in listdir(path_dataset):
        if file[-8:]=='.parquet':        
            current_file = fastparquet.ParquetFile(path_dataset + file)
            current_df = current_file.to_pandas(['post_id', 'uid'])
            
            for row in current_df.values:
                current_oid = int(row[0].split('_')[0])
                current_uid = int(row[1])
                try:
                    temp = [oid2ind[current_oid], uid2ind[current_uid]]
                    interactions_remained += 1
                except KeyError:
                    pass
                
    print('Interactions remained:', interactions_remained) 
    return interactions_remained           


def get_idx(uid2ind, oid2ind, target_info):
    to_load = ['post_id', 'uid'] + target_info
    w = np.array([0, 0, 10, 10, 30,
                  -10, 0, 0, -10,
                  0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  -10, 0, 0, 0, 0])

    list_idx = []
    list_targets = []
    for file in listdir(path_dataset):
        if file[-8:]=='.parquet':
            current_file = fastparquet.ParquetFile(path_dataset + file)
            current_df = current_file.to_pandas(to_load)
            
            for row in current_df.values:
                current_oid = int(row[0].split('_')[0])
                current_uid = int(row[1])

                target = 0

                if row[3]>0:
                    target = 0.5
                if row[4]>0:
                    target = 1
                if row[5]>0:
                    target = 1
                if row[6]>0:
                    target = 2

                if row[7]>0:
                    target = -2
                if row[10]>0:
                    target = -1 
                if row[22]>0:
                    target = -3  

                # target = row[2:].dot(w)             
                        
                try:
                    temp = [uid2ind[current_uid], oid2ind[current_oid]]
                    list_idx.append(temp)  
                    list_targets.append(target)                        
                except KeyError:
                    pass

    X_train = np.asarray(list_idx)
    Y_train = np.asarray(list_targets)
    pickle_dump(X_train, path_data + 'z_X_train.pckl')
    pickle_dump(Y_train, path_data + 'z_Y_train.pckl')
    print('Percentage of positive interactions:', np.mean((Y_train>0).astype('int')))




def get_more_data(to_load, val2ind, flag):
    dict_temp = {}

    if flag==0:
        to_load = ['post_id'] + to_load
    for file in listdir(path_dataset):
        if file[-8:]=='.parquet':
            current_file = fastparquet.ParquetFile(path_dataset + file)
            current_df = current_file.to_pandas(to_load)
            
            for row in current_df.values:
                if flag==1:
                    current_val = int(row[0])
                else:
                    current_val = int(row[0].split('_')[0])

                try:
                    temp = val2ind[current_val]
                    dict_temp[current_val] = row[1:]       
                except KeyError:
                    pass                      

    temp_list = []
    for val in val2ind.keys():
        temp_list.append(dict_temp[val])
    return np.asarray(temp_list)    
  
    



def create_siamese_train(split, folder, batch_size):
    X_train = pickle_load(path_data + 'z_X_train.pckl')
    Y_train = pickle_load(path_data + 'z_Y_train.pckl')
    X_sorted = X_train[X_train[:,0].argsort()]
    Y_sorted = Y_train[X_train[:,0].argsort()]
    I = max(X_train[:, 1]) + 1

    list_next = []
    previous_user = -1
    for n, curr in enumerate(X_sorted[:, 0]):
        if curr != previous_user:
            list_next.append(n)
            previous_user = curr

    c = 0
    no_pos = 0
    no_neg = 0

    tt = time()
    file_count_train = 0
    file_count_test = 0
    list_train = []
    list_test = []
    for n in range(0, len(list_next)-1):
        current_indices = np.arange(list_next[n], list_next[n+1])

        pos_ind = []
        neg_ind = []
        for ci in current_indices:
            if Y_sorted[ci] > 0:
                pos_ind.append(ci)
            else:
                neg_ind.append(ci)

        current_pos = len(pos_ind)
        current_neg = len(neg_ind)

        if (current_pos>current_neg>0):
            c += 1
            for pos in pos_ind:
                current_uid = X_sorted[pos, 0]
                neg = np.random.choice(neg_ind)
                temp = [current_uid.astype('int'), X_sorted[pos, 1].astype('int'), Y_sorted[pos].astype('int'),
                                           X_sorted[neg, 1].astype('int'), Y_sorted[neg].astype('int')]

                rand = np.random.rand()

                if rand<=split:
                    list_train.append(temp)
                else:
                    list_test.append(temp)
                    # list_test.append([temp[0], temp[1], temp[2]])
                    # list_test.append([temp[0], temp[3], temp[4]])


        if (current_neg>=current_pos>0):
            c += 1
            gen = random.sample(neg_ind, current_pos)
            for enum, pos in enumerate(pos_ind):
                current_uid = X_sorted[pos, 0]
                neg = gen[enum]
                temp = [current_uid.astype('int'), X_sorted[pos, 1].astype('int'), Y_sorted[pos].astype('int'),
                                           X_sorted[neg, 1].astype('int'), Y_sorted[neg].astype('int')]

                rand = np.random.rand()

                if rand<=split:
                    list_train.append(temp)
                else:
                    list_test.append(temp)
                    # list_test.append([temp[0], temp[1], temp[2]])
                    # list_test.append([temp[0], temp[3], temp[4]])                    


        if (current_pos>0) and (current_neg==0):
            no_pos += 1
            for pos in pos_ind:
                current_uid = X_sorted[pos, 0]
                neg = np.random.randint(I)
                while neg in pos_ind:
                    neg = np.random.randint(I)

                    temp = [current_uid.astype('int'), X_sorted[pos, 1].astype('int'), Y_sorted[pos].astype('int'),
                                           X_sorted[neg, 1].astype('int'), Y_sorted[neg].astype('int')]

                    rand = np.random.rand()

                    if rand<=split:
                        list_train.append(temp)
                    else:
                        pass
                        # list_test.append(temp)
                        # list_test.append([temp[0], temp[1], temp[2]])

        if (current_pos>0) and (current_neg==0):
            no_neg += 1

        if (len(list_train)>=batch_size) or (n==(len(list_next)-2)):
            batch = np.asarray(list_train)
            pickle_dump(batch, path_data + folder + 'z_batch_train' + str(file_count_train) + '.pckl')
            list_train = []
            file_count_train += 1

        if (len(list_test)>=batch_size) or (n==(len(list_next)-2)):
            batch = np.asarray(list_test)
            pickle_dump(batch, path_data + folder + 'z_batch_test' + str(file_count_test) + '.pckl')
            list_test = []
            file_count_test += 1

    print('POS+NEG users procentage:', (c+no_pos)/(c + no_pos + no_neg))
    print('Siamese train done; time:', (time()-tt))


def create_simple_train(split, folder, batch_size):
    X_train = pickle_load(path_data + 'z_X_train.pckl')
    Y_train = pickle_load(path_data + 'z_Y_train.pckl')
    L = len(Y_train)

    perm = np.random.permutation(L)
    X_train = X_train[perm, :]
    Y_train = Y_train[perm]

    tt = time()
    file_count_train = 0
    file_count_test = 0
    list_train = []
    list_test = []
    for idx in range(0, L):
        current_uid = X_train[idx, 0].astype('int')
        current_oid = X_train[idx, 1].astype('int')
        current_target = Y_train[idx].astype('int')
        
        temp = [current_uid, current_oid, current_target]
        
        rand = np.random.rand()
        
        if rand<=split:
            list_train.append(temp)
        else:
            list_test.append(temp)
            
        
        if (len(list_train)>=batch_size) or (idx==(L-1)):
            batch = np.asarray(list_train)
            pickle_dump(batch, path_data + folder + 'z_batch_train' + str(file_count_train) + '.pckl')
            list_train = []
            file_count_train += 1
            
        if (len(list_test)>=batch_size) or (idx==(L-1)):
            batch = np.asarray(list_test)
            pickle_dump(batch, path_data + folder + 'z_batch_test' + str(file_count_test) + '.pckl')
            list_test = []
            file_count_test += 1             
    print('Simple train done; time:', (time()-tt))    




print('\n\n', sys.argv)
print('▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ START ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇')
TT = time()


if (sys.argv[1] != 'train'):
    interactions = count_interactions()
    oid2count, uid2count = get_val_counts()          

    uids_from_users = get_val_vector_counts(path_users)
    print('Uids with vectors total:', len(uids_from_users))
    oids_from_items = get_val_vector_counts(path_items)
    print('Oids with vectors total:', len(oids_from_items)) 


    uid2ind, oid2ind = filter(oid2count, oids_from_items, uid2count, uids_from_users, filter_threshold=1)
    ind2oid = {i:o for o, i in oid2ind.items()} 
    ind2uid = {i:u for u, i in uid2ind.items()} 
    pickle_dump(uid2ind, path_data + 'z_uid2ind.pckl')
    pickle_dump(oid2ind, path_data + 'z_oid2ind.pckl')


    P = get_matrix(path_users, uid2ind)
    print('P shape:', P.shape)
    Q = get_matrix(path_items, oid2ind)
    print('Q shape:', Q.shape)
    pickle_dump(P, path_data + 'z_P.pckl')
    pickle_dump(Q, path_data + 'z_Q.pckl')


    interactions_remained = count_remaining_interactions()
    get_idx(uid2ind, oid2ind, target_info)
    print('IDX done')

    USER_INFO = get_more_data(uid_info, uid2ind, flag=1)
    pickle_dump(USER_INFO, path_data + 'z_USER_INFO.pckl')
    print('USER_INFO shape:', USER_INFO.shape) 

    ITEM_INFO = get_more_data(oid_info, oid2ind, flag=0)
    pickle_dump(ITEM_INFO, path_data + 'z_ITEM_INFO.pckl')
    print('ITEM INFO shape:', ITEM_INFO.shape)


if (sys.argv[1]=='train'):
    uid2ind = pickle_load(path_data + 'z_uid2ind.pckl')
    oid2ind = pickle_load(path_data + 'z_oid2ind.pckl')    
    interactions_remained = count_remaining_interactions()
    number_of_files = 50
    batch_size = interactions_remained/number_of_files
    create_siamese_train(0.8, 'SIAMESE_TRAIN/', batch_size)
    create_simple_train(0.8, 'ONE_FLOW/', batch_size)


print('Total time:', (time()-TT))
print('▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ DONE ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇')















