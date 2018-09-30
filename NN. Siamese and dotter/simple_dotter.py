from utils import *





NAME = sys.argv[0][:-3].lower()
MODE = sys.argv[1].lower()


if MODE=='train':
    EPOCHS = int(sys.argv[2])

    folder = path_data + 'ONE_FLOW/'
    paths_train = [folder + path for path in listdir(folder) if path[:13]=='z_batch_train']

    P = pickle_load(path_data + 'z_P.pckl')
    Q = pickle_load(path_data + 'z_Q.pckl')
    USER_INFO = pickle_load(path_data + 'z_USER_INFO.pckl')
    ITEM_INFO = pickle_load(path_data + 'z_ITEM_INFO.pckl')

    user_shape = np.shape(P)[1] + np.shape(USER_INFO)[1]
    item_shape = np.shape(Q)[1] + np.shape(ITEM_INFO)[1]

    # user_sizes = [50, 40, 30]
    # grp_sizes = [140, 120, 100, 80, 60, 50, 30]
    user_sizes = [50, 30]
    grp_sizes = [100, 75, 50, 30]    
    dropout_prob = 0.2


    class MySequence(Sequence):
        def __init__(self, paths, P, Q, USER_INFO, ITEM_INFO):
            self.paths = paths
            self.P = P
            self.Q = Q
            self.USER_INFO = USER_INFO
            self.ITEM_INFO = ITEM_INFO


        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            IDX = pickle_load(self.paths[idx])
            
            user_p = self.P[IDX[:, 0], :]
            user_more = self.USER_INFO[IDX[:, 0], :]

            grp_q = self.Q[IDX[:, 1], :]
            grp_more = self.ITEM_INFO[IDX[:, 1], :]

            input_user_1 = StandardScaler().fit_transform(user_p)
            input_user_2 = StandardScaler().fit_transform(user_more)
            # input_user_2 = user_more
            input_user = np.hstack([input_user_1, input_user_2])

            input_grp_1 = StandardScaler().fit_transform(grp_q)
            input_grp_2 = StandardScaler().fit_transform(grp_more)
            # input_grp_2 = grp_more
            input_grp = np.hstack([input_grp_1, input_grp_2])

            target = (IDX[:, 2]>0).astype('int')

            return ([input_user, input_grp], target)


    # user branch
    user_input = Input(shape=(user_shape,))
    user = Dense(user_sizes[0], activation='relu')(user_input)
    user = Dropout(dropout_prob)(user)
    for layer in user_sizes[1:]:
        user = Dense(layer, activation='relu')(user)
        user = Dropout(dropout_prob, name=('user' + str(layer)))(user) 

    # grp branch
    grp_input = Input(shape=(item_shape,))
    grp = Dense(grp_sizes[0], activation='relu')(grp_input)
    grp = Dropout(dropout_prob)(grp)
    for layer in grp_sizes[1:]:
        grp = Dense(layer, activation='relu')(grp)
        grp = Dropout(dropout_prob, name=('grp' + str(layer)))(grp)


    # merge via dot
    dot = merge([user, grp], mode='dot')
    dot = Dense(1, activation='sigmoid')(dot)
    target = Dropout(dropout_prob)(dot)                          


    model = keras.Model(inputs=[user_input, grp_input], outputs=target)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])        





    if str(socket.gethostname())[:6] != 'hadoop':
        print('\nImage done')
        plot_model(model, to_file=('model_' + NAME + '.png'), show_shapes=True, show_layer_names=True)
        sys.exit(0)


    with open('MODELS/' + NAME + '_architecture.json', 'w') as f:
        f.write(model.to_json())

    checkpoint = ModelCheckpoint('CHECKPOINTS/' + NAME +'_weights.{epoch:02d}.hdf5')

    print('\n\nTrain started')
    model.fit_generator(generator=MySequence(paths_train, P, Q, USER_INFO, ITEM_INFO), use_multiprocessing=True, workers=24, callbacks=[checkpoint], epochs=EPOCHS)
    print('\nTrain done;')



if MODE=='check':

    EPOCH = sys.argv[2].lower()
    # AMOUNT = sys.argv[3].lower()

    folder = path_data + 'ONE_FLOW/'
    paths_test = [folder + path for path in listdir(folder) if path[:12]=='z_batch_test']

    P = pickle_load(path_data + 'z_P.pckl')
    Q = pickle_load(path_data + 'z_Q.pckl')
    USER_INFO = pickle_load(path_data + 'z_USER_INFO.pckl')
    ITEM_INFO = pickle_load(path_data + 'z_ITEM_INFO.pckl')


    with open('MODELS/' + NAME + '_architecture.json', 'r') as f:
        model = model_from_json(f.read())

    model.load_weights('CHECKPOINTS/' + NAME + '_weights.'+ EPOCH + '.hdf5')

    print('\n\nModel loaded ok')
    roc_aucs = []
    f1s = []
    for file in paths_test:

        IDX = pickle_load(file)

            
        user_p = P[IDX[:, 0], :]
        user_more = USER_INFO[IDX[:, 0], :]

        grp_q = Q[IDX[:, 1], :]
        grp_more = ITEM_INFO[IDX[:, 1], :]

        input_user_1 = StandardScaler().fit_transform(user_p)
        input_user_2 = StandardScaler().fit_transform(user_more)
        # input_user_2 = user_more
        input_user = np.hstack([input_user_1, input_user_2])

        input_grp_1 = StandardScaler().fit_transform(grp_q)
        input_grp_2 = StandardScaler().fit_transform(grp_more)
        # input_grp_2 = grp_more
        input_grp = np.hstack([input_grp_1, input_grp_2])

        ground_truth = (IDX[:, 2]>0).astype('int')

        predict = model.predict(([input_user, input_grp]))
        
        roc_auc = roc_auc_score(ground_truth, predict)
        roc_aucs.append(roc_auc)
        print('ROC_AUC:', roc_auc)

        f1 = f1_score(ground_truth, (predict>0).astype('int'))
        f1s.append(f1)
        print('F1 score:', f1)
    print('AVG ROC_AUC:', np.mean(roc_aucs)) 
    print('AVG F1 score:', np.mean(f1s))   





if MODE=='embeddings':
    EPOCH = sys.argv[2].lower()

    folder = path_data + 'ONE_FLOW/'
    paths_test = [folder + path for path in listdir(folder) if path[:12]=='z_batch_test']

    P = pickle_load(path_data + 'z_P.pckl')
    Q = pickle_load(path_data + 'z_Q.pckl')
    USER_INFO = pickle_load(path_data + 'z_USER_INFO.pckl')
    ITEM_INFO = pickle_load(path_data + 'z_ITEM_INFO.pckl')


    with open('MODELS/' + NAME + '_architecture.json', 'r') as f:
        model = model_from_json(f.read())

    model.load_weights('CHECKPOINTS/' + NAME + '_weights.'+ EPOCH + '.hdf5')

    new_model = Model(inputs=model.input, outputs=[model.get_layer('user30').output, model.get_layer('grp30').output])
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

    IDX = pickle_load(paths_test[1])

    user_p = P[IDX[:, 0], :]
    user_more = USER_INFO[IDX[:, 0], :]

    grp_q = Q[IDX[:, 1], :]
    grp_more = ITEM_INFO[IDX[:, 1], :]

    input_user_1 = StandardScaler().fit_transform(user_p)
    input_user_2 = StandardScaler().fit_transform(user_more)
    # input_user_2 = user_more
    input_user = np.hstack([input_user_1, input_user_2])

    input_grp_1 = StandardScaler().fit_transform(grp_q)
    input_grp_2 = StandardScaler().fit_transform(grp_more)
    # input_grp_2 = grp_more
    input_grp = np.hstack([input_grp_1, input_grp_2])

    embeddings_user, embeddings_grp = new_model.predict(([input_user, input_grp]))













