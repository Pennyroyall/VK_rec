from utils import *


NAME = sys.argv[0][:-3].lower()
MODE = sys.argv[1].lower()




if MODE=='train':
    EPOCHS = int(sys.argv[2])

    try:
        P = pickle_load(path_data + 'z_P.pckl')
        Q = pickle_load(path_data + 'z_Q.pckl')
        # USER_INFO = pickle_load(path_data + 'z_USER_INFO.pckl')
        # ITEM_INFO = pickle_load(path_data + 'z_ITEM_INFO.pckl')

        user_shape = np.shape(P)[1] 
        item_shape = np.shape(Q)[1]
    except FileNotFoundError:
        print('\n\nPrinting image mode') 
        user_shape = 30
        item_shape = 30   

    user_sizes = [30]
    shared_sizes = [30]    
    dropout_prob = 0.2


    # Format of inputs
    class MySequence(Sequence):
        def __init__(self, paths, P, Q):
            self.paths = paths
            self.P = P
            self.Q = Q

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            IDX = pickle_load(self.paths[idx])
            
            #User input
            user_p = self.P[IDX[:, 0], :]
            input_user_1 = StandardScaler().fit_transform(user_p)

            #Positive interaction
            positive_grp_q = self.Q[IDX[:, 1], :]
            input_positive_grp_1 = StandardScaler().fit_transform(positive_grp_q)

            #Negative interaction
            negative_grp_q = self.Q[IDX[:, 3], :]           
            input_negative_grp_1 = StandardScaler().fit_transform(negative_grp_q)
          

            #Targets and placeholder (for triplet_loss)
            target_positive = (IDX[:, 2]>0).astype('int')
            target_negative = (IDX[:, 4]>0).astype('int')
            placeholder = np.zeros_like(target_positive) + 1

            return ([input_positive_grp_1, input_negative_grp_1, input_user_1], [target_positive, target_negative, placeholder])



    def identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)


    def bpr_triplet_loss_2(X):

        positive_item_latent, negative_item_latent, user_latent = X

        # BPR loss
        loss = 1.0 - K.sigmoid(
            K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
            K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))
        return loss   
   

    def dot_sigmoid(X):
        (user_embedding, item_embedding) = X
        return K.sigmoid(K.sum(user_embedding * item_embedding, axis=-1, keepdims=True))



    # shared branch
    shared_input = Input(shape=(item_shape,))
    shared = Dense(shared_sizes[0], activation='relu')(shared_input)
    shared_out = Dropout(dropout_prob, name='grp_30')(shared)
    shared_model = Model(shared_input, shared_out)

    # user branch
    user_input = Input(shape=(user_shape,))
    user = Dense(user_sizes[0], activation='relu')(user_input)
    user = Dropout(dropout_prob)(user)

    # grp inputs
    positive_input = Input(shape=(item_shape,))                       
    negative_input = Input(shape=(item_shape,)) 

    # Applying shared model
    positive_embedding = shared_model(positive_input)
    negative_embedding = shared_model(negative_input)

    # All losses
    positive_output = merge([user, positive_embedding], mode=dot_sigmoid, output_shape=(1, ), name='positive_output')
    negative_output = merge([user, negative_embedding], mode=dot_sigmoid, output_shape=(1, ), name='negative_output')

    target = merge([positive_embedding, negative_embedding, user],
                    mode=bpr_triplet_loss_2, name='triplet', output_shape=(1, ))                      

    # Params
    losses = {'positive_output' : 'binary_crossentropy', 
              'negative_output' : 'binary_crossentropy', 
              'triplet' : identity_loss}
    losses_weights = [1, 1, 8]      

    model = keras.Model(inputs=[positive_input, negative_input, user_input], outputs=[positive_output, negative_output, target])
    model.compile(optimizer='adam', metrics=['accuracy'], 
                  loss=losses, loss_weights=losses_weights)        



    if str(socket.gethostname())[:6] != 'hadoop':
        print('\nImage done')
        plot_model(model, to_file=('model_' + NAME + '.png'), show_shapes=True, show_layer_names=True)
        sys.exit(0)


    with open('MODELS/' + NAME + '_architecture.json', 'w') as f:
        f.write(model.to_json())
    checkpoint = ModelCheckpoint('CHECKPOINTS/' + NAME +'_weights.{epoch:02d}.hdf5')

    folder = path_data + 'SIAMESE_TRAIN/'
    paths_train = [folder + path for path in listdir(folder) if path[:13]=='z_batch_train']

    print('\n\nTrain started')
    model.fit_generator(generator=MySequence(paths_train, P, Q), use_multiprocessing=True, workers=24, callbacks=[checkpoint], epochs=EPOCHS)
    print('\nTrain done;')



if MODE=='check':

    EPOCH = sys.argv[2].lower()

    folder = path_data + 'SIAMESE_TRAIN/'
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

        #User input
        user_p = P[IDX[:, 0], :]
        input_user_1 = StandardScaler().fit_transform(user_p)

        #Positive interaction
        positive_grp_q = Q[IDX[:, 1], :]
        input_positive_grp_1 = StandardScaler().fit_transform(positive_grp_q)

        #Negative interaction
        negative_grp_q = Q[IDX[:, 3], :]           
        input_negative_grp_1 = StandardScaler().fit_transform(negative_grp_q)            

        #Targets combining
        target_positive = (IDX[:, 2]>0).astype('int')
        target_negative = (IDX[:, 4]>0).astype('int')
        target_all = np.asarray([target_positive, target_negative]).flatten()
        target_rev = np.asarray([target_negative, target_positive]).flatten()

        #Predictions
        predict_pos = model.predict(([input_positive_grp_1, input_positive_grp_1, input_user_1]))[0].flatten()
        predict_neg = model.predict(([input_negative_grp_1, input_negative_grp_1, input_user_1]))[0].flatten()
        predict_all = np.asarray([predict_pos, predict_neg]).flatten()
        predict_rev = np.asarray([predict_neg, predict_pos]).flatten()
        
        #Scoring
        ra_all = roc_auc_score(target_all, predict_all)
        ra_rev = roc_auc_score(target_rev, predict_rev)
        print('ROC_AUC:', ra_all)

        # print('ASD:', predict_pos[10], predict_neg[10])
        roc_aucs.append(ra_all)
        if ra_all - ra_rev > 0.001:
            print("??????????")
    
    print('AVG ROC_AUC:', np.mean(roc_aucs)) 







