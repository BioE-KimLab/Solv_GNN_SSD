import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    device = "/gpu:0"
else:
    device = "/cpu:0"

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras import layers
from gnn import *
import nfp
import json 
import sys

from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold, train_test_split

def add_student_DBs(csv_name):
    data2 = pd.read_csv(csv_name)
    data2 = data2[['can_smiles_solute', 'can_smiles_solvent', 'DGsolv']]
    data2['index'] = list(range(len(data2)))
    data2['Exp_or_QM'] = 'QM'
    index_train_valid, index_test, dummy_train_valid, dummy_test = train_test_split(data2['index'], data2['index'], 
                                                                                    test_size = 0.1, random_state = args.random_seed)
    test_QM = data2[data2['index'].isin(index_test)]
    train_valid = data2[data2['index'].isin(index_train_valid)]

    kfold = KFold(n_splits = 10, shuffle = True, random_state = args.random_seed)
    train_valid_split = list(kfold.split(train_valid))[args.fold_number]
    train_index, valid_index = train_valid_split
    train_QM = train_valid.iloc[train_index]
    valid_QM = train_valid.iloc[valid_index]

    return train_QM, valid_QM, test_QM

def main(args):
    #print(np.__version__)
    #data = pd.read_csv('CombiSolvQM_can_smiles+atom_bond_type_analyzed.csv')
    #####################################
    data = pd.read_csv('220607_exp_solu_DB_v2.csv')
    data = data[~data.DGsolv.isna()]
    data = data[['can_smiles_solute', 'can_smiles_solvent', 'DGsolv','index']]
    data['Exp_or_QM'] = 'Exp'
    data = data.sample(frac=args.data_frac, random_state=1)
 
    index_train_valid, index_test, dummy_train_valid, dummy_test = train_test_split(data['index'], data['index'], 
                                                                                    test_size = 0.1, random_state = args.random_seed)
    test_exp = data[data['index'].isin(index_test)]
    train_valid = data[data['index'].isin(index_train_valid)]

    kfold = KFold(n_splits = 10, shuffle = True, random_state = args.random_seed)
    train_valid_split = list(kfold.split(train_valid))[args.fold_number]
    train_index, valid_index = train_valid_split
    train_exp = train_valid.iloc[train_index]
    valid_exp = train_valid.iloc[valid_index]

    train = train_exp
    valid = valid_exp
    test = test_exp

    student_DBs = ['First_student_cutoff_02.csv', 'Second_student_cutoff_02.csv', 
                   'Third_student_cutoff_02.csv', 'Fourth_student_cutoff_02.csv',
                   'Fifth_student_cutoff_02.csv', 'Sixth_student_cutoff_02.csv',
                   'Seventh_student_cutoff_02.csv', 'Eighth_student_cutoff_02.csv',
                   'Ninth_student_cutoff_02.csv', 'Tenth_student_cutoff_02.csv',
                   'Eleventh_student_cutoff_02.csv', 'Twelfth_student_cutoff_02.csv',
                   '13th_student_cutoff_02.csv', '14th_student_cutoff_02.csv', '15th_student_cutoff_02.csv']
    for csv in student_DBs:
        new_train, new_valid, new_test = add_student_DBs(csv)

        train = pd.concat([train, new_train], ignore_index = True)
        valid = pd.concat([valid, new_valid], ignore_index = True)
        test =  pd.concat([test,  new_test], ignore_index = True)

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(len(train_exp), len(train) - len(train_exp), len(train))
    print(len(valid_exp), len(valid) - len(valid_exp), len(valid))
    print(len(test_exp), len(test) - len(test_exp), len(test))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    train['Train/Valid/Test'] = 'Train'
    valid['Train/Valid/Test'] = 'Valid'
    test['Train/Valid/Test'] = 'Test'

    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)

    ####

    # 220711 - always read preprocessor from the QM-DB-trained model!
    # Since it contains the broadest atom/bond classes

    print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
    print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")
    
    #preprocessor.from_json('noleakyrelu_1M_data_seed2_fold0/preprocessor.json')  
    preprocessor.from_json('json_with_atom_formal_charges/preprocessor.json')   #220829
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))
    
    print(f'Atom classes after: {preprocessor.atom_classes}')
    print(f'Bond classes after: {preprocessor.bond_classes}')


    '''
    if args.tfl:
        print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
        print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")
        preprocessor.from_json('noleakyrelu_1M_data_seed2_fold0/preprocessor.json')  
        output_signature = (preprocessor.output_signature,
                            tf.TensorSpec(shape=(), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.float32))
        print(f'Atom classes after: {preprocessor.atom_classes}')
        print(f'Bond classes after: {preprocessor.bond_classes}')

    else:
        # 220829: just to get all atom and bond types
        output_signature = (preprocessor.output_signature,
                            tf.TensorSpec(shape=(), dtype=tf.float32),
                            tf.TensorSpec(shape=(), dtype=tf.float32))
        print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
        print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")

        print("NOT FOR TRAINING, JUST TO GET PREPROCESSOR JSON - DON'T USE THIS PART FOR TRAINING")
        train_all_smiles = list( set(list(data['can_smiles_solvent']) + list(data['can_smiles_solute']) ) )

        for smiles in train_all_smiles:
            preprocessor.construct_feature_matrices(smiles, train=True)

        print(f'Atom classes after: {preprocessor.atom_classes}')
        print(f'Bond classes after: {preprocessor.bond_classes}')
        
        preprocessor.to_json("model_files/"+ args.modelname  +"/preprocessor.json")
        print("json saved")
    ####
    '''

    train_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, True), output_signature=output_signature)\
        .cache().shuffle(buffer_size=1000)\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    valid_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(valid, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(test, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    ##################
    features_dim = args.num_hidden
    num_messages = args.layers

    #solute
    atom_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='atom_solute')
    bond_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='bond_solute')
    connectivity_Input_solute = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solute')
    global_Input_solute = layers.Input(shape=[4], dtype=tf.float32, name='mol_features_solute')

    #solvent
    atom_Input_solv = layers.Input(shape=[None], dtype=tf.int32, name='atom_solv')
    bond_Input_solv = layers.Input(shape=[None], dtype=tf.int32, name='bond_solv')
    connectivity_Input_solv = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solv')
    global_Input_solv = layers.Input(shape=[4], dtype=tf.float32, name='mol_features_solv')
    ######

    #solute
    atom_state_solute = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solute', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solute)
    bond_state_solute = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solute', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solute)
    global_state_solute = layers.Dense(features_dim, activation='relu')(global_Input_solute)

    #solvent
    atom_state_solv = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solv', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solv)
    bond_state_solv = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solv', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solv)
    global_state_solv = layers.Dense(features_dim, activation='relu')(global_Input_solv)
   
    for i in range(num_messages):
        if args.surv_prob != 1.0:
            surv_prob_i = 1.0 - (((1.0 - args.surv_prob) / (num_messages - 1)) * i)
        else:
            surv_prob_i = 1.0

        atom_state_solute, bond_state_solute, global_state_solute = message_block(atom_state_solute, 
                                                                                   bond_state_solute, 
                                                                                   global_state_solute, 
                                                                                   connectivity_Input_solute, 
                                                                                   features_dim, i, 1.0e-10, surv_prob_i) #dropout at readout only?
                                                                                   #features_dim, i, args.dropout, surv_prob_i)
        atom_state_solv,   bond_state_solv,   global_state_solv   = message_block(atom_state_solv, 
                                                                                bond_state_solv, 
                                                                                global_state_solv, 
                                                                                connectivity_Input_solv, 
                                                                                features_dim, i, 1.0e-10, surv_prob_i) #dropout at readout only?
                                                                                #features_dim, i, args.dropout, surv_prob_i)

    readout_vector = tf.concat([global_state_solute, global_state_solv], -1)

    readout_vector = layers.Dense(features_dim*2, activation='relu')(readout_vector)
    readout_vector = layers.Dropout(args.dropout)(readout_vector)
    '''
    if args.dropout > 0:
        readout_vector = layers.Dropout(args.dropout)(readout_vector)
    '''

    readout_vector = layers.Dense(features_dim*2, activation='relu')(readout_vector)
    readout_vector = layers.Dropout(args.dropout)(readout_vector)
    '''
    if args.dropout > 0:
        readout_vector = layers.Dropout(args.dropout)(readout_vector)
    ''' 
    
    prediction = layers.Dense(1)(readout_vector)

    input_tensors = [atom_Input_solute, bond_Input_solute, connectivity_Input_solute, 
                     atom_Input_solv, bond_Input_solv, connectivity_Input_solv,
                     global_Input_solute, global_Input_solv]

    model = tf.keras.Model(input_tensors, [prediction])

    #model.load_weights('model_files/teacher_220829_w_formal_charges/best_model.h5') 
    model.load_weights('model_files/student15_220926_2/best_model.h5') 
    
    if args.tfl:
        model.load_weights('noleakyrelu_1M_data_seed2_fold0/best_model.h5')

        for layer in model.layers:
            if layer.name not in ['dense_22','dense_23','dense_24']:
                layer.trainable = False
                #print(layer.name, [w.shape for w in layer.get_weights()])

    model.summary()
    ###############################

    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(args.lr))
    model_path = "model_files/"+args.modelname+"/best_model.h5"

    checkpoint = ModelCheckpoint(model_path, monitor="val_loss",\
                                 verbose=2, save_best_only = True, mode='auto', period=1 )

    hist = model.fit(train_data,
                     validation_data=valid_data,
                     epochs=args.epoch,
                     verbose=2, callbacks = [checkpoint])
                     #use_multiprocessing = True, workers = 24

    model.load_weights(model_path)

    train_data_final = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    train_results = model.predict(train_data_final).squeeze()
    valid_results = model.predict(valid_data).squeeze()
    test_results = model.predict(test_data).squeeze()

    mae_train = np.abs(train_results - train['DGsolv']).mean()
    mae_valid = np.abs(valid_results - valid['DGsolv']).mean()
    mae_test = np.abs(test_results - test['DGsolv']).mean()

    train['predicted'] = train_results
    valid['predicted'] = valid_results
    test['predicted'] =  test_results

    print("Fold number", args.fold_number)
    print(len(train),len(valid),len(test))
    print(mae_train,mae_valid,mae_test)

    pd.concat([train, valid, test], ignore_index=True).to_csv('model_files/' + args.modelname +'/kfold_'+str(args.fold_number)+'.csv',index=False)
    preprocessor.to_json("model_files/"+ args.modelname  +"/preprocessor.json")

def cnpred(smi):
    # best model name: 2_sw6
    model = tf.keras.models.load_model('model_files/2_sw6/best_model.h5', custom_objects = nfp.custom_objects)

    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
    preprocessor.from_json('model_files/2_sw6/preprocessor.json')

    inputs = preprocessor.construct_feature_matrices(smi)
    inputs = {key: np.expand_dims(inputs[key], axis=0) for key in ['atom','bond','connectivity','mol_features']} 
    predicted_CN = model.predict( inputs   ).squeeze().squeeze()
    predicted_CN = np.round(predicted_CN,2)

    extractor = tf.keras.Model(model.inputs, [model.layers[-1].input])
    features = np.array(extractor.predict(inputs).squeeze())

    ### Tanimoto similarity analysis ###
    df = pd.read_csv('data/CNdb_pred_results.csv')

    Similarities = []
    for _, row in df.iterrows():
        glob_vector_in_db = np.array([ float(x) for x in row['glob_vector'].split() ])

        C = np.sum( glob_vector_in_db * features  )
        A = np.sum( glob_vector_in_db * glob_vector_in_db   )
        B = np.sum( features * features   )

        S = C / ( A + B - C )

        Similarities.append(S)

    df['similarity'] = Similarities
    df = df.sort_values(by=['similarity'], ascending=False)

    if np.isclose(df.iloc[0].similarity, 1.0):
        df = df.iloc[1:]

    return predicted_CN, df.head(n=10)

def predict_df(df,args):
    model = tf.keras.models.load_model('model_files/'+ args.modelname +'/best_model.h5', custom_objects = nfp.custom_objects)
    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
    preprocessor.from_json('model_files/' + args.modelname +'/preprocessor.json')
    
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))

    df_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(df, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=len(df))\
        .prefetch(tf.data.experimental.AUTOTUNE)

    pred_results = model.predict(df_data).squeeze()
    df['predicted'] = pred_results


    import pickle

    model.summary()
    layer_name_list = [layer.name for layer in model.layers]
     

    layer_indices_for_glob_feat = [  layer_name_list.index(x) for x in ['stochastic_depth_6', 'stochastic_depth_7']   ]
    layer_indices_for_atom_feat = [  layer_name_list.index(x) for x in ['global_average_pooling1d_8', 'global_average_pooling1d_9']   ]

    layers_indices_for_vec_before_concat = [ layer_name_list.index(x) for x in ['stochastic_depth_8', 'stochastic_depth_9']   ]

    layers_indices_dense = [layer_name_list.index(x) for x in ['dense_18', 'dense_19', 'dense_20', 'dense_21','dense_22', 'dense_23', 'dense_24']]

    glob_feat_extractor = tf.keras.Model(model.inputs, [model.layers[x].output for x in layer_indices_for_glob_feat])
    atom_feat_extractor = tf.keras.Model(model.inputs, [model.layers[x].input for x in layer_indices_for_atom_feat])

    glob_feat, atom_feat = glob_feat_extractor.predict(df_data), atom_feat_extractor.predict(df_data)

    concat_vec_extractor = tf.keras.Model(model.inputs, [model.layers[x].output for x in layers_indices_for_vec_before_concat])
    concat_vec_each_side = concat_vec_extractor.predict(df_data)


    last_256_vector_extractor = tf.keras.Model(model.inputs, [model.layers[-1].input])
    last_256_vector = last_256_vector_extractor.predict(df_data)

    with open('last_256_vector.pkl','wb') as f:
        pickle.dump(last_256_vector, f)

    #print(glob_feat[0].shape, glob_feat[1].shape)
    #print(atom_feat[0].shape, atom_feat[1].shape)

    with open('feat_vectors_for_shap.pkl','wb') as f:
        pickle.dump([glob_feat, atom_feat, concat_vec_each_side], f)

    weights_dense_layers = [  model.layers[x].weights for x in layers_indices_dense  ]

    '''
    for x in weights_dense_layers:
        for y in x:
            print(y.shape)
    '''

    with open('weights_last_layer.pkl','wb') as f:
        pickle.dump(weights_dense_layers, f)


    #extractor = tf.keras.Model(model.inputs, [model.layers[-1].input]) #global feature
    #for i, layer in enumerate(model.layers):
    #    print(i, layer.name)

     
    #### This part is for extracting feature (state) vectors (related to Fig. 5) ####
    #extractor = tf.keras.Model(model.inputs, [model.layers[-1].input]) #global feature
    #extractor = tf.keras.Model(model.inputs, [model.layers[-6].output]) #atom features
    #features = extractor.predict(df_data)

    return df

    ##################

if __name__ == '__main__':
    with tf.device(device):
        parser = ArgumentParser()
        parser.add_argument('-lr', type=float, default=1.0e-4, help='Learning rate (default=1.0e-4)')
        parser.add_argument('-batchsize', type=int, default=1024, help='batch_size (default=1024)')
        parser.add_argument('-epoch', type=int, default=1000, help='epoch (default=1000)')
        parser.add_argument('-layers', type=int, default=5, help='number of gnn layers (default=5)')
        parser.add_argument('-num_hidden', type=int, default=128, help='number of nodes in hidden layers (default=128)')

        parser.add_argument('-random_seed', type=int, default=1, help='random seed number used when splitting the dataset (default=1)')
        parser.add_argument('-split_option', type=int, default=2, help='8:1:1 split options - 0: just a random 8:1:1 split,\
                                                                                              1: Training set: Tier1,2,3, validation/test set: Tier 1 only,\
                                                                                              2: split from Tier 1 + split from Tier 2,3  (default=2)')

        parser.add_argument('-sample_weight', type=float, default=1.0, help='whether to use sample weights (default=0.6) If 1.0 -> no sample weights, if < 1.0 -> sample weights to Tier 2,3 methods')
        parser.add_argument('-fold_number', type=int, default=0, help='fold number for Kfold')
        parser.add_argument('-modelname', type=str, default='test_model', help='model name (default=test_model)')
        parser.add_argument('-data_frac', type=float, default=1.0, help='default=1.0')
        parser.add_argument('-tfl', action="store_true", default=False, help='If specified, transfer learning is carried out (default=False)')

        ########
        parser.add_argument('-predict', action="store_true", default=False, help='If specified, prediction is carried out (default=False)')
        parser.add_argument('-predict_df', action="store_true", default=False, help='If specified, prediction is carried out for molecules_to_predict.csv (default=False)')
        
        parser.add_argument('-chunk_number', type=int, default=0, help='')
        parser.add_argument('-smi', type=str, default='', help='SMILES for prediction')
        parser.add_argument('-dropout', type=float, default=0.0, help='default=0.0')
        parser.add_argument('-surv_prob', type=float, default=1.0, help='default=1.0')
        args = parser.parse_args()

    if args.predict:
        predicted_CN, similarity_df = cnpred(args.smi)
        print(predicted_CN)
    elif args.predict_df:
        #df = pd.read_csv('noleakyrelu_1M_data_seed2_fold0/temp_for_evaluate_tl.csv')
        #df = df.sample(frac=0.1, random_state=1)

        #df = pd.read_csv('220607_exp_solu_DB_v2.csv')
        #df = df[~df.DGsolv.isna()]
        print(args.chunk_number)

        #df = pd.read_csv('prediction_for_appl_examples_221001.csv')
        #df = pd.read_csv('rxns_chosen_examples.csv')
        #df = pd.read_csv('input_for_solvent_tsne.csv')
        df = pd.read_csv('molecules_to_predict_logP_examples.csv')
        #df = pd.read_csv('prediction_for_appl_example_HMF_221012.csv')
        #df = pd.read_csv('expdb_group_by_m062x_cosmo_accur.csv')
        #df = df[df.Atom_bond_type_overlap_with_exp == 'Y']
        df = df.iloc[10000 * args.chunk_number : 10000 * (args.chunk_number + 1)]

        df2 = predict_df(df,args) 
        df2.to_csv('prediction_results'+ str(args.chunk_number) +'.csv', index=False)
    else:
        import datetime
        start = datetime.datetime.now()
        main(args)
        end = datetime.datetime.now()
        print(end-start)
