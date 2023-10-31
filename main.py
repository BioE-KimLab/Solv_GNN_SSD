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
    #####################################
    data = pd.read_csv('data/Exp-DB.csv')
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

    num_Aug_DBs = 35
    student_DBs = ['Aug-DB'+str(x)+'.csv' for x in range(1,1+num_Aug_DBs)]
    student_DBs = ['data/Aug-DBs/' + x for x in student_DBs]
    #student_DBs = []
                   
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

    print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
    print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")
    
    preprocessor.from_json('json_with_atom_formal_charges/preprocessor.json')
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))

    #train_all_smiles = list( set(list(data['can_smiles_solvent']) + list(data['can_smiles_solute']) ) )
    #for smiles in train_all_smiles:
    #    preprocessor.construct_feature_matrices(smiles, train=True)
    
    print(f'Atom classes after: {preprocessor.atom_classes}')
    print(f'Bond classes after: {preprocessor.bond_classes}')

    #preprocessor.to_json("model_files/"+ args.modelname  +"/preprocessor.json")

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

    readout_vector = layers.Dense(features_dim*2, activation='relu')(readout_vector)
    readout_vector = layers.Dropout(args.dropout)(readout_vector)
    
    prediction = layers.Dense(1)(readout_vector)

    input_tensors = [atom_Input_solute, bond_Input_solute, connectivity_Input_solute, 
                     atom_Input_solv, bond_Input_solv, connectivity_Input_solv,
                     global_Input_solute, global_Input_solv]

    model = tf.keras.Model(input_tensors, [prediction])

    #model.load_weights('model_files/student34_230110_1/best_model.h5')
    
    if args.tfl:
        model.load_weights('noleakyrelu_1M_data_seed2_fold0/best_model.h5')

        for layer in model.layers:
            if layer.name not in ['dense_22','dense_23','dense_24']:
                layer.trainable = False
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

    #### This part is for extracting feature (state) vectors (related to shap) ####
    #extractor = tf.keras.Model(model.inputs, [model.layers[-1].input]) #global feature
    #for i, layer in enumerate(model.layers):
    #    print(i, layer.name)

    '''
    extractor_solvent_af = tf.keras.Model(model.inputs, [model.layers[1].output, 
                                                         model.layers[0].output,
                                                         model.layers[15].output,
                                                         model.layers[12].output])
    features = extractor_solvent_af.predict(df_data)
    af = sorted(list(set(np.concatenate((features[0].flatten(), features[1].flatten()), axis=None))))
    bf = sorted(list(set(np.concatenate((features[2].flatten(), features[3].flatten()), axis=None))))

    print(af)
    print('------------')
    print(bf)
    '''
     
    #extractor = tf.keras.Model(model.inputs, [model.layers[-1].input]) #global feature
    #extractor = tf.keras.Model(model.inputs, [model.layers[-6].output]) #atom features
    #features = extractor.predict(df_data)

    df['predicted'] = pred_results
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
        parser.add_argument('-sample_weight', type=float, default=1.0, help='whether to use sample weights (default=0.6) If 1.0 -> no sample weights, if < 1.0 -> sample weights to Tier 2,3 methods')
        parser.add_argument('-fold_number', type=int, default=0, help='fold number for Kfold')
        parser.add_argument('-modelname', type=str, default='test_model', help='model name (default=test_model)')
        parser.add_argument('-data_frac', type=float, default=1.0, help='default=1.0')
        parser.add_argument('-tfl', action="store_true", default=False, help='If specified, transfer learning is carried out (default=False)')

        ########
        parser.add_argument('-predict_df', action="store_true", default=False, help='If specified, prediction is carried out for molecules_to_predict.csv (default=False)')
        
        parser.add_argument('-chunk_number', type=int, default=0, help='in case one wants to predict a chunk of 10,000 data points among a huge number of data points (>10^4)')
        parser.add_argument('-dropout', type=float, default=0.0, help='default=0.0')
        parser.add_argument('-surv_prob', type=float, default=1.0, help='default=1.0')
        args = parser.parse_args()

    if args.predict_df:
        #df = pd.read_csv('noleakyrelu_1M_data_seed2_fold0/temp_for_evaluate_tl.csv')
        #df = df.sample(frac=0.1, random_state=1)

        #df = pd.read_csv('220607_exp_solu_DB_v2.csv')
        #df = df[~df.DGsolv.isna()]
        #print(args.chunk_number)

        #df = pd.read_csv('remaining_after_first_student.csv')
        #df = pd.read_csv('remaining_after_second_student.csv')
        #df = pd.read_csv('remaining_after_third_student.csv')
        #df = pd.read_csv('remaining_after_fourth_student.csv')
        #df = pd.read_csv('remaining_after_fifth_student.csv')
        #df = pd.read_csv('remaining_after_sixth_student.csv')
        #df = pd.read_csv('remaining_after_seventh_student.csv')
        #df = pd.read_csv('remaining_after_eighth_student.csv')
        #df = pd.read_csv('remaining_after_ninth_student.csv')
        #df = pd.read_csv('remaining_after_tenth_student.csv')
        #df = pd.read_csv('remaining_after_eleventh_student.csv')
        #df = pd.read_csv('remaining_after_twelfth_student.csv')
        #df = pd.read_csv('remaining_after_29th_student.csv')
        #df = pd.read_csv('remaining_after_30th_student.csv')
        #df = pd.read_csv('remaining_after_34th_student.csv')
        #df = pd.read_csv('remaining_after_38th_student.csv')
        #df = pd.read_csv('remaining_after_43rd_student.csv')
        df = pd.read_csv('remaining_after_40pth_student.csv')
        #df = pd.read_csv('prediction_for_appl_examples_220927.csv')
        #df = pd.read_csv('solvent-solvent-pred.csv')
        #df = pd.read_csv('prediction_for_appl_examples_221206.csv')
        #df = df[df.Atom_bond_type_overlap_with_exp == 'Y']
        #df = pd.read_csv('remaining_after_30pth_student.csv')
        df = df.iloc[10000 * args.chunk_number : 10000 * (args.chunk_number + 1)]

        df2 = predict_df(df,args) 
        #df2.to_csv('prediction_results_control_'+ args.modelname.split('_')[0] +'.csv', index=False)
        df2.to_csv('prediction_results'+ str(args.chunk_number) +'.csv', index=False)
    else:
        import datetime
        start = datetime.datetime.now()
        main(args)
        end = datetime.datetime.now()
        print(end-start)
