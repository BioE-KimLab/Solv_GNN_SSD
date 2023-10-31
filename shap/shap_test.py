
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from rdkit import Chem

import shap
shap.explainers._deep.deep_tf.op_handlers['AddV2'] = shap.explainers._deep.deep_tf.passthrough

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

device = "/cpu:0"

with tf.device(device):
    df = pd.read_csv('prediction_results0.csv')
    #df = pd.read_csv('prediction_results_hmf_230608.csv')
    df['total_atoms_solute'] = [ Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in df.can_smiles_solute]
    df['total_atoms_solvent'] = [ Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in df.can_smiles_solvent]

    #with open('last_256_vector.pkl','rb') as f:

    with open('feat_vectors_for_shap.pkl','rb') as f:
       glob_feat, atom_feat, concat_vec_each_side = pickle.load(f)
       #X = pickle.load(f)
       #print(X)

    with open('weights_last_layer.pkl', 'rb') as f:
        w1,w2, w3,w4, w5,w6,w7 = pickle.load(f)

        #w1, w2 - solute side
        #w3, w4 - solvent side
        #w5, w6, w7 - dense layers after concat


    ### prepare input
    gf_solute, gf_solvent = glob_feat
    af_solute, af_solvent = atom_feat
    concat_vec_solute, concat_vec_solvent = concat_vec_each_side

    af_solute_avg = []
    af_solvent_avg = []

    for i, af_solute_per_molecule in enumerate(af_solute):
        af_solute_dummy_rmvd = af_solute_per_molecule[0:df.iloc[i]['total_atoms_solute']]
        af_avg = np.mean(af_solute_dummy_rmvd, axis = 0)
        af_solute_avg.append(af_avg)

    for i, af_solvent_per_molecule in enumerate(af_solvent):
        af_solvent_dummy_rmvd = af_solvent_per_molecule[0:df.iloc[i]['total_atoms_solvent']]
        af_avg = np.mean(af_solvent_dummy_rmvd, axis = 0)
        af_solvent_avg.append(af_avg)

    af_solute_avg = np.array(af_solute_avg)
    af_solvent_avg = np.array(af_solvent_avg)

    ### build model
    af_solute_input = layers.Input(shape=[128], dtype=tf.float32, name='af_solute')
    gf_solute_input = layers.Input(shape=[128], dtype=tf.float32, name='gf_solute')

    af_solute_hidden = layers.Dense(128, activation='relu', name = 'denserelu_solute')(af_solute_input)
    af_solute_hidden = layers.Dense(128, name = 'dense_solute')(af_solute_hidden)
    concat_solute = layers.Add()([gf_solute_input, af_solute_hidden]) ## solute_latent_vector



    af_solvent_input = layers.Input(shape=[128], dtype=tf.float32, name='af_solvent')
    gf_solvent_input = layers.Input(shape=[128], dtype=tf.float32, name='gf_solvent')

    af_solvent_hidden = layers.Dense(128, activation='relu', name = 'denserelu_solvent')(af_solvent_input)
    af_solvent_hidden = layers.Dense(128, name = 'dense_solvent')(af_solvent_hidden)
    concat_solvent = layers.Add()([gf_solvent_input, af_solvent_hidden]) ## solvent_latent_vector

    #prediction = [concat_solute, concat_solvent]

    readout_vector = tf.concat([concat_solute, concat_solvent], -1)
    readout_vector = layers.Dense(256, activation = 'relu', name = 'readout1')(readout_vector)
    readout_vector = layers.Dense(256, activation = 'relu', name = 'readout2')(readout_vector)
    prediction = layers.Dense(1, name = 'readout3')(readout_vector)


    ####
    input_tensors = [af_solute_input, gf_solute_input, af_solvent_input, gf_solvent_input]
    model = tf.keras.Model(input_tensors, [prediction])

    ### read trained weights

    #print([layer.name for layer in model.layers])

    model.layers[2].set_weights( [w1[0].numpy() , w1[1].numpy()] )
    model.layers[5].set_weights( [w2[0].numpy() , w2[1].numpy()]  )

    model.layers[3].set_weights( [w3[0].numpy() , w3[1].numpy()] )
    model.layers[7].set_weights( [w4[0].numpy() , w4[1].numpy()]  )

    model.layers[-3].set_weights( [w5[0].numpy() , w5[1].numpy()]  )
    model.layers[-2].set_weights( [w6[0].numpy() , w6[1].numpy()]  )
    model.layers[-1].set_weights( [w7[0].numpy() , w7[1].numpy()]  )

    #pred_solute, pred_solvent = pred[0]
    #print(np.all(np.isclose(pred_solute, concat_vec_solute, rtol=0, atol = 1e-05))) #True
    #print(np.all(np.isclose(pred_solvent, concat_vec_solvent, rtol=0, atol = 1e-05))) #True

    pred = model.predict([af_solute_avg, gf_solute, af_solvent_avg, gf_solvent]).squeeze()
    #print(np.all(np.isclose(pred, np.array(df['predicted']), rtol=0, atol = 1e-05))) # True
    #print(np.abs(pred -  df.predicted).mean(), np.abs(pred -  df.predicted).max())

    #### SHAP part ####
    e = shap.DeepExplainer(model, [af_solute_avg, gf_solute, af_solvent_avg, gf_solvent])
    shap_values = e.shap_values([af_solute_avg, gf_solute, af_solvent_avg, gf_solvent])

    af_solute_shap, gf_solute_shap, af_solvent_shap, gf_solvent_shap = shap_values[0]
    all_shap = af_solute_shap + gf_solute_shap + af_solvent_shap + gf_solvent_shap 


    atomwise_shap = np.zeros((len(df), int(df.total_atoms_solute.max()), 128))
    print(atomwise_shap.shape)


    for i in range(len(df)):
        for j in range(df.iloc[i]['total_atoms_solute']):
            for k in range(128):
                atomwise_shap[i][j][k] = all_shap[i][k] * ( af_solute[i][j][k] / ( df.iloc[i]['total_atoms_solute'] * af_solute_avg[i][k] )  )
                #atomwise_shap[i][j][k] = all_shap[i][k] * np.abs( af_solute[i][j][k] / ( df.iloc[i]['total_atoms_solute'] * af_solute_avg[i][k] )  )

        #if np.sum(atomwise_shap[i]) * np.sum(all_shap[i]) < 0:
        #    print("????????")
        #for k in range(128):

    '''
    for i in range(len(df)):
        for k in range(128):
            af_solute_allatoms = af_solute[i][:df.iloc[i]['total_atoms_solute'],k]

            af_solute_pos_ind = [ ind for ind, x in enumerate(af_solute_allatoms) if x >= 0 ]
            af_solute_neg_ind = [ ind for ind, x in enumerate(af_solute_allatoms) if x < 0 ]

            af_solute_pos_sum = sum( [af_solute_allatoms[ind] for ind in af_solute_pos_ind] )
            af_solute_neg_sum = sum( [af_solute_allatoms[ind] for ind in af_solute_neg_ind] )

            #print(len(af_solute_allatoms))
            #print("*****************")
            #print(df.iloc[i].can_smiles_solute, len( af_solute_pos_ind) + len(af_solute_neg_ind), df.iloc[i]['total_atoms_solute'])
            #print("*****************")
            #print([af_solute_allatoms[ind] for ind in af_solute_pos_ind], af_solute_pos_sum)
            #print([af_solute_allatoms[ind] for ind in af_solute_neg_ind], af_solute_neg_sum)
            #print('-------------')

            correction_factor = ( all_shap[i][k] - af_solute_pos_sum - af_solute_neg_sum ) / (af_solute_pos_sum + af_solute_neg_sum)
            for j in range(df.iloc[i]['total_atoms_solute']):

                pos_sum_corrected = af_solute_pos_sum - af_solute_neg_sum * correction_factor
                neg_sum_corrected = af_solute_neg_sum - af_solute_pos_sum * correction_factor

                if pos_sum_corrected < 0 or neg_sum_corrected > 0:
                    #print("is this happening?")
                    pos_sum_corrected = af_solute_pos_sum + af_solute_neg_sum * correction_factor
                    neg_sum_corrected = af_solute_neg_sum + af_solute_pos_sum * correction_factor


                if j in af_solute_pos_ind:
                    atomwise_shap[i][j][k] = pos_sum_corrected * ( af_solute[i][j][k] / af_solute_pos_sum  )
                    if atomwise_shap[i][j][k] < 0:
                        print("?????????????????????")
                else:
                    atomwise_shap[i][j][k] = neg_sum_corrected * ( af_solute[i][j][k] / af_solute_neg_sum  )
                    if atomwise_shap[i][j][k] > 0:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(np.sum(all_shap[i]), np.sum(atomwise_shap[i]))
    '''


    atomwise_shap_for_plot = np.sum(atomwise_shap, axis = 2)
    #print(atomwise_shap_for_plot.shape)
    #print(atomwise_shap_for_plot[0:22])

    print(atomwise_shap_for_plot.min(), atomwise_shap_for_plot.max() )




    df['shap_values'] = [x for x in atomwise_shap_for_plot]


    df.to_csv('shap_results.csv',index=False)

    atomwise_sum = np.sum(atomwise_shap_for_plot, axis = -1)

    reg = LinearRegression().fit(atomwise_sum.reshape(-1,1), df.predicted)
    print(reg.coef_, reg.intercept_, reg.score(atomwise_sum.reshape(-1,1), df.predicted))

