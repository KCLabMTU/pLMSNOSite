import pandas as pd
import tqdm
from Bio import SeqIO
import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,StratifiedKFold
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv2D,Embedding, MaxPooling2D, Conv1D, Dense, MaxPooling1D, Input, Flatten, LSTM, Dropout, Bidirectional,Normalization, Flatten, Reshape, Lambda, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import auc,confusion_matrix, matthews_corrcoef,accuracy_score,classification_report,f1_score,roc_curve,accuracy_score,roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from numpy import array,argmax
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.backend import expand_dims
from keras import backend as K 
from sklearn.manifold import TSNE
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

import sys
import time
from random import randint

def get_input_for_embedding(fasta_file):
    """
    input: fasta file with fixed window size
    returns: integer encoding for all sequences 
    """
    
    encodings = []
    
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYVUX-'
    
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data = seq_record.seq
        for char in data:
            if char not in alphabet:
                return
        integer_encoded = [char_to_int[char] for char in data]
        encodings.append(integer_encoded)
    encodings = np.array(encodings)
    return encodings

def extract_one_windows_position(protein_id,sequence,site_residue,site,window_size):
    
    '''
    Description: Extract a window from the given string at given position of given size
                (Need to test more conditions, optimizations)
    Parameters:
        protein_id (str): just used for debug purpose
        sequence (str): 
        site_residue (chr):
        window_size(int):
    Returns:
        string: a window/section
    '''
    
    
    if (window_size%2)==0:
        print('Error: Enter odd number window size')
        return 0
    
    half_window = int((window_size-1)//2)
    
    if(sequence is None):
        print('No sequence for [protein_id,site_residue,window_size] ='+str([protein_id, site_residue,window_size]))
        return 0
    else:
        seq_length = len(sequence)
    
    if(sequence[site-1] != site_residue): # check if site_residue at position site is valid
        print('Given site-residue and site does not match [protein_id,site_residue, site] ='+str([protein_id, site_residue,site]))
        return 0
    
    
    # if window is greater than seq length, make the sequence long by introducing virtual amino acids
    # To avoid different conditions for virtual amino acids, add half window everywhere
    sequence = "-" * half_window + sequence + "-" * half_window
    #sequence = sequence[::-1][:half_window][::-1] + sequence + sequence[:half_window]
    site=site+half_window
    
    section = sequence[site - 1-half_window : site + half_window]
    return section

def load_all_models(model_names):
	"""
	description: combines different pretrained models
	"""
	all_models = list()

	for model in model_names:
		filename = 'models/'+ model + '.h5'
		model = load_model(filename, custom_objects={"K": K},compile = False)
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

def get_predictions(model,data):
        print('\n')
        print('Generating features from final hidden layer\n')
        #model.summary()
        layer_name = model.layers[len(model.layers)-2].name #-1 for last layer, -2 for second last and so on"
        print('\nGetting outputs from layer: '+layer_name)
        intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model.predict(data)
        print('\nObtained feature vector shape: '+str(intermediate_output.shape[1]))
        print('\nShape of returned data: '+str(intermediate_output.shape))
        print('-'*100)
        return pd.DataFrame(intermediate_output)
  
     
# size of the fragment
win_size=37

# load pretrained model
filename = 'models/pLMSNOSite.h5'
model_final = load_model(filename, custom_objects={"K": K},compile = False)

# load and prepare both protT5 and Embedding test dataset
X_test=pd.read_csv('data/test/protT5_test.csv')


df2=pd.read_csv('data/test/sequence_test.csv')
test_embedding = get_input_for_embedding('data/test/embedding_test.fasta')
X_test_embedding = test_embedding
y_test = df2.Target

X_test = X_test.iloc[:,3:-1]
X_test_pt5=np.asarray(X_test).astype('float32')


members = load_all_models(['ProtT5','Embedding'])

for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False

prot_pred_test = get_predictions(members[0],X_test_pt5)
emb_pred_test = get_predictions(members[1],X_test_embedding)
X_stacked_test = pd.concat([prot_pred_test,emb_pred_test],axis=1)
y_stacked_test = y_test

# evaluate loaded model on test data
y_pred = model_final.predict(X_stacked_test)
y_pred = (y_pred > 0.50)
y_pred = np.array(y_pred)

mcc=matthews_corrcoef(y_stacked_test, y_pred)
print("Matthews Correlation : ",mcc)

cm=confusion_matrix(y_stacked_test, y_pred)
print("Confusion Matrix : \n",cm)
print("Sensitivity : ",cm[1][1]/(cm[1][1]+cm[1][0]))
print("Specificity: ",cm[0][0]/(cm[0][0]+cm[0][1]))
roc_auc = roc_auc_score(y_stacked_test,y_pred)
print("AUROC : ", roc_auc)
