"""
import required libraries
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from Bio import SeqIO
from keras import backend as K
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

# for ProtT5 model
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc

"""
define file paths and other parameters
"""
input_fasta_file = "input/sequence.fasta" # load test sequence
output_csv_file = "output/results.csv" 
model_path = 'models/pLMSNOSite.h5'
win_size = 37


"""
Load tokenizer and pretrained model ProtT5
"""
# install SentencePiece transformers if not installed already
#!pip install -q SentencePiece transformers


tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
pretrained_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
# pretrained_model = pretrained_model.half()
gc.collect()

# define devices
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
pretrained_model = pretrained_model.to(device)
pretrained_model = pretrained_model.eval()

def get_protT5_features(sequence): 
    """
    Description: Extract a window from the given string at given position of given size
                (Need to test more conditions, optimizations)
    Input:
        sequence (str): str of length l
    Returns:
        tensor: l*1024
    """
    # add space in between amino acids
    sequence = [' '.join(e) for e in sequence]
    
    # replace rare amino acids with X
    sequence = [re.sub(r"[UZOB]", "X", seq) for seq in sequence]
    
    # set configurations and extract features
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens = True, padding = True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
        embedding = pretrained_model(input_ids = input_ids, attention_mask = attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()

    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding[0][: seq_len - 1]
    
    return seq_emd
    

def extract_one_windows_position(sequence,site,window_size=37):
    '''
    Description: Extract a window from the given string at given position of given size
                (Need to test more conditions, optimizations)
    Parameters:
        sequence (str):
        site:
        window_size(int):
    Returns:
        string: a window/section
    '''
    
    half_window = int((window_size-1)/2)
    
    # if window is greater than seq length, make the sequence long by introducing virtual amino acids
    # To avoid different conditions for virtual amino acids, add half window everywhere
    sequence = "X" * half_window + sequence + "X" * half_window
    
    section = sequence[site - 1 : site + 2 * half_window]
    return section

def get_predictions(model, data):

    layer_name = model.layers[len(model.layers)-2].name #-1 for last layer, -2 for second last and so on"
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

    
    intermediate_output = intermediate_layer_model.predict(data, verbose=0)

    return pd.DataFrame(intermediate_output)

def get_input_for_embedding(window):
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    
    for char in window:
        if char not in alphabet:
            return
    integer_encoded = np.array([char_to_int[char] for char in window])
    return integer_encoded
    
    
# initialize empty result dataframe
results_df = pd.DataFrame(columns = ['prot_desc', 'position','site_residue', 'probability', 'prediction'])

# load base models
ProtT5_model = load_model('models/ProtT5.h5', compile = False)
Embedding_model = load_model('models/Embedding.h5', custom_objects={"K": K}, compile = False)


for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = seq_record.seq
    
    positive_predicted = []
    negative_predicted = []
    
    # extract protT5 for full sequence and store in temporary dataframe 
    pt5_all = get_protT5_features([sequence])
    print(pt5_all.shape)
    # generate embedding features and window for each amino acid in sequence
    for index, amino_acid in enumerate(sequence):
        
        # check if AA is 'C' (cysteine)
        if amino_acid in ['C']:
            site = index + 1

            # extract window
            window = extract_one_windows_position(sequence, site)
            
            # extract embedding_encoding
            X_test_embedding = np.reshape(get_input_for_embedding(window), (1, win_size))
            
            # get ProtT5 features extracted above
            X_test_pt5 = np.reshape(pt5_all[index], (1,1024))
            
            prot_pred_test = get_predictions(ProtT5_model, X_test_pt5)
            emb_pred_test = get_predictions(Embedding_model, [X_test_embedding])
            
            X_stacked_test = pd.concat([emb_pred_test, prot_pred_test],axis=1)

            # load combined model
            combined_model = load_model(model_path)
            y_pred = combined_model.predict(X_stacked_test, verbose = 0)[0][0]

            # append results to results_df
            results_df.loc[len(results_df)] = [prot_id, site, amino_acid, y_pred, int(y_pred > 0.5)]

# Export results 
print('Saving results ...')
results_df.to_csv(output_csv_file, index = False)
print('Results saved to ' + output_csv_file)
