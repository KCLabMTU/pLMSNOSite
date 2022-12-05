#!pip install -q SentencePiece transformers
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import tqdm
from Bio import SeqIO
import gc
import pandas as pd
import numpy as np

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
# model = model.half()
gc.collect()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = model.to(device)
model = model.eval()

def find_features_full_seq(sequences_Example): 
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu().numpy()

    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding[0][:seq_len-1]
    return seq_emd
