# <p align="center">pLMSNOSite</p>
<p align="center">
An ensemble-based approach for predicting protein S-nitrosylation sites by integrating supervised word embedding and embedding from pre-trained protein language model
</p>
<p align="center">
<img src="images/Screenshot from 2023-06-22 15-40-37.png"/> 
</p>


## About pLMNOSite
pLMSNOSite is a robust predictor of S-nitrosylation modification sites in protein sequences. It employs an intermediate-fusion based stacked generalization approach to harness the representational power of embeddings obtained from protein language model trained on full sequence combined with supervised word embedding layer trained on window sequence.

## Cite this article
Pratyush, P., Pokharel, S., Saigo, H. et al. pLMSNOSite: an ensemble-based approach for predicting protein S-nitrosylation sites by integrating supervised word embedding and embedding from pre-trained protein language model. BMC Bioinformatics 24, 41 (2023). https://doi.org/10.1186/s12859-023-05164-9

## Authors
Pawel Pratyush<sup>1</sup>, Suresh Pokharel<sup>1</sup>, Hiroto Saigo<sup>2</sup>, Dukka B KC<sup>1*</sup>
<br>
<sup>1</sup>Department of Computer Science, Michigan Technological University, Houghton, MI, USA.
<br>
<sup>2</sup>Department of Electrical Engineering and Computer Science, Kyushu University, 744, Motooka, Nishi-ku, 819-0395, Japan

<sup>*</sup> Corresponding Author: dbkc@mtu.edu

## Install Libraries
Python version: `3.9.7`

Install from requirement.txt: 
<code>
pip install -r requirements.txt
</code>

Required libraries and versions: 
<code>
Bio==1.5.2
keras==2.9.0
matplotlib==3.5.1
numpy==1.23.5
pandas==1.5.0
requests==2.27.1
scikit_learn==1.2.0
seaborn==0.11.2
tensorflow==2.9.1
torch==1.11.0
tqdm==4.63.0
transformers==4.18.0
xgboost==1.5.0
</code>

## Install Transformers
<code>
pip install -q SentencePiece transformers
</code>

## Evaluate pLMSNOSite on Independent Test Set
To evaluate our model on the independent test set, we have already placed the test sequences and corresponding ProtT5 features in `data/test/` folder. After installing all the requirements, run the following command:
<br>
<code>
 python evaluate_model.py
</code>

## Predict Nitrosylation in your own sequence
1. Place the fasta file in `input/sequence.fasta`
2. run `python predict.py`
3. Find the results at `output/` folder.

## Training and other experiments
1. Find training data at `data/train/` folder
2. Find all the codes and models related to training at `training_experiments` folder (To be updated).


## Contact
For any type of inquiry related to this work, please send an email to dbkc@mtu.edu (CC: ppratyush@mtu.edu and sureshp@mtu.edu).
