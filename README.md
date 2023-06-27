 <p align="center">
 <img src="images/Screenshot from 2023-06-26 15-47-02.png" width="250" height="80"/> 
 </p>

# <p align="center">
An ensemble-based approach for predicting protein S-nitrosylation sites by integrating supervised word embedding and embedding from pre-trained protein language model
</p>

<p align="center">
<img src="images/Screenshot from 2023-06-22 15-40-37.png"/> 
</p>

<p align="center">
<a href="https://www.python.org/"><img alt="python" src="https://img.shields.io/badge/Python-3.9.7-blue.svg"/></a>
<a href="https://www.tensorflow.org/"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-2.9.1-orange.svg"/></a>
<a href="https://www.tensorflow.org/"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-2.9.1-orange.svg"/></a>
<a href="https://keras.io/"><img alt="Keras" src="https://img.shields.io/badge/Keras-2.9.0-red.svg"/></a>
<a href="https://huggingface.co/transformers/"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.18.0-yellow.svg"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.11.0-orange.svg"/></a>
<a href="https://biopython.org/"><img alt="Bio" src="https://img.shields.io/badge/Bio-1.5.2-brightgreen.svg"/></a>
<a href="https://scikit-learn.org/"><img alt="scikit_learn" src="https://img.shields.io/badge/scikit_learn-1.2.0-blue.svg"/></a>
<a href="https://matplotlib.org/"><img alt="matplotlib" src="https://img.shields.io/badge/matplotlib-3.5.1-blueviolet.svg"/></a>
<a href="https://numpy.org/"><img alt="numpy" src="https://img.shields.io/badge/numpy-1.23.5-red.svg"/></a>
<a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-1.5.0-yellow.svg"/></a>
<a href="https://docs.python-requests.org/en/latest/"><img alt="requests" src="https://img.shields.io/badge/requests-2.27.1-green.svg"/></a>
<a href="https://seaborn.pydata.org/"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-0.11.2-lightgrey.svg"/></a>
<a href="https://tqdm.github.io/"><img alt="tqdm" src="https://img.shields.io/badge/tqdm-4.63.0-blue.svg"/></a>
<a href="https://xgboost.readthedocs.io/en/latest/"><img alt="xgboost" src="https://img.shields.io/badge/xgboost-1.5.0-purple.svg"/></a>

 
</p>

## About
pLMSNOSite is a robust predictor of S-nitrosylation modification sites in protein sequences. It employs an intermediate-fusion based stacked generalization approach to harness the representational power of embeddings obtained from transformer protein language model trained on full sequence combined with supervised word embedding layer trained on window sequence.

## Cite this article
Pratyush, P., Pokharel, S., Saigo, H. et al. pLMSNOSite: an ensemble-based approach for predicting protein S-nitrosylation sites by integrating supervised word embedding and embedding from pre-trained protein language model. BMC Bioinformatics 24, 41 (2023). https://doi.org/10.1186/s12859-023-05164-9

## Authors
Pawel Pratyush<sup>1</sup>, Suresh Pokharel<sup>1</sup>, Hiroto Saigo<sup>2</sup>, Dukka B KC<sup>1*</sup>
<br>
<sup>1</sup>Department of Computer Science, Michigan Technological University, Houghton, MI, USA.
<br>
<sup>2</sup>Department of Electrical Engineering and Computer Science, Kyushu University, 744, Motooka, Nishi-ku, 819-0395, Japan

<sup>*</sup> Corresponding Author: dbkc@mtu.edu

## Webserver

You can access our webserver at [kcdukkalab.org/pLMSNOSite/](http://kcdukkalab.org/pLMSNOSite/).


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
