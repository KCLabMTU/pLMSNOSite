# pLMSNOSite
An ensemble-based approach for predicting protein S-nitrosylation sites by integrating supervised word embedding and embedding from pre-trained protein language model

Pawel Pratyush<sup>1</sup>, Suresh Pokharel1, Hiroto Saigo2, Dukka B KC1*
1.     Department of Computer Science, Michigan Technological University, Houghton, MI, USA.
2.     Department of Electrical Engineering and Computer Science, Kyushu University, 744, Motooka, Nishi-ku, 819-0395, Japan

*  Corresponding Author: dbkc@mtu.edu


## Install Libraries
<code>
pip install -r requirements.txt
</code>

## Install Transformers
<code>
pip install -q SentencePiece transformers
</code>

## Evaluate pLMSNOSite on Independent Test Set
To evaluate our model on the independent test set, we have already placed the test sequences and corresponding ProtT5 features in `data/test/` folder. After installing all the requirements, run the following command:
<code>
 python evaluate_model.py
</code>

## Predict Nitrosylation in your own sequence
1. Place the fasta file in `input/sequence.fasta`
2. run `python predict.py`
3. Find the results at `output/` folder.

## Training and other experiments
1. Find training data at `data/train/` folder
2. Find all the codes and models related to training at `training_experiments` folder.


## Contact
For any type of inquiry related to this work, please send an email to these addresses:
1. dbkc@mtu.edu
2. ppratyush@mtu.edu
3. sureshp@mtu.edu
