<div align="center">

# <span style="color:blue;">pLMSNOSite</span> 
</div>


 <p align="center">
 Use Transformer-based Protein Language Model (pLM) for prediction of S-nitrosylation(SNO) modification sites in protein sequences
 </p>
 

 
</p>


## Webserver 

You can access the webserver of pLMSNOSite at [kcdukkalab.org/pLMSNOSite/](http://kcdukkalab.org/pLMSNOSite/).

## Cite this article
Pratyush, P., Pokharel, S., Saigo, H. et al. pLMSNOSite: an ensemble-based approach for predicting protein S-nitrosylation sites by integrating supervised word embedding and embedding from pre-trained protein language model. BMC Bioinformatics 24, 41 (2023). https://doi.org/10.1186/s12859-023-05164-9

The corresponding BibTeX:
```
@article{ WOS:000934967300003,
Author = {Pratyush, Pawel and Pokharel, Suresh and Saigo, Hiroto and Kc, Dukka B.},
Title = {pLMSNOSite: an ensemble-based approach for predicting protein
   S-nitrosylation sites by integrating supervised word embedding and
   embedding from pre-trained protein language model},
Journal = {BMC BIOINFORMATICS},
Year = {2023},
Volume = {24},
Number = {1},
Month = {FEB 8},
DOI = {10.1186/s12859-023-05164-9},
Article-Number = {41},
ISSN = {1471-2105},
ORCID-Numbers = {Pratyush, Pawel/0000-0002-4210-1200},
Unique-ID = {WOS:000934967300003},
}
```

## Authors
Pawel Pratyush<sup>1</sup>, Suresh Pokharel<sup>1</sup>, Hiroto Saigo<sup>2</sup>, Dukka B KC<sup>1*</sup>
<br>
<sup>1</sup>Department of Computer Science, Michigan Technological University, Houghton, MI, USA.
<br>
<sup>2</sup>Department of Electrical Engineering and Computer Science, Kyushu University, 744, Motooka, Nishi-ku, 819-0395, Japan

<sup>*</sup> Corresponding Author: dbkc@mtu.edu

## Getting Started  :rocket: 

To get a local copy of the repository, you can either clone it or download it directly from GitHub.

### Clone the Repository

If you have Git installed on your system, you can clone the repository by running the following command in your terminal:

```shell
git clone git@github.com:KCLabMTU/pLMSNOSite.git
```
### Download the Repository
Alternatively, if you don't have Git or prefer not to use it, you can download the repository directly from GitHub. Click [here](https://github.com/KCLabMTU/pLMSNOSite/archive/refs/heads/main.zip) to download the repository as a zip file.

Note: In the 'Download the Repository' section, the link provided is a direct download link to the repository's `main` branch as a zip file. This may differ if your repository's default branch is named differently.

## Install Libraries

Python version: `3.9.7`

To install the required libraries, run the following command:

```shell
pip install -r requirements.txt
```

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
```shell
pip install -q SentencePiece transformers
```
## Evaluate pLMSNOSite on Independent Test Set
To evaluate our model on the independent test set, we have already placed the test sequences and corresponding ProtT5 features in `data/test/` folder. After installing all the requirements, run the following command:
<br>
```shell
 python evaluate_model.py
```

## Predict S-Nitrosylation modification in your own sequence
1. Place your FASTA file in the `input/sequence.fasta` directory.
2. Run the following command:
   ```shell
   python predict.py
   ```
3. Find the results at `output/` folder.

## Training and other experiments
1. Find training data at `data/train/` folder
2. Find all the codes and models related to training at `training_experiments` folder (To be updated).



## Collaboration  :handshake: 
<p>
  <a href="https://www.mtu.edu/">
    <img src="images/mtu_logo.png" alt="MTU Logo" width="130" height="100">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.kyushu-u.ac.jp/en/">
    <img src="images/Kyushu_University_Logo-586x700.png" alt="Kyushu University Logo" width="100" height="110">
  </a>
</p>



## Funding 
<p>
  <a href="https://www.nsf.gov/">
    <img src="images/NSF_Official_logo.svg" alt="NSF Logo" width="110" height="110" style="margin-right: 20px;">
  </a>
  <a href="https://www.jsps.go.jp/english/">
    <img src="images/JSPS-logo.jpg" alt="JSPS Logo" width="180" height="70" style="margin-left: 20px;">
  </a>
</p>




## Contact  :mailbox: 
Should you have any inquiries related to this project, please feel free to reach out via email. Kindly CC all of the following recipients in your communication for a swift response:

- Main Contact: [dbkc@mtu.edu](mailto:dbkc@mtu.edu)
- CC: [ppratyush@mtu.edu](mailto:ppratyush@mtu.edu)
- CC: [sureshp@mtu.edu](mailto:sureshp@mtu.edu)

We look forward to addressing your queries and concerns.

