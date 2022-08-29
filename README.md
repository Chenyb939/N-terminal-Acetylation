
# User Guide 
- [User Guide](#user-guide)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [Data preparation](#data-preparation)
  - [Training](#training)
  - [Testing](#testing)

# Installation
We do not yet have a mechanism for "installing" MTNA directly from sources like pip or conda for the time being. You can clone the MTNA repository with the following command:

```git clone https://github.com/Chenyb939/N-terminal-Acetylation```

After cloning the repository, the necessary software dependencies (i.e. the environment) to run MTNA can be installed using conda:

```conda env create -f environment.yml```

This will create a new environment named `MTNA`. This environment needs to be activated via `conda activate MTNA` before using any of the code in this repository.

# Quick start

The data schema to run our code is as follows: 
```
[N-TERMINAL-ACETYLATION]/
 |5-fold-data/
 |  |__1.fa fasta file
 |  |__2.fa fasta file
 |  |__3.fa fasta file
 |  |__4.fa fasta file
 |  |__5.fa fasta file
 |
 |test_data/
 |  |__test.fa fasta file
 |
 |model/
 |  |__checkpoint.pt pt file
```

We provide a human pre-trained model at the fold `model`. This provided model is trained on the dataset we discuss in the manuscript. Running `test.py`(see below) will use pre-trained model. 
```
python test.py
```

# Usage
## Data preparation
Before training the data shuould be pleased in the corresponding folders as described below.

```
[N-TERMINAL-ACETYLATION]/
 |5-fold-data/
 |  |__1.fa fasta file
 |  |__2.fa fasta file
 |  |__3.fa fasta file
 |  |__4.fa fasta file
 |  |__5.fa fasta file
 |
 |test_data/
 |  |__test.fa fasta file
 |
 |model/
 |  |__checkpoint.pt pt file
```
## Training
After prepared the data,  you can use the `train.py` to train  the custom model. 
```python train.py```
## Testing
Befor making prediction on your own model, please make sure the test data is in the location specified above. Then the custom model can use by the `test.py`
