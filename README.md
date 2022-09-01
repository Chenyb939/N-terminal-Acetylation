<font size="6">MTNA: A Deep Learning based Predictor for
Identifying Multiple Types of N-terminal Protein
Acetylated Sites</font>
![structure](https://github.com/Chenyb939/N-terminal-Acetylation/blob/master/structure.png?raw=true)
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

```conda env create -f environment.yaml```

This will create a new environment named `MTNA`. This environment needs to be activated via `conda activate MTNA` before using any of the code in this repository.

# Quick start

The data schema to run our code is as follows: 
```
[N-TERMINAL-ACETYLATION]/
 |5-fold-data/
 |  |__1.fa  # train data
 |  |__2.fa  # train data
 |  |__3.fa  # train data
 |  |__4.fa  # train data
 |  |__5.fa  # train data
 |
 |test_data/
 |  |__test.fa  # test data
 |
 |model/
 |  |__checkpoint.pt  # model file
```

We provide a human pre-trained model at the fold `model`. This provided model is trained on the dataset we discuss in the manuscript. Running `test.py`(see below) will use pre-trained model. 
```
python test.py
```

# Usage
## Data preparation
Before training the data shuould be pleased in the corresponding folders as described in [quick start](#quick-start).

The remaining files required for MTNA are shown below.
```
[N-TERMINAL-ACETYLATION]/
 |aaindex_preprocessing/
 |  |__aaindex_feature_to_dp.txt  # raw aaindex feature
 |  |__aaindex_feature.txt  # processed aaindex feature
 |  |__aaindex_normalization.py  # python file for process aaindex feature
 |
 |check_data.py  #python file for check raw data find in pdb
 |divide_5-fold-data.py  #python file for divided 5 fold
 |dp.py  #python file for processing data after check
 |environment.yaml  # environment file 
 |README.md  # README
 |test.py  #python file for testing
 |train.py  #python file for training
```

## Training
After prepared the data,  you can use the `train.py` to train  the custom model. 
```python train.py```
## Testing
Befor making prediction on your own model, please make sure the test data is in the location specified above. Then the custom model can use by the `test.py`
