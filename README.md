# DrugRec NeurIPS'2022
This repository is a PyTorch implementation of our NeurIPS 2022 paper **"Debiased, Longitudinal and Coordinated Drug Recommendation through Multi-Visit Clinic Records"**.

## Installation
We recommend installing these key packages with the following command.
```
pip install torch  # version >= '1.10.1+cu113'
pip install nltk
pip install transformers
pip install scikit-learn==0.24.2
conda install -c conda-forge rdkit
cd ./data/mimic-iii/fariseq && pip install ./
```
Other packages can generally be installed with `pip` or `conda` command.
## Data Processing
### Step 0: Certificate registration and input preparation
Get the certificate first, and then download the MIMIC-III and MIMIC-IV datasets.
+ MIMIC-III: https://physionet.org/content/mimiciii/1.4/
+ MIMIC-IV: https://physionet.org/content/mimiciv/

Get the following input files from https://github.com/ycq091044/SafeDrug and put them in the folder **./data/input/**.
+ RXCUI2atc4.csv: NDC-RXCUI-ATC4 mapping
+ rxnorm2RXCUI.txt: NDC-RXCUI mapping
+ drugbank_drugs_info.csv: Drugname-SMILES mapping
+ drug-atc.csv: CID-ATC mapping
+ drug-DDI.csv: DDI information coded by CID

Taking MIMIC-III as an example, its data processing includes the following five steps. All generated files will be placed in the output folder **./data/mimic-iii/output/**.
### Step 1: Load the data and merge the original tables.
After downloading the raw dataset, put these required files in the folder path: **./data/mimic-iii/**.
+ DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv, PRESCRIPTIONS.csv (diagnosis, procedure, prescription information)
+ D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv (dictionary tables for diagnosis and procedure)
+ NOTEEVENTS.csv, ADMISSIONS.csv (used for symptom extraction)

Run **processing_iii.py** to generate **data_iii_sym0.pkl**, the merged data frame without symptom information.
```
cd ./data/mimic-iii/
python processing_iii.py
```

### Step 2: Symptom extraction
Run **sym_tagger_iii.py** to generate **data_iii_sym1_mulvisit.pkl** by extracting symptom information from clinical notes and admission tables, and removing the patients with only a single visit or no symptom.
```
python sym_tagger_iii.py
```

### Step 3: Tokenization for symptom, diagnosis, procedure, and medication
We employ the Clinical BERT (https://github.com/EmilyAlsentzer/clinicalBERT) to tokenize the text of symptoms, diagnoses, and procedures by running **input_ids_sdp_iii.py**. 

We employ a roberta-large based pre-trained model (https://github.com/microsoft/DVMP, checkpoints need to be registered by their authors) to tokenize SMILES strings of medications and encode them by running **input_smiles_iii.py**.
```
python input_ids_sdp_iii.py
python input_smiles_iii.py
```

### Step 4: Additional symptom information
Run **sym_info_iii.py** to generate additional symptom information used for the input of model training and inference.
```
python sym_info_iii.py
```

### Step 5: Generate multi-visit records and DDI adjacency matrix
Generate **records_final_iii.pkl** and **ddi_A_iii.pkl** by running **gen_records_ddi.py**.
```
python gen_records_ddi.py
```

## Model Training and Inference
Here is the key argument:
```
usage: main.py [--Test] [--model_name MODEL_NAME]
               [--resume_path RESUME_PATH] [--lr LR]
               [--target_ddi TARGET_DDI] [--kp KP] [--dim DIM]
               [--CI] [--multivisit]

optional arguments:
  --Test                test mode
  --model_name MODEL_NAME
                        model name
  --resume_path RESUME_PATH
                        resume path
  --lr LR               learning rate
  --target_ddi TARGET_DDI
                        target ddi
  --kp KP               coefficient of P signal
  --dim DIM             dimension
  --CI                  causal inference (ATE) loss
  --mulitivisit         multi visit or single visit
```

To train a DrugRec model from scratch, run the following command.
```
python main.py --model_name [YOUR_MODEL_NAME]
```

To evaluate an existing DrugRec model, you can place the model to the corresponding *resume_path* and run the following command.
```
python main.py --Test --model_name [YOUR_MODEL_NAME] --resume_path [YOUR_RESUME_PATH]
```


## Citation
```bibtex
@inproceedings{sun2022debiased,
title={Debiased, Longitudinal and Coordinated Drug Recommendation through Multi-Visit Clinic Records},
author={Sun, Hongda and Xie, Shufang and Li, Shuqi and Chen, Yuhan and Wen, Ji-Rong and Yan, Rui},
booktitle={Advances in Neural Information Processing Systems},
year={2022}
}
```

Feel free to contact me sunhongda98@ruc.edu.cn for any question.

Partial credit to previous reprostories:
+ https://github.com/ycq091044/SafeDrug
+ https://github.com/sjy1203/GAMENet


