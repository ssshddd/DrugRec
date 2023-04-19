from rdkit import Chem
import dill
import os
import numpy as np
from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
import re
from tqdm import tqdm
import torch
SMI_PATTERN = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])")

roberta = RobertaModel.from_pretrained(
	'/home/sunhd/DrugRec/data/pubchem',
	checkpoint_file='pubchem_roberta_10M_large.pt'
)
roberta.eval()


''' 
canonize the SMILES with RDkit and tokenize it with https://doi.org/10.1021/acscentsci.9b00576.
'''
def tokenize(x):
    try:
        mol = Chem.MolFromSmiles(x)
        x = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        pass

    x = ' '.join(SMI_PATTERN.findall(x))
    x = "<s> " + x + " </s>"
    return x


atc3toSMILES_file = './output/atc3toSMILES_iii.pkl'
atc3tosmi = dill.load(open(atc3toSMILES_file, 'rb'))

smiles_tok = {}
for smi_list in atc3tosmi.values():
    for smi in smi_list:
        if smi not in smiles_tok:
            smiles_tok[smi] = tokenize(smi)

print(list(smiles_tok.items())[:5], len(smiles_tok))  # 234

smiles_tok_ids = {}
for smi, tok in smiles_tok.items():
    enc = roberta.task.source_dictionary.encode_line(tok, append_eos=False, add_if_not_exist=False).long()
    enc1 = collate_tokens([enc.long()], pad_idx=1, pad_to_length=300)[0]
    smiles_tok_ids[smi] = np.asarray(enc1)


print(list(smiles_tok_ids.items())[0], len(smiles_tok_ids))  # 234
dill.dump(smiles_tok_ids, open('./output/SMILES_tok_ids_iii.pkl', 'wb'))



voc = dill.load(open('./output/voc_iii_sym1_mulvisit.pkl', 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
atc_list = med_voc.idx2word.values()
# print(med_voc.idx2word, len(med_voc.idx2word))  # 112

input_smiles_reps_total = torch.zeros(len(atc_list), 1024)
for k, atc in tqdm(enumerate(atc_list)):
    with torch.no_grad():
        input_smiles_reps = [roberta.extract_features(torch.LongTensor(smiles_tok_ids[smi]))[:,0] for smi in atc3tosmi[atc]]
        input_smiles_reps_total[k] = sum(input_smiles_reps)/len(input_smiles_reps)

print(input_smiles_reps_total.shape)  # torch.Size([112, 1024])
dill.dump(input_smiles_reps_total, open('./output/input_smiles_init_rep_iii.pkl', 'wb'))
