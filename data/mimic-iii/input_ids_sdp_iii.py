import dill
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

data = dill.load(open('./output/data_iii_sym1_mulvisit.pkl', 'rb'))
diag_txt = list(set([j for i in data['ICD9_TEXT'].values for j in list(i)]))
pro_txt = list(set([j for i in data['PRO_TEXT'].values for j in list(i)]))
sym_txt = list(set([j for i in data['SYM_LIST'].values for j in list(i)]))
print(len(diag_txt))
print(len(pro_txt))
print(len(sym_txt))


diag_txt_input_ids = {}
for i, txt in enumerate(diag_txt):
    enc = tokenizer(txt, return_tensors='pt')
    diag_txt_input_ids[txt] = enc.input_ids

dill.dump(diag_txt_input_ids, open('./output/diag_txt_input_ids_iii.pkl', 'wb'))


pro_txt_input_ids = {}
for i, txt in enumerate(pro_txt):
    enc = tokenizer(txt, return_tensors='pt')
    pro_txt_input_ids[txt] = enc.input_ids

dill.dump(pro_txt_input_ids, open('./output/pro_txt_input_ids_iii.pkl', 'wb'))


sym_txt_input_ids = {}
for i, sym in enumerate(sym_txt):
    enc = tokenizer(sym, return_tensors='pt')
    sym_txt_input_ids[sym] = enc.input_ids

dill.dump(sym_txt_input_ids, open('./output/sym_txt_input_ids_iii.pkl','wb'))