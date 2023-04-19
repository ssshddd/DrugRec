import pandas as pd
import dill
import numpy as np
from collections import defaultdict


# create final records
def create_patient_record(df, diag_voc, pro_voc, med_voc):
    diag_txt_input_ids = dill.load(open('./output/diag_txt_input_ids_iii.pkl','rb'))
    pro_txt_input_ids = dill.load(open('./output/pro_txt_input_ids_iii.pkl','rb'))
    smiles_input_ids = dill.load(open('./output/SMILES_tok_ids_iii.pkl','rb'))
    symp_txt_input_ids = dill.load(open('./output/sym_txt_input_ids_iii.pkl','rb'))
    records, records0 = [], []
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient, patient0 = [], []
        for index, row in item_df.iterrows():
            admission, admission0 = [], []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['ATC3']])
            admission0.append([diag_txt_input_ids[diag] for diag in row['ICD9_TEXT']])
            admission0.append([pro_txt_input_ids[pro] for pro in row['PRO_TEXT']])
            admission0.append([symp_txt_input_ids[symp] for symp in row['SYM_LIST']])
            smi_tok_ids_ = []
            for smiles in row['SMILES']:
                smi_tok_ids = []
                for smi in smiles.split('\t'):
                    smi_tok_ids.append(smiles_input_ids[smi])
                smi_tok_ids_.append(smi_tok_ids)
            admission0.append(smi_tok_ids_)
            admission0.append([med_voc.word2idx[i] for i in row['ATC3']])
            patient.append(admission)
            patient0.append(admission0)
        records.append(patient)
        records0.append(patient0) 
    
    return records, records0
        
# get ddi matrix
def get_ddi_matrix(med_voc, ddi_file, cid2atc6_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)
    
    with open(cid2atc6_file, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
            
    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect 
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name'])\
        .size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)

    # ddi adj
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                
                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1

    return ddi_adj


if __name__ == '__main__':
    ddi_adjacency_file = "./output/ddi_A_iii.pkl"
    ehr_sequence_file = "./output/records_final_iii.pkl"

    data = dill.load(open('./output/data_iii_sym1_mulvisit.pkl', 'rb'))
    voc = dill.load(open('./output/voc_iii_sym1_mulvisit.pkl','rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print ("obtain voc")

    # create ehr sequence data
    records, records0 = create_patient_record(data, diag_voc, pro_voc, med_voc)
    print ("obtain ehr sequence data")
    dill.dump(obj=records0, file=open(ehr_sequence_file, 'wb'))
    # dill.dump(obj=records, file=open('./output/records_ori_iii.pkl', 'wb'))  # it may be used in some baselines.


    # create ddi adj matrix
    ddi_file = '../input/drug-DDI.csv'
    cid2atc6_file = '../input/drug-atc.csv'
    ddi_adj = get_ddi_matrix(med_voc, ddi_file, cid2atc6_file)
    print ("obtain ddi adj matrix")
    dill.dump(ddi_adj, open(ddi_adjacency_file, 'wb'))

