import pandas as pd
import dill
import numpy as np
from collections import defaultdict


##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'}, usecols=['SUBJECT_ID','HADM_ID', 'STARTTIME', 'NDC', 'DRUG'])

    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['STARTTIME'] = pd.to_datetime(med_pd['STARTTIME'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'STARTTIME'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd


# ATC3-to-drugname
def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[['ATC3', 'DRUG']].values:
        if atc3 in atc3toDrugDict:
            if drugname not in atc3toDrugDict[atc3]:
                atc3toDrugDict[atc3].append(drugname)
        else:
            atc3toDrugDict[atc3] = []
            atc3toDrugDict[atc3].append(drugname)

    return atc3toDrugDict

def atc3toSMILES(ATC3toDrugDict, druginfo):
    drug2smiles = {}
    atc3tosmiles = {}
    for drugname, smiles in druginfo[['name', 'moldb_smiles']].values:
        if type(smiles) == type('a'):
            drug2smiles[drugname] = smiles
    for atc3, drug in ATC3toDrugDict.items():
        temp = []
        for d in drug:
            try:
                temp.append(drug2smiles[d])
            except:
                pass
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3]
    
    return atc3tosmiles


# medication mapping
def codeMapping2atc4(med_pd, RXCUI2atc4_file, rxnorm2RXCUI_file):
    with open(rxnorm2RXCUI_file, 'r') as f:
        rxnorm2RXCUI = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(rxnorm2RXCUI)
    med_pd.dropna(inplace=True)

    rxnorm2atc4 = pd.read_csv(RXCUI2atc4_file)
    rxnorm2atc4 = rxnorm2atc4.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc4.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc4, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: x[:4])
    med_pd = med_pd.rename(columns={'ATC4':'ATC3'})
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# visit >= 2
def process_visit_lg2(med_pd):
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a 

# most common medications
def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['ATC3']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['ATC3'].isin(med_count.loc[:299, 'ATC3'])]
    
    return med_pd.reset_index(drop=True)

##### process diagnosis #####
def diag_process(diag_file):
    ## columns: 'SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD_CODE', 'ICD_VERSION'
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID', 'SEQ_NUM'], inplace=True)
    diag_pd['ICD_C_V'] = diag_pd['ICD_CODE']+'_'+diag_pd['ICD_VERSION'].map(str)
    diag_pd.drop(columns=['SEQ_NUM', 'ICD_CODE', 'ICD_VERSION'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD_C_V']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD_C_V'].isin(diag_count.loc[:1999, 'ICD_C_V'])]
        
        return diag_pd.reset_index(drop=True)

    def append_title(dict_file, diag_pd):
        ## columns: ICD_CODE,ICD_VERSION,LONG_TITLE
        diag_dict = pd.read_csv(dict_file)
        diag_dict['ICD_C_V'] = diag_dict['ICD_CODE']+'_'+diag_dict['ICD_VERSION'].map(str)
        diag_dict.drop(columns=['ICD_CODE', 'ICD_VERSION'], inplace=True)
        diag_pd = diag_pd.merge(diag_dict, on='ICD_C_V')
        return diag_pd

    diag_pd = filter_2000_most_diag(diag_pd)

    diag_pd = append_title('d_icd_diagnoses.csv', diag_pd)
    diag_pd = diag_pd.rename(columns={'LONG_TITLE':'ICD_TEXT'})

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    ## columns: SUBJECT_ID   HADM_ID  SEQ_NUM   CHARTDATE ICD_CODE  ICD_VERSION
    pro_pd = pd.read_csv(procedure_file)
    pro_pd.dropna(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd['ICD_C_V'] = pro_pd['ICD_CODE']+'_'+pro_pd['ICD_VERSION'].map(str)
    pro_pd.drop(columns=['SEQ_NUM', 'CHARTDATE', 'ICD_CODE', 'ICD_VERSION'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    def append_title_(dict_file, pro_pd):
        ## columns: ICD_CODE,ICD_VERSION,LONG_TITLE
        pro_dict = pd.read_csv(dict_file)
        pro_dict['ICD_C_V'] = pro_dict['ICD_CODE']+'_'+pro_dict['ICD_VERSION'].map(str)
        pro_dict.drop(columns=['ICD_CODE', 'ICD_VERSION'], inplace=True)
        pro_pd = pro_pd.merge(pro_dict, on='ICD_C_V')

        return pro_pd

    pro_pd = append_title_('d_icd_procedures.csv', pro_pd)
    pro_pd = pro_pd.rename(columns={'LONG_TITLE':'PRO_TEXT', 'ICD_C_V':'PRO_C_V'})

    return pro_pd


###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd0 = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD_C_V'].unique().reset_index()
    diag_pd = diag_pd0.merge(diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD_TEXT'].unique().reset_index(), on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd0 = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ATC3'].unique().reset_index()
    med_pd = med_pd0.merge(med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['SMILES'].unique().reset_index(), on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd0 = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['PRO_C_V'].unique().reset_index()
    pro_pd = pro_pd0.merge(pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['PRO_TEXT'].unique().reset_index(), on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd['ICD_C_V'] = diag_pd['ICD_C_V'].map(lambda x: list(x))
    med_pd['ATC3'] = med_pd['ATC3'].map(lambda x: list(x))
    pro_pd['PRO_C_V'] = pro_pd['PRO_C_V'].map(lambda x: list(x))    
    diag_pd['ICD_TEXT'] = diag_pd['ICD_TEXT'].map(lambda x: list(x))
    med_pd['SMILES'] = med_pd['SMILES'].map(lambda x: list(x))
    pro_pd['PRO_TEXT'] = pro_pd['PRO_TEXT'].map(lambda x: list(x))

    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    data['ATC3_num'] = data['ATC3'].map(lambda x: len(x))
    print(data.columns)  # ['SUBJECT_ID', 'HADM_ID', 'ICD_C_V', 'ICD_TEXT', 'ATC3', 'PRO_C_V', 'PRO_TEXT', 'ATC3_num']
    # print(data['ICD_TEXT'].values[0], data['PRO_TEXT'].values[0], data['ATC3'].values[0])

    return data

def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['ICD_C_V'].values
    med = data['ATC3'].values
    pro = data['PRO_C_V'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#procedure', len(unique_pro))
    print('#med ', len(unique_med))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]

        d, p, y = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            d.extend(list(row['ICD_C_V']))
            p.extend(list(row['PRO_C_V']))
            y.extend(list(row['ATC3']))
        d, p, y = set(d), set(p), set(y)
        avg_diag += len(d)
        avg_pro += len(p)
        avg_med += len(y)
        avg_visit += visit_cnt
        if len(d) > max_diag:
            max_diag = len(d)
        if len(p) > max_pro:
            max_pro = len(p)
        if len(y) > max_med:
            max_med = len(y) 
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of vists ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of procedures ', max_pro)
    print('#max of medicines ', max_med)
    print('#max of visit ', max_visit)

def statistics_with_sym(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['ICD_C_V'].values
    med = data['ATC3'].values
    pro = data['PRO_C_V'].values
    symp = data['SYM_LIST'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    unique_symp = set([j for i in symp for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#procedure', len(unique_pro))
    print('#symptom', len(unique_symp))
    print('#med ', len(unique_med))
    
    avg_diag, avg_med, avg_pro, avg_symp, max_diag, max_med, max_pro, max_symp, cnt, max_visit, avg_visit = [0 for i in range(11)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]

        d, p, s, y = [], [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            d.extend(list(row['ICD_C_V']))
            p.extend(list(row['PRO_C_V']))
            s.extend(list(row['SYM_LIST']))
            y.extend(list(row['ATC3']))
        d, p, s, y = set(d), set(p), set(s), set(y)
        avg_diag += len(d)
        avg_pro += len(p)
        avg_symp += len(s)
        avg_med += len(y)
        avg_visit += visit_cnt
        if len(d) > max_diag:
            max_diag = len(d)
        if len(p) > max_pro:
            max_pro = len(p)
        if len(s) > max_symp:
            max_symp = len(s)
        if len(y) > max_med:
            max_med = len(y) 
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of symptoms ', avg_symp/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of vists ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of procedures ', max_pro)
    print('#max of symptoms ', max_symp)
    print('#max of medicines ', max_med)
    print('#max of visit ', max_visit)


##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD_C_V'])
        med_voc.add_sentence(row['ATC3'])
        pro_voc.add_sentence(row['PRO_C_V'])
    
    return diag_voc, med_voc, pro_voc

# create final records
def create_patient_record(df, diag_voc, pro_voc, med_voc):
    diag_txt_input_ids = dill.load(open('./diag_txt_input_ids_iv.pkl','rb'))
    pro_txt_input_ids = dill.load(open('./pro_txt_input_ids_iv.pkl','rb'))
    smiles_input_ids = dill.load(open('./SMILES_tok_ids_iv.pkl','rb'))
    symp_txt_input_ids = dill.load(open('./sym_txt_input_ids_iv.pkl','rb'))
    records, records0 = [], []
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient, patient0 = [], []
        for index, row in item_df.iterrows():
            admission, admission0 = [], []
            admission.append([diag_voc.word2idx[i] for i in row['ICD_C_V']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_C_V']])
            admission.append([med_voc.word2idx[i] for i in row['ATC3']])
            admission0.append([diag_txt_input_ids[diag] for diag in row['ICD_TEXT']])
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
    
    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    med_file = 'prescriptions.csv'
    diag_file = 'diagnoses_icd.csv'
    procedure_file = 'procedures_icd.csv'

    # input files
    RXCUI2atc4_file = '../input/RXCUI2atc4.csv' 
    rxnorm2RXCUI_file = '../input/rxnorm2RXCUI.txt'
    drugbankinfo = '../input/drugbank_drugs_info.csv'

    # for med
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True)

    med_pd = codeMapping2atc4(med_pd, RXCUI2atc4_file, rxnorm2RXCUI_file)
    med_pd = filter_300_most_med(med_pd)

    # med to SMILES mapping
    atc3toDrug = ATC3toDrug(med_pd)
    druginfo = pd.read_csv(drugbankinfo)
    atc3tosmi = atc3toSMILES(atc3toDrug, druginfo)
    dill.dump(atc3tosmi, open('./output/atc3toSMILES_iv.pkl','wb'))
    
    med_pd = med_pd[med_pd.ATC3.isin(atc3tosmi.keys())]
    med_pd['SMILES'] = med_pd['ATC3'].map(lambda x: '\t'.join(atc3tosmi[x]))
    print(med_pd)      
    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file)
    print(diag_pd)  # SUBJECT_ID  HADM_ID  ICD_C_V  ICD_TEXT
    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    print(pro_pd)   # SUBJECT_ID  HADM_ID  PRO_C_V  PRO_TEXT
    print ('complete procedure processing')

    # combine
    data0 = combine_process(med_pd, diag_pd, pro_pd)
    print ('complete combining')
    data0.to_pickle('./output/data_iv_sym0.pkl')
    statistics(data0)



    '''
    ############ after symptom extraction ##############
    # create vocab for final data with symptoms
    data = dill.load(open('./output/data_iv_sym1_mulvisit.pkl', 'rb'))
    statistics_with_sym(data)
    diag_voc, pro_voc, med_voc = create_str_token_mapping(data)
    print ("obtained voc")
    vocabulary_file = "./output/voc_iv_sym1_mulvisit.pkl"
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open(vocabulary_file,'wb'))
    '''
    


    '''
    ############ generate multi-visit records and DDI adj matrix ##############
    ddi_adjacency_file = "./output/ddi_A_iv.pkl"
    ehr_sequence_file = "./output/records_final_iv.pkl"

    data = dill.load(open('./output/data_iv_sym1_mulvisit.pkl', 'rb'))
    voc = dill.load(open('./output/voc_iv_sym1_mulvisit.pkl','rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    # create ehr sequence data
    records, records0 = create_patient_record(data, diag_voc, pro_voc, med_voc)
    print ("obtain ehr sequence data")
    dill.dump(obj=records0, file=open(ehr_sequence_file, 'wb'))
    # dill.dump(obj=records, file=open('./output/records_ori_iv.pkl', 'wb'))  # it may be used in some baselines.

    # create ddi adj matrix
    ddi_file = '../input/drug-DDI.csv'
    cid2atc6_file = '../input/drug-atc.csv'
    ddi_adj = get_ddi_matrix(med_voc, ddi_file, cid2atc6_file)
    print ("obtain ddi adj matrix")
    dill.dump(ddi_adj, open(ddi_adjacency_file, 'wb'))
    '''
    


