import pandas as pd
import dill
import numpy as np
from collections import defaultdict

##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})

    med_pd.drop(columns=['ROW_ID','DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# ATC3-to-drugname
def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[['ATC3', 'DRUG']].values:
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)

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
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    def append_title(dict_file, diag_pd):
        ## columns: "ROW_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"
        diag_dict = pd.read_csv(dict_file)
        diag_dict.drop(columns=['ROW_ID', 'SHORT_TITLE'], inplace=True)
        diag_pd = diag_pd.merge(diag_dict, on='ICD9_CODE')
        return diag_pd

    diag_pd = filter_2000_most_diag(diag_pd)

    diag_pd = append_title('D_ICD_DIAGNOSES.csv', diag_pd)
    diag_pd = diag_pd.rename(columns={'LONG_TITLE':'ICD9_TEXT'})

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    def append_title_(dict_file, pro_pd):
        pro_dict = pd.read_csv(dict_file, dtype={'ICD9_CODE':'category'})
        pro_dict.drop(columns=['ROW_ID', 'SHORT_TITLE'], inplace=True)
        pro_pd = pro_pd.merge(pro_dict, on='ICD9_CODE')
        return pro_pd

    pro_pd = append_title_('D_ICD_PROCEDURES.csv', pro_pd)
    pro_pd = pro_pd.rename(columns={'LONG_TITLE':'PRO_TEXT', 'ICD9_CODE':'PRO_CODE'})

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

    diag_pd0 = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()
    diag_pd = diag_pd0.merge(diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_TEXT'].unique().reset_index(), on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd0 = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ATC3'].unique().reset_index()
    med_pd = med_pd0.merge(med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['SMILES'].unique().reset_index(), on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd0 = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['PRO_CODE'].unique().reset_index()
    pro_pd = pro_pd0.merge(pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['PRO_TEXT'].unique().reset_index(), on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    diag_pd['ICD9_TEXT'] = diag_pd['ICD9_TEXT'].map(lambda x: list(x))
    med_pd['SMILES'] = med_pd['SMILES'].map(lambda x: list(x))
    pro_pd['PRO_TEXT'] = pro_pd['PRO_TEXT'].map(lambda x: list(x))

    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    data['ATC3_num'] = data['ATC3'].map(lambda x: len(x))
    print(data.columns)  # ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ICD9_TEXT', 'ATC3', 'SMILES', 'PRO_CODE', 'PRO_TEXT', 'ATC3_num']
    # print(data['ICD9_TEXT'].values[0], data['PRO_TEXT'].values[0], data['SMILES'].values[0])

    return data


if __name__ == '__main__':

    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    med_file = 'PRESCRIPTIONS.csv'
    diag_file = 'DIAGNOSES_ICD.csv'
    procedure_file = 'PROCEDURES_ICD.csv'

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
    # print(med_pd)  # 809563 rows x 5 columns

    # med to SMILES mapping
    atc3toDrug = ATC3toDrug(med_pd)
    druginfo = pd.read_csv(drugbankinfo)
    atc3tosmi = atc3toSMILES(atc3toDrug, druginfo)
    dill.dump(atc3tosmi, open('./output/atc3toSMILES_iii.pkl','wb'))

    med_pd = med_pd[med_pd.ATC3.isin(atc3tosmi.keys())]
    med_pd['SMILES'] = med_pd['ATC3'].map(lambda x: '\t'.join(atc3tosmi[x]))
    print(med_pd)  # 803488 rows x 6 columns
    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file)
    print(diag_pd)  # 609601 rows x 4 columns

    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    print(pro_pd)  # 226636 rows x 4 columns

    print ('complete procedure processing')

    # combine
    data0 = combine_process(med_pd, diag_pd, pro_pd)
    data0.to_pickle('./output/data_iii_sym0.pkl')
    print('complete combining')

