import pandas as pd
import dill

import re
import nltk
import dill
from nltk.corpus import stopwords

symptoms_list = dill.load(open('../input/symptoms_list.pkl', 'rb'))
symptoms_list = list(set([sym.lower() for sym in symptoms_list]))
# print(symptoms_list, len(symptoms_list))  # 887

def match_symptoms(search):

    """re for symptom matching"""

    symptomList = []

    for item in search:
        pattern = r'<c>(.*?)<\/c>'
        noun_prases = re.findall(pattern, item, flags=0)
        for s in noun_prases:
            if s.lower() in symptoms_list:
                symptomList.append(s)

    return symptomList


def main_text(full_text):
    sentence = full_text.replace('"','')

    sentences = nltk.sent_tokenize(sentence)
    word_tokens = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in word_tokens]

    def chunk(text):
        """ Return the noun phrases using reguler expressoins"""

        '''pattern = ['JJ', 'NN', 'VB', 'NN']
            matches = []

            for i in range(len(tagged)):
                if tagged[i:i+len(pattern)] == pattern:
                    matches.append(sentences[i:i+len(pattern)])

            matches = [' '.join(match) for match in matches]
            print(matches)'''

        grammar = """NP: {<V.*>+(<RP?><NN>)?}
                    NP: {(<NN.*><DT>)?(<NN.*><IN>)?<NN.*>?<JJ.>*<NN.*>+}
                    NP: {<V.*>}
                    ENTITY: {<NN.*>}"""

        parser = nltk.RegexpParser(grammar)
        result = parser.parse(text)
        t_sent = ' '.join(word for word, pos in text)
        for subtree in result.subtrees():
            if subtree.label() == 'NP':
                noun_phrases_list = ' '.join(word for word, pos in subtree.leaves())
                t_sent = t_sent.replace(noun_phrases_list, "<c>"+noun_phrases_list+"</c>", 1)
        return t_sent
    
    chunk_sent = []
    for sentence in sentences:
        chunk_sent.append(chunk(sentence))
    return chunk_sent

def symptoms_tagger(x):

    search = main_text(x)

    tagged_symptom_list = match_symptoms(search)
    return list(set(tagged_symptom_list))


def text_to_symptom(text):
    text_list = text.split('\n')
    sym_list = []
    for i in range(len(text_list)):
        sym = symptoms_tagger(text_list[i])
        sym_list += sym
    return list(set([sym.lower() for sym in sym_list]))

def process_visit_lg2(med_pd):
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a


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
    pro_voc = Voc()
    med_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['ATC3'])
        pro_voc.add_sentence(row['PRO_CODE'])
    
    return diag_voc, pro_voc, med_voc


def statistics_with_sym(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['ICD9_CODE'].values
    med = data['ATC3'].values
    pro = data['PRO_CODE'].values
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
            d.extend(list(row['ICD9_CODE']))
            p.extend(list(row['PRO_CODE']))
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


from tqdm import tqdm
tqdm.pandas()
if __name__ == '__main__':
    notes = pd.read_csv('NOTEEVENTS.csv', usecols=['SUBJECT_ID','HADM_ID','CATEGORY','TEXT'])
    adm = pd.read_csv('ADMISSIONS.csv', usecols=['SUBJECT_ID','HADM_ID','DIAGNOSIS'])
    data = dill.load(open('./output/data_iii_sym0.pkl', 'rb'))

    notes1 = notes[notes.CATEGORY=='Discharge summary'].sort_values(by=['SUBJECT_ID', 'HADM_ID'])
    notes2 = notes1.groupby(by=['SUBJECT_ID','HADM_ID'])['TEXT'].apply('\n'.join).reset_index()

    data1 = data.merge(notes2, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data2 = data1.merge(adm, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    print(data2)

    data2['SYM_LIST'] = data2['TEXT'].progress_apply(text_to_symptom) + data2['DIAGNOSIS'].map(text_to_symptom)
    data2['SYM_len'] = data2['SYM_LIST'].map(len)

    data3 = data2[data2['SYM_len'] > 0].reset_index()
    data_lg2 = process_visit_lg2(data3).reset_index(drop=True)
    data = data3.merge(data_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True)
    data = data.drop(columns=['index', 'TEXT'])
    print(data)
    data.to_pickle('./output/data_iii_sym1_mulvisit.pkl')

    # create vocab for final data with symptoms
    statistics_with_sym(data)
    diag_voc, pro_voc, med_voc = create_str_token_mapping(data)
    print ("obtained voc")
    vocabulary_file = "./output/voc_iii_sym1_mulvisit.pkl"
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open(vocabulary_file,'wb'))

