import numpy as np
import dill

data = dill.load(open('./output/data_iii_sym1_mulvisit.pkl', 'rb'))
sym_list = data.SYM_LIST.tolist()

sym_total = []
for sym in sym_list:
    sym_total += sym

sym_keys = list(set([j for i in sym_list for j in list(i)]))
sym2idx = {s:i for i, s in enumerate(sym_keys)}
dill.dump(sym2idx, open('./output/sym2idx_iii.pkl', 'wb'))

sym_count = {}
for sym in sym_keys:
    sym_count[sym] = sym_total.count(sym)
assert sum(sym_count.values()) == len(sym_total)
dill.dump(sym_count, open('./output/sym_count_iii.pkl', 'wb'))

co = np.zeros((len(sym_keys), len(sym_keys)))
for k, sym in enumerate(sym_list):
    if len(sym) > 0:
        for i in range(len(sym)-1):
            for j in range(i+1, len(sym)):
                co[sym2idx[sym[i]], sym2idx[sym[j]]] += 1
                co[sym2idx[sym[j]], sym2idx[sym[i]]] += 1


np.save('./output/sym_comatrix_iii.npy', co)