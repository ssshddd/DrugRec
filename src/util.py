from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import sys
import warnings
import dill
warnings.filterwarnings('ignore')


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0.0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0.0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0.0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0.0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0.0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0.0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, path):
    # ddi rate
    ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0.0
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0.0
    return dd_cnt / all_cnt


from satsolver import *
def Post_DDI(pred_result, ddi_pair, ehr_train_pair):
    # input: numpy (seq, voc_size) (0~1)
    post_result, y_pred = np.zeros_like(pred_result), np.zeros_like(pred_result)
    y_pred[pred_result >= 0.5] = 1
    for k in range(pred_result.shape[0]):
        pred_idx = np.nonzero(y_pred[k])[0]
        tmp_dict = {idx:n for n, idx in enumerate(pred_idx)}
        pred_prob = {str(n):pred_result[k, idx] for n, idx in enumerate(pred_idx)}
        formula = two_cnf(pred_prob)
        ddi_list = []
        for (i, j) in ddi_pair:
            if i in pred_idx and j in pred_idx and i<j:
                if (i, j) not in ehr_train_pair:
                    # print(['~' + str(tmp_dict[i]), '~' + str(tmp_dict[j])])
                    formula.add_clause(['~' + str(tmp_dict[i]), '~' + str(tmp_dict[j])])
                    if i not in ddi_list:
                        ddi_list.append(i)
                    if j not in ddi_list:
                        ddi_list.append(j)
        f = two_sat_solver(formula)
        if f:
            pos = [list(tmp_dict.keys())[int(n)] for n, x in f.items() if x == 1] + [idx for idx in pred_idx if idx not in ddi_list]
            post_result[k, pos] = 1
        else:
            post_result[k] = pred_result[k]

    return post_result


from scipy.stats import t

def ttest(mean1,mean2,std1,std2,n1=10,n2=10):
    mu = mean1 - mean2
    df = n1 + n2 - 2
    denominator = (((n1-1)*std1**2 + (n2-1)*std2**2) * (1/n1+1/n2) / (df))**0.5
    tval = mu/denominator
    if mu > 0:
        pval = (1-t.cdf(tval, df=df))*2
    else:
        pval = t.cdf(tval, df=df)*2
    return pval