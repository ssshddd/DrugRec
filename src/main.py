import dill
import numpy as np
import argparse
from collections import defaultdict
from torch.optim import Adam
import os
import torch
import torch.nn as nn
import time
from models import *
from util import *
import torch.nn.functional as F
import math


# setting
model_name = 'DrugRec_mimic-iii'
resume_path = 'saved/{}/drugrec_mimic3.model'.format(model_name)

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Running settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', default=False, action='store_true', help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--target_ddi', type=float, default=0.05, help='target ddi')
parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--epoch', type=int, default=200, help='training epoches')
parser.add_argument('--decay', type=float, default=0, help='weight decay')
parser.add_argument('--warmup', type=bool, default=True, help='lr warm up')
parser.add_argument('--w_ddi', type=float, default=0.5, help='weight of ddi loss')
parser.add_argument('--CI', type=bool, default=True, help='causal inference loss')
parser.add_argument('--w_ci', type=float, default=0.0005, help='weight of causal inference loss')
parser.add_argument('--k_ci', type=int, default=5, help='k symptom samples of causal inference loss')
parser.add_argument('--k_mul', type=int, default=3, help='k_mul previous visits for multi-visit')
parser.add_argument('--multivisit', type=bool, default=True, help='multi-visit or single-visit')
parser.add_argument('--mulhistory', type=bool, default=False, help='concat history representations or not')
parser.add_argument('--fix_smi_rep', type=bool, default=True, help='fix input_smiles_init_rep when training')

args = parser.parse_args()
print(vars(args))

# inference stage
def eval(model, data_eval, voc_size, epoch, ddi_adj_path, ehr_train_pair):
    model.eval()
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_pair = np.where(ddi_adj == 1)
    ddi_pair = [(ddi_pair[0][i], ddi_pair[1][i]) for i in range(len(ddi_pair[0]))]

    smm_record = []
    y_gt_list = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0.0, 0.0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        target_output, _ = model(input, 'eval')
        for adm_idx, adm in enumerate(input):

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[-1]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output_ = F.sigmoid(target_output[[adm_idx]]).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output_)
            
            # prediction med set
            y_pred_tmp = target_output_.copy()

            # Post process DDI with 2-SAT
            y_pred_tmp = Post_DDI(y_pred_tmp.reshape(1,-1), ddi_pair, ehr_train_pair).reshape(-1)

            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0

            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        y_gt_list.append(y_gt)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path=ddi_adj_path)

    llprint('\nDDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def main():
    
    # load data
    data_path = '../data/mimic-iii/output/records_final_iii.pkl'
    voc_path = '../data/mimic-iii/output/voc_iii_sym1_mulvisit.pkl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = dill.load(open(data_path, 'rb'))

    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    # split train/val/test set
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
      
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    ddi_adj_path = '../data/mimic-iii/output/ddi_A_iii.pkl'
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))

    # construct cooccurence matrix from train data, used for loss_pair
    drug_co_train = np.zeros((voc_size[-1], voc_size[-1]))
    for patient in data_train:
        for adm in patient:
            med_set = adm[-1]
            for med_i in med_set:
                for med_j in med_set:
                    if med_j<=med_i:
                        continue
                    drug_co_train[med_i, med_j] += 1
                   
    drug_train_pair = np.zeros_like(drug_co_train)
    drug_train_pair[drug_co_train >= 1000] = 1
    ehr_train_pair = np.where(drug_train_pair == 1)
    ehr_train_pair = [(ehr_train_pair[0][i], ehr_train_pair[1][i]) for i in range(len(ehr_train_pair[0]))]

    input_smiles_init_rep = dill.load(open('../data/mimic-iii/output/input_smiles_init_rep_iii.pkl','rb'))
    sym_count = dill.load(open('../data/mimic-iii/output/sym_count_iii.pkl','rb'))
    sym_input_ids = dill.load(open('../data/mimic-iii/output/sym_txt_input_ids_iii.pkl','rb'))
    sym2idx = dill.load(open('../data/mimic-iii/output/sym2idx_iii.pkl','rb'))
    sym_comatrix = np.load('../data/mimic-iii/output/sym_comatrix_iii.npy')
    sym_information = [sym_count, sym2idx, sym_comatrix, sym_input_ids]

    model = DrugRec_all(args, sym_information, ddi_adj, input_smiles_init_rep, emb_dim=args.dim, device=device)
    # model = DrugRec_nosym(args, ddi_adj, input_smiles_init_rep, emb_dim=args.dim, device=device)
    print(model)

    model.to(device=device)
    if torch.cuda.device_count() > 1:
        print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
    model = nn.DataParallel(model, dim = 0)

    if args.Test:
        # inference stage
        model.load_state_dict(torch.load(open(args.resume_path, 'rb'), map_location=torch.device('cpu')))
        model.to(device=device)
        tic = time.time()

        result = []
        for _ in range(10):
            test_sample = np.random.choice(data_test, round(len(data_test) * 0.8), replace=True)
            with torch.set_grad_enabled(False):
                ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, test_sample, voc_size, 0, ddi_adj_path, ehr_train_pair)
                result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
        
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print (outstring)

        print ('test time: {}'.format(time.time() - tic))
        return 

    # training stage
    optimizer = Adam(list(model.parameters()), lr=args.lr, weight_decay=args.decay)

    # start iterations
    history = defaultdict(list)
    best_ja_epoch, best_auc_epoch, best_f1_epoch, best_ja, best_auc, best_f1 = 0, 0, 0, 0, 0, 0 

    EPOCH = args.epoch

    if args.warmup:
        warmup_epoch = int(EPOCH * 0.06)
        iter_per_epoch = len(data_train)
        warm_up_with_cosine_lr = lambda epoch: epoch / (warmup_epoch * iter_per_epoch) if epoch <= (warmup_epoch * iter_per_epoch) else 0.5 * (math.cos((epoch - warmup_epoch * iter_per_epoch) / ((EPOCH - warmup_epoch) * iter_per_epoch) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    learning_rates = []
    
    k = 0
    pair_dict = {}
    pair_size = int(voc_size[2]*(voc_size[2]-1)/2)
    trans_p1, trans_p2 = torch.zeros(pair_size), torch.zeros(pair_size)
    for i in range(voc_size[2]):
        for j in range(voc_size[2]):
            if j>i:
                pair_dict[(i,j)] = k
                trans_p1[k], trans_p2[k] = i,j
                k+=1
    assert len(pair_dict) == pair_size
    trans_pair_iii = (trans_p1.long(), trans_p2.long())

    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch))
        
        model.train()
        for step, input in enumerate(data_train):
            loss = np.zeros(len(input))
            loss_bce_target = np.zeros((len(input), voc_size[2]))
            loss_multi_target = np.full((len(input), voc_size[2]), -1)
            loss_bce_pair_target = np.zeros((len(input), int(voc_size[2]*(voc_size[2]-1)/2)))

            for idx, adm in enumerate(input):
                
                loss_bce_target[idx, adm[-1]] = 1 
                
                for i, item in enumerate(adm[-1]):
                    loss_multi_target[idx][i] = item

                for i in range(len(adm[-1])):
                    for j in range(len(adm[-1])):
                        if j>i:
                            loss_bce_pair_target[idx, pair_dict[i,j]] = 1

            if args.CI:
                result, neg_pred_prob, loss_ddi, pred_prob, pred_prob0 = model(input, 'train')
                loss_ci = - torch.log(F.sigmoid(torch.sign(torch.FloatTensor(loss_bce_target-0.5).to(device)) * (pred_prob - pred_prob0))).mean()
            else:
                result, neg_pred_prob, loss_ddi = model(input, 'train')

            result_pair = torch.zeros(len(input), int(voc_size[2]*(voc_size[2]-1)/2))
            for i in range(len(input)):
                result_pair[i] = neg_pred_prob[i][trans_pair_iii]

            loss_bce = F.binary_cross_entropy_with_logits(result, torch.FloatTensor(loss_bce_target).to(device))
            loss_multi = F.multilabel_margin_loss(F.sigmoid(result), torch.LongTensor(loss_multi_target).to(device))
            loss_bce_pair = F.binary_cross_entropy(result_pair.to(device), torch.FloatTensor(loss_bce_pair_target).to(device))

            result = F.sigmoid(result).detach().cpu().numpy()
            
            result[result >= 0.5] = 1
            result[result < 0.5] = 0

            y_label = np.where(result == 1)[0]

            current_ddi_rate = ddi_rate_score([[y_label]], path=ddi_adj_path)

            if current_ddi_rate <= args.target_ddi:
                loss = loss_bce + 0.1 * loss_multi + loss_bce_pair
            else:
                loss = loss_bce + 0.1 * loss_multi + args.w_ddi * loss_ddi + loss_bce_pair

            if args.CI:
                loss = loss + args.w_ci * loss_ci

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if args.warmup:
                scheduler.step()
                learning_rates.append(scheduler.get_last_lr()[0])

            if step % 10 == 0:
                if args.CI:
                    llprint('\rtraining step: {} / {}, loss: {} (loss_ci: {}, loss_ddi: {}, loss_bce_pair: {}), time: {}s'.format(step, len(data_train), str(loss.item())[:7], str(loss_ci.item())[:7], str(loss_ddi.item())[:7], str(loss_bce_pair.item())[:7], str(time.time()-tic)[:5]))
                else:
                    llprint('\rtraining step: {} / {}, loss: {} (loss_ddi: {}), time: {}s'.format(step, len(data_train), str(loss.item())[:7], str(loss_ddi.item())[:7], str(time.time()-tic)[:5]))

        print(model_name)
        tic2 = time.time()
        with torch.set_grad_enabled(False):
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, ddi_adj_path, ehr_train_pair)
            print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)
            history['med'].append(avg_med)

            torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
                'Epoch_{}_TARGET_{:.2}_JA_{:.4}_AUC_{:.4}_F1_{:.4}_DDI_{:.4}.model'.format(epoch, args.target_ddi, ja, prauc, avg_f1, ddi_rate)), 'wb'))

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval, voc_size, epoch, ddi_adj_path, ehr_train_pair)
            if epoch >= 5:
                print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                    np.mean(history['ddi_rate'][-5:]),
                    np.mean(history['med'][-5:]),
                    np.mean(history['ja'][-5:]),
                    np.mean(history['avg_f1'][-5:]),
                    np.mean(history['prauc'][-5:])
                    ))
            
                if best_ja < np.mean(history['ja'][-5:]):
                    best_ja_epoch = epoch
                    best_ja = ja

                print ('best_ja_epoch: {}'.format(best_ja_epoch))
                
                if best_auc < np.mean(history['prauc'][-5:]):
                    best_auc_epoch = epoch
                    best_auc = prauc

                print ('best_auc_epoch: {}'.format(best_auc_epoch))
                
                if best_f1 < np.mean(history['avg_f1'][-5:]):
                    best_f1_epoch = epoch
                    best_f1 = avg_f1

                print ('best_f1_epoch: {}'.format(best_f1_epoch))

    dill.dump(learning_rates, open("saved/{}/learning_rates".format(model_name), 'wb'))

if __name__ == '__main__':
    main()
    print(model_name)
