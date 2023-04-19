import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import *


class DrugRec_nosym(nn.Module):
    def __init__(self, args, ddi_adj, input_smiles_init_rep, emb_dim=256, device=torch.device('cpu:0')):
        super(DrugRec_nosym, self).__init__()

        self.device = device
        self.args = vars(args)
        trans_dim = emb_dim
        self.transformer_dp = Transformer_Encoder(n_layer=2, hidden_size=trans_dim, num_attention_heads=4, vocab_size=28996, max_position_size=576, max_segment=40, word_emb_padding_idx=0, dropout=0.1)
        self.lin = nn.Sequential(nn.ReLU(), nn.Linear(trans_dim, emb_dim))

        self.diag_seq_enc = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.pro_seq_enc = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
 
        if self.args['fix_smi_rep']:
            self.med_rep = input_smiles_init_rep.to(device=self.device)
        else:
            self.med_rep = torch.tensor(input_smiles_init_rep, requires_grad=True).to(device=self.device)
        
        # ddi graph adj matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(self.device)

    def forward(self, input, type):
        DIAG_SEQ, PRO_SEQ = [], []

        for j, adm in enumerate(input):
            diag_ids_list = [diag.to(self.device) for diag in adm[0]]
            diag_representations = self.transformer_dp(diag_ids_list)
            pro_ids_list = [pro.to(self.device) for pro in adm[1]]
            pro_representations = self.transformer_dp(pro_ids_list)

            diag_rep = diag_representations.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
            pro_rep = pro_representations.mean(dim=1).unsqueeze(dim=0)
            DIAG_SEQ.append(diag_rep)
            PRO_SEQ.append(pro_rep)
        
        diag_seq = self.lin(torch.cat(DIAG_SEQ, dim=1)) #(1,seq,dim)
        pro_seq = self.lin(torch.cat(PRO_SEQ, dim=1)) #(1,seq,dim)

        diag_out,_ = self.diag_seq_enc(diag_seq)
        pro_out,_ = self.pro_seq_enc(pro_seq)
        patient_representations = torch.cat([diag_out, pro_out], dim=-1).squeeze(dim=0)  # (seq, dim*2)
        query = self.query(patient_representations) # (seq, dim)

        med_rep = self.mlp(self.med_rep)
        result = torch.mm(query, med_rep.t())  # [seq, voc_size]
        
        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = torch.einsum("nc,nk->nck",[neg_pred_prob, neg_pred_prob])  # [seq, voc_size, voc_size]

        batch_neg = 1/self.tensor_ddi_adj.shape[0] * neg_pred_prob.mul(self.tensor_ddi_adj).sum(dim=[1,2]).mean() # (seq,).mean()

        if type == 'train':
            return result, neg_pred_prob, batch_neg
        else:
            return result, batch_neg

    # def init_weights(self):
    #     """Initialize weights."""
    #     initrange = 0.1
    #     for item in self.embeddings:
    #         item.weight.data.uniform_(-initrange, initrange)



class DrugRec_all(nn.Module):
    def __init__(self, args, sym_information, ddi_adj,  input_smiles_init_rep, emb_dim, device=torch.device('cpu:0')):
        super(DrugRec_all, self).__init__()

        self.device = device
        self.args = vars(args)
        sym_count, sym2idx, sym_comatrix, sym_input_ids = sym_information

        trans_dim = emb_dim
        self.transformer_dp = Transformer_Encoder(n_layer=2, hidden_size=trans_dim, num_attention_heads=4, vocab_size=28996, max_position_size=576, max_segment=40, word_emb_padding_idx=0, dropout=0.1)
        self.transformer_s = Transformer_Encoder(n_layer=2, hidden_size=trans_dim, num_attention_heads=4, vocab_size=28996, max_position_size=256, max_segment=40, word_emb_padding_idx=0, dropout=0.1)

        self.sym_d = nn.Sequential(nn.ReLU(), nn.Linear(trans_dim*2, trans_dim))
        self.sym_p = nn.Sequential(nn.ReLU(), nn.Linear(trans_dim*2, trans_dim))
        self.lin = nn.Sequential(nn.ReLU(), nn.Linear(trans_dim, emb_dim))

        if self.args['multivisit']:
            self.diag_seq_enc = Mul_Attention(hidden_size=emb_dim, device=device)
            self.pro_seq_enc = Mul_Attention(hidden_size=emb_dim, device=device)
            self.sym_seq_enc = Mul_Attention(hidden_size=emb_dim, device=device)
            if self.args['mulhistory']:
                self.diag_agg = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim))
                self.pro_agg = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim))
                self.sym_agg = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim))
                

        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(3 * emb_dim, emb_dim)
        )
 
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        if self.args['fix_smi_rep']:
            self.med_rep = input_smiles_init_rep.to(device=self.device)
        else:
            self.med_rep = torch.tensor(input_smiles_init_rep, requires_grad=True).to(device=self.device)


        # ddi graph adj matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(self.device)

        
        if self.args['CI']:
            sym_prob = [0] * len(sym2idx)
            self.sym_list = [''] * len(sym2idx)
            for sym, idx in sym2idx.items():
                sym_prob[idx] = sym_count[sym]
                self.sym_list[idx] = sym
            self.sym_prob = [i/sum(sym_prob) for i in sym_prob]
            self.sym_input_ids, self.sym2idx = sym_input_ids, sym2idx
            nonco = np.where(sym_comatrix.sum(0) == 0)[0]
            self.sym_co_prob = (sym_comatrix/sym_comatrix.sum(0)).transpose()
            self.sym_co_prob[nonco] = 1/sym_comatrix.shape[0]

            self.query_sym = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim*2, emb_dim))


    def forward(self, input, type):
        DIAG_SEQ, PRO_SEQ, SYM_SEQ = [], [], []

        for j, adm in enumerate(input):
            sym_ids_list = [sym.to(self.device) for sym in adm[2]]
            sym_representations = self.transformer_s(sym_ids_list)
            diag_ids_list = [diag.to(self.device) for diag in adm[0]]
            diag_representations = self.transformer_dp(diag_ids_list)
            pro_ids_list = [pro.to(self.device) for pro in adm[1]]
            pro_representations = self.transformer_dp(pro_ids_list)

            sym_rep = sym_representations[:,0,:].unsqueeze(dim=0)
            diag_rep = diag_representations[:,0,:].unsqueeze(dim=0)
            pro_rep = pro_representations[:,0,:].unsqueeze(dim=0)
            diag_rep = self.sym_d(torch.cat([diag_rep, sym_rep], dim=-1))
            pro_rep = self.sym_p(torch.cat([pro_rep, sym_rep], dim=-1))

            DIAG_SEQ.append(diag_rep)
            PRO_SEQ.append(pro_rep)
            SYM_SEQ.append(sym_rep)
        
        diag_seq = self.lin(torch.cat(DIAG_SEQ, dim=1)) # (1,seq,dim)
        pro_seq = self.lin(torch.cat(PRO_SEQ, dim=1)) # (1,seq,dim)
        sym_seq = self.lin(torch.cat(SYM_SEQ, dim=1)) # (1,seq,dim)

        if self.args['mulhistory']:
            diag_seq, pro_seq, sym_seq = diag_seq.cumsum(1), pro_seq.cumsum(1), sym_seq.cumsum(1)
            diag_seq = self.diag_agg(diag_seq)
            pro_seq = self.pro_agg(pro_seq)
            sym_seq = self.sym_agg(sym_seq)
            self.args['k_mul'] = 1
            

        if self.args['multivisit']:
            diag_out,_ = self.diag_seq_enc(diag_seq, self.args['k_mul'])
            pro_out,_ = self.pro_seq_enc(pro_seq, self.args['k_mul'])
            sym_out,_ = self.sym_seq_enc(sym_seq, self.args['k_mul'])
        else:
            diag_out, pro_out, sym_out = diag_seq, pro_seq, sym_seq
        
        patient_representations = torch.cat([diag_out, pro_out, sym_out], dim=-1).squeeze(dim=0)  # (seq, dim*3) 
        query = self.query(patient_representations) # (seq, dim)

        med_rep = self.mlp(self.med_rep)  # transform hidden size for medications

        if self.args['CI'] and type == 'train':
            # Perform intervention on the symptom in the training stage.
            patient_representations0 = torch.cat([diag_out, pro_out, torch.zeros_like(sym_out)], dim=-1).squeeze(dim=0)  # (seq, dim*3)

            query0 = self.query(patient_representations0) # (seq, dim)

            # Sampling
            if not self.args['multivisit']:
                s_count = np.random.multinomial(self.args['k_ci'], self.sym_prob)
                idx = np.nonzero(s_count)[0]
                s_prob = s_count[idx] / self.args['k_ci']

                sym_representations_ = torch.zeros((1, self.args['dim'])).to(self.device)

                for n, i in enumerate(idx):
                    sym_ids = [self.sym_input_ids[self.sym_list[i]].to(self.device)]
                    sym_representations_ += s_prob[n] * self.transformer_s(sym_ids)[:,0,:]  # (1,dim)
                sym_representations_ = sym_representations_.repeat(len(input), 1)  # (seq, dim)
                
            else:
                sym_representations_ = torch.zeros((len(input), self.args['dim'])).to(self.device)
                for j in range(len(input)):
                    if j == 0:
                        s_count = np.random.multinomial(self.args['k_ci'], self.sym_prob)
                        idx = np.nonzero(s_count)[0]
                        s_prob = s_count[idx] / self.args['k_ci']                    
                        for n, i in enumerate(idx):
                            sym_ids = [self.sym_input_ids[self.sym_list[i]].to(self.device)]
                            sym_representations_[[j]] += s_prob[n] * self.transformer_s(sym_ids)[:,0,:]  # (1,dim)
                    else:
                        s_count_ = np.zeros_like(s_count)
                        for j_ in range(len(s_count[idx])):
                            s_count_ += np.random.multinomial(s_count[idx][j_], self.sym_co_prob[idx[j_]])

                        idx_ = np.nonzero(s_count_)[0]
                        s_prob_ = s_count_[idx_] / self.args['k_ci']
                        for n_, i_ in enumerate(idx_):
                            sym_ids_ = [self.sym_input_ids[self.sym_list[i_]].to(self.device)]
                            sym_representations_[[j]] += s_prob_[n_] * self.transformer_s(sym_ids_)[:,0,:]  # (1,dim)
                        s_count, idx = s_count_, idx_
            
            # do(s_t) vs do(s_0)
            new_query = self.query_sym(torch.cat([query, sym_representations_], dim=-1))
            new_query0 = self.query_sym(torch.cat([query0, sym_representations_], dim=-1))
            result_avg = torch.mm(new_query, med_rep.t())
            result_avg0 = torch.mm(new_query0, med_rep.t())

            pred_prob = F.sigmoid(result_avg)
            pred_prob0 = F.sigmoid(result_avg0)

        result = torch.mm(query, med_rep.t())  # [seq, voc_size]

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = torch.einsum("nc,nk->nck",[neg_pred_prob, neg_pred_prob])  # [seq, voc_size, voc_size]


        batch_neg = 1/self.tensor_ddi_adj.shape[0] * neg_pred_prob.mul(self.tensor_ddi_adj).sum(dim=[1,2]).mean() # (seq,).mean()

        if self.args['CI'] and type == 'train':
            return result, neg_pred_prob, batch_neg, pred_prob, pred_prob0
        elif type == 'train':
            return result, neg_pred_prob, batch_neg
        else:
            return result, batch_neg

