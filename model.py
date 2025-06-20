import numpy as np
import torch
import torch.nn.functional as F

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, llm_emb):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.llm_emb = llm_emb
        self.llm_proj = torch.nn.Linear(llm_emb.shape[1], args.hidden_units)

        self.llm_weight = args.llm_weight

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs, cf_llm): # TODO: fp64 and int64 as default in python, trim?
        if cf_llm == "cf":
            seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
            seqs *= self.item_emb.embedding_dim ** 0.5
        elif cf_llm == "llm":
            llm_raw_emb = self.llm_emb[torch.LongTensor(log_seqs).to(self.dev)]  # (B, T, D_llm)
            seqs = self.llm_proj(llm_raw_emb)  # (B, T, D_model)
            seqs *= self.llm_proj.out_features ** 0.5
        else:
            print("cf_llm error!")

        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, view): # for training        
        log_feats = self.log2feats(log_seqs, "cf") # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        # cl
        unique_items = np.unique(log_seqs)
        cf_view = self.item_emb(torch.LongTensor(unique_items).to(self.dev))
        llm_view = self.llm_proj(self.llm_emb[torch.LongTensor(unique_items).to(self.dev)])  # (B, T, D_llm)
        
        if view == 0:
            return pos_logits, neg_logits, cf_view, llm_view
        else:
            log_feats_llm = self.log2feats(log_seqs, "llm")

            pos_llm_raw = self.llm_emb[torch.LongTensor(pos_seqs).to(self.dev)]
            neg_llm_raw = self.llm_emb[torch.LongTensor(neg_seqs).to(self.dev)]
            pos_llm_embs = self.llm_proj(pos_llm_raw)
            neg_llm_embs = self.llm_proj(neg_llm_raw)

            pos_logits_llm = (log_feats_llm * pos_llm_embs).sum(dim=-1)
            neg_logits_llm = (log_feats_llm * neg_llm_embs).sum(dim=-1)

            final_pos_logits = pos_logits + self.llm_weight*pos_logits_llm
            final_neg_logits = neg_logits + self.llm_weight*neg_logits_llm

            return final_pos_logits, final_neg_logits, cf_view, llm_view  # pos_pred, neg_pred

    def info_nce_loss(self, emb1, emb2, temperature=0.07):
        """
        emb1, emb2: [B, N, D] where N is number of items (e.g., 10)
        Computes InfoNCE loss for each corresponding item pair in a batch.
        """
        
        # Normalize
        emb1 = F.normalize(emb1, dim=-1)  # [B, N, D]
        emb2 = F.normalize(emb2, dim=-1)  # [B, N, D]

        # Compute similarity matrix: (B*N, B*N)
        sim_matrix = torch.matmul(emb1, emb2.T)  # cosine similarity

        # 5. InfoNCE loss 
        sim_matrix /= temperature
        labels = torch.arange(sim_matrix.size(0)).to(self.dev)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


    def predict(self, user_ids, log_seqs, item_indices, view): # for inference
        log_feats = self.log2feats(log_seqs, "cf") # user_ids hasn't been used yet
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        if view == 0:
            return logits
        elif view == 1:
            log_feats_llm = self.log2feats(log_seqs, "llm")
            llm_raw = self.llm_emb[torch.LongTensor(item_indices).to(self.dev)]  # (B, C)
            llm_proj = self.llm_proj(llm_raw)  # (B, D)
            final_feat_llm = log_feats_llm[:,-1,:]
            logits_llm = llm_proj.matmul(final_feat_llm.unsqueeze(-1)).squeeze(-1)

            # preds = self.pos_sigmoid(logits) # rank same item list for different users
            final_logits = logits + self.llm_weight*logits_llm
            
            return final_logits # preds # (U, I)
