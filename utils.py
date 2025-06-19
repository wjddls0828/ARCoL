import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
from tqdm import tqdm

def head_tail_split(user_train, head_proportion, dataset):
    all_items=[]
    f = open('./data/'+dataset+'.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        i = int(i)
        all_items.append(i)
    f.close()
    item_counts = Counter(all_items)
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

    train_items = np.array([item[0] for item in sorted_items])
    frequency = np.array([item[1] for item in sorted_items])

    top_20_percent_index = int(len(sorted_items) * head_proportion)
    head_items = np.array([item[0] for item in sorted_items[:top_20_percent_index]])
    tail_items=np.array([item[0] for item in sorted_items[top_20_percent_index:]])
   
    return head_items, tail_items, train_items, frequency, sorted_items

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sim_neq(l, r, s, probs): 
    t = np.random.choice(np.arange(0,r), p=probs/probs.sum())
    while t in s:
        t = np.random.choice(np.arange(0,r), p=probs/probs.sum())
    return t

def sim_neq_pop(l, r, s, probs, train_items, frequency): 
    new_score = probs[1:] + frequency
    t = np.random.choice(np.arange(1,r), p=new_score/new_score.sum())
    while t in s:
        t = np.random.choice(np.arange(1,r), p=new_score/new_score.sum())
    return t


def sim_neq_pop_mix_simple(r, s, probs, frequency, alpha=0.5, beta=0.3, gamma=0.2):
    """
    r: 전체 아이템 수 + 1 (ID 0 제외)
    s: 제외할 아이템 집합
    probs: similarity score (length r)
    frequency: 등장 횟수 (length r)
    alpha/beta/gamma: 각 요소의 가중치 (합 1 추천)
    """
    base_len = r - 1  # 아이템 1 ~ r-1
    
    probs = probs[1:]
    freq = frequency
    
    # 단순 정규화
    probs = probs / (probs.sum() + 1e-8)
    freq = freq / (freq.sum() + 1e-8)
    
    # uniform 분포는 동일한 값
    uniform = np.ones(base_len) / base_len

    # 혼합
    mix_score = alpha * uniform + beta * probs + gamma * freq
    mix_score = mix_score / mix_score.sum()  # 안전한 정규화

    while True:
        t = np.random.choice(np.arange(1, r), p=mix_score)
        if t not in s:
            return t

def sim_neq_pop_mix_fast(r, s, mix_score):
    while True:
        t = np.random.choice(np.arange(1, r), p=mix_score / mix_score.sum())
        if t not in s:
            return t
   
"""def sim_neq(l, r, s, probs):
    print("start here")
    print("probs.shape", probs.shape)
    probs = probs.clone()
    print("start here 1")
    probs[s] = 0
    print("start here 2")
    probs /= probs.sum()
    print("end here")
    sampled = torch.multinomial(probs, num_samples=1).item()
    print("sampled")
    return sampled"""

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, mix_score,  train_items, frequency, args):
    def sample(uid):
        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: 
            uid = np.random.randint(1, usernum + 1)
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        if args.neg_sampling == "random":
            for i in reversed(user_train[uid][:-1]):
                seq[idx] = i
                pos[idx] = nxt
                # if nxt != 0: neg[idx] = sim_neq_pop_mix_simple(itemnum + 1, ts, probs[nxt], frequency, args.alpha, args.beta, args.gamma)
                if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
                nxt = i
                idx -= 1
                if idx == -1: break
        elif args.neg_sampling == "mix":
            for i in reversed(user_train[uid][:-1]):
                seq[idx] = i
                pos[idx] = nxt
                # if nxt != 0: neg[idx] = sim_neq_pop_mix_simple(itemnum + 1, ts, probs[nxt], frequency, args.alpha, args.beta, args.gamma)
                if nxt != 0: neg[idx] = sim_neq_pop_mix_fast(itemnum + 1, ts, mix_score[nxt-1])
                nxt = i
                idx -= 1
                if idx == -1: break

        return (uid, seq, pos, neg)
    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, emb_matrix, train_items, frequency, args, batch_size=64, maxlen=10, n_workers=1, alpha=5.):
        """# Normalize embeddings
        emb_norms = torch.norm(emb_matrix, p=2, dim=1, keepdim=True)
        emb_matrix = emb_matrix / emb_norms  # [n_items+1, emb_dim]

        # Compute cosine similarity matrix
        sims = emb_matrix @ emb_matrix.T  # [n_items+1, n_items+1]

        # Convert to cosine distance
        dists = 1 - sims  # ∈ [0, 2]

        # Apply softmax over scaled distance
        exp_scores = torch.exp(alpha * dists)

        # Zero out item 0 related scores and self-scores
        exp_scores[0, :] = 0
        exp_scores[:, 0] = 0
        exp_scores.fill_diagonal_(0)  # In-place diagonal zero

        # Normalize to get probability distribution
        self.probs = exp_scores / exp_scores.sum()
        self.probs = self.probs.cpu()
        print("sim finish")"""

        emb_matrix = emb_matrix / emb_matrix.norm(p=2, dim=1, keepdim=True)
        sims = emb_matrix @ emb_matrix.T  
        sims= sims.cpu()
        dists = 1 - sims
        exp_scores = torch.exp(alpha * dists)
        exp_scores[0, :] = 0
        exp_scores[:, 0] = 0
        exp_scores.fill_diagonal_(0)
        probs = (exp_scores / exp_scores.sum()).cpu().numpy()
        print("done probs")

        base_len = itemnum
        probs = probs[1:, 1:]
        
        # 단순 정규화
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
        freq = frequency / (frequency.sum() + 1e-8)
        
        # uniform 분포는 동일한 값
        uniform = np.ones(base_len) / base_len

        # 혼합
        self.mix_score = args.alpha * uniform + args.beta * probs + args.gamma * freq
        print("done mix score")

        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      self.mix_score,
                                                      train_items,
                                                      frequency,
                                                      args
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def my_evaluate_test(model, dataset, args, tailset, k, k2, k3):
    print("test eval")

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    metrics = {
        k: {'NDCG': 0.0, 
            'HT': 0.0, 
            'tail_NDCG': 0.0, 
            'tail_HT': 0.0, 
            'head_NDCG': 0.0, 
            'head_HT': 0.0, 
            'total_items': [], 
            'freq_items': np.zeros(itemnum + 1),
            'freq_items_tail': np.zeros(itemnum + 1),
            'freq_items_head': np.zeros(itemnum + 1),}
        for k in [k, k2, k3]
    }

    valid_user = 0.0
    tail_valid_user = 0.0
    head_valid_user = 0.0

    users = range(1, usernum+1)

    for u in tqdm(users):
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        item_idx = list(set(range(1,itemnum+1)) - set(train[u])- set([valid[u][0]]) | set([test[u][0]]))

        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], args.twoview)
        predictions = predictions[0] # - for 1st argsort DESC

        _, topk3 = torch.topk(predictions, k3)
        topk3_cpu = np.array(item_idx)[topk3.cpu()]

        topk_all = {}
        topk_all[k3] = topk3_cpu
        topk_all[k2] = topk3_cpu[:k2]
        topk_all[k] = topk3_cpu[:k]
        
        valid_user += 1
        is_tail = test[u][0] in tailset
        if is_tail:
            tail_valid_user += 1
        else:
            head_valid_user += 1

        for topk in [k, k2, k3]:
            topk_items = topk_all[topk]
            if test[u][0] in topk_items:
                rank = np.where(topk_items == test[u][0])[0]
                gain = 1 / np.log2(rank + 2)
                metrics[topk]['NDCG'] += gain
                metrics[topk]['HT'] += 1
                if is_tail:
                    metrics[topk]['tail_NDCG'] += gain
                    metrics[topk]['tail_HT'] += 1
                else:
                    metrics[topk]['head_NDCG'] += gain
                    metrics[topk]['head_HT'] += 1
            metrics[topk]['total_items'].extend(topk_items)
            for i in topk_items:
                metrics[topk]['freq_items'][i] += 1
                if i in tailset:
                    metrics[topk]['freq_items_tail'][i] += 1
                else:
                    metrics[topk]['freq_items_head'][i] += 1

    
    headset = set(range(1,itemnum+1)) - tailset
    results = {}
    for topk in [k, k2, k3]:
        tag = f"@{topk}"
        total_items = metrics[topk]['total_items']
        freq = metrics[topk]['freq_items']
        p_k = (freq / freq.sum())[1:]
        entropy = -np.sum(p_k * np.log2(p_k + 1e-9))

        tail_indices = list(tailset)
        freq_tail = freq[tail_indices]
        p_tail = freq_tail / freq_tail.sum()
        entropy_tail = -np.sum(p_tail * np.log2(p_tail + 1e-9))

        head_indices = list(headset)
        freq_head = freq[head_indices]
        p_head = freq_head / freq_head.sum()
        entropy_head = -np.sum(p_head * np.log2(p_head + 1e-9))


        results[f"NDCG{tag}"] = metrics[topk]['NDCG'] / valid_user
        results[f"HT{tag}"] = metrics[topk]['HT'] / valid_user
        results[f"TailNDCG{tag}"] = metrics[topk]['tail_NDCG'] / tail_valid_user
        results[f"TailHT{tag}"] = metrics[topk]['tail_HT'] / tail_valid_user
        results[f"HeadNDCG{tag}"] = metrics[topk]['head_NDCG'] / head_valid_user
        results[f"HeadHT{tag}"] = metrics[topk]['head_HT'] / head_valid_user
        results[f"Coverage{tag}"] = len(Counter(total_items).keys()) / itemnum
        results[f"AggreTCov{tag}"] = len(set(Counter(total_items).keys()) & tailset) / len(tailset)
        results[f"AggreHCov{tag}"] = len(set(Counter(total_items).keys()) & headset) / len(headset)
        results[f"Entropy{tag}"] = entropy
        results[f"TailEntropy{tag}"] = entropy_tail
        results[f"HeadEntropy{tag}"] = entropy_head
        
    return results

# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        # item_idx = [valid[u][0]]
        item_idx = list(set(range(1,itemnum+1)) - set(train[u]) | set([valid[u][0]]))
        """for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)"""

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def my_evaluate_valid(model, dataset, args, tailset, k, k2, k3):
    print("valid eval")
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    metrics = {
        k: {'NDCG': 0.0, 
            'HT': 0.0, 
            'tail_NDCG': 0.0, 
            'tail_HT': 0.0, 
            'head_NDCG': 0.0, 
            'head_HT': 0.0, 
            'total_items': [],
            'freq_items': np.zeros(itemnum + 1),
            'freq_items_tail': np.zeros(itemnum + 1),
            'freq_items_head': np.zeros(itemnum + 1),}
        for k in [k, k2, k3]
    }

    valid_user = 0.0
    tail_valid_user = 0.0
    head_valid_user = 0.0

    users = range(1, usernum + 1)
    for u in tqdm(users):
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        item_idx = list(set(range(1,itemnum+1)) - set(train[u]) | set([valid[u][0]]))

        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]], args.twoview)
        predictions = predictions[0]

        _, topk3 = torch.topk(predictions, k3)
        topk3_cpu = np.array(item_idx)[topk3.cpu()]

        topk_all = {}
        topk_all[k3] = topk3_cpu
        topk_all[k2] = topk3_cpu[:k2]
        topk_all[k] = topk3_cpu[:k]
        
        valid_user += 1
        is_tail = valid[u][0] in tailset
        if is_tail:
            tail_valid_user += 1
        else:
            head_valid_user += 1

        for topk in [k, k2, k3]:
            topk_items = topk_all[topk]
            if valid[u][0] in topk_items:
                rank = np.where(topk_items == valid[u][0])[0]
                gain = 1 / np.log2(rank + 2)
                metrics[topk]['NDCG'] += gain
                metrics[topk]['HT'] += 1
                if is_tail:
                    metrics[topk]['tail_NDCG'] += gain
                    metrics[topk]['tail_HT'] += 1
                else:
                    metrics[topk]['head_NDCG'] += gain
                    metrics[topk]['head_HT'] += 1
            metrics[topk]['total_items'].extend(topk_items)
            for i in topk_items:
                metrics[topk]['freq_items'][i] += 1
                if i in tailset:
                    metrics[topk]['freq_items_tail'][i] += 1
                else:
                    metrics[topk]['freq_items_head'][i] += 1
    
    headset = set(range(1,itemnum+1)) - tailset
    results = {}
    for topk in [k, k2, k3]:
        tag = f"@{topk}"
        total_items = metrics[topk]['total_items']
        freq = metrics[topk]['freq_items']
        p_k = (freq / freq.sum())[1:]
        entropy = -np.sum(p_k * np.log2(p_k + 1e-9))

        tail_indices = list(tailset)
        freq_tail = freq[tail_indices]
        p_tail = freq_tail / freq_tail.sum()
        entropy_tail = -np.sum(p_tail * np.log2(p_tail + 1e-9))

        head_indices = list(headset)
        freq_head = freq[head_indices]
        p_head = freq_head / freq_head.sum()
        entropy_head = -np.sum(p_head * np.log2(p_head + 1e-9))

        results[f"NDCG{tag}"] = metrics[topk]['NDCG'] / valid_user
        results[f"HT{tag}"] = metrics[topk]['HT'] / valid_user
        results[f"TailNDCG{tag}"] = metrics[topk]['tail_NDCG'] / tail_valid_user
        results[f"TailHT{tag}"] = metrics[topk]['tail_HT'] / tail_valid_user
        results[f"HeadNDCG{tag}"] = metrics[topk]['head_NDCG'] / head_valid_user
        results[f"HeadHT{tag}"] = metrics[topk]['head_HT'] / head_valid_user
        results[f"Coverage{tag}"] = len(Counter(total_items).keys()) / itemnum
        results[f"AggreTCov{tag}"] = len(set(Counter(total_items).keys()) & tailset) / len(tailset)
        results[f"AggreHCov{tag}"] = len(set(Counter(total_items).keys()) & headset) / len(headset)
        results[f"Entropy{tag}"] = entropy
        results[f"TailEntropy{tag}"] = entropy_tail
        results[f"HeadEntropy{tag}"] = entropy_head

    return results
