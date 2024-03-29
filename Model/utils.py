
import json
import math
import os
import random

import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from tqdm import tqdm
import scipy

import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items, model_name):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    if model_name in ['TFIDF', 'COSINE', 'BM25']:
        rating_matrix = csr_matrix((data.astype(np.float32), (row, col)), shape=(num_users, num_items))
    else:
        rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(data_file, preds, model_name):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()
    item_ids = rating_df['item'].unique()
    
    if model_name in ['BERT4Rec']:  
        idx2item = pd.Series(data=item_ids, index=np.arange(len(item_ids))+1)  # item idx -> item id
    else:
        idx2item = pd.Series(data=item_ids, index=np.arange(len(item_ids)))
    

    result = []

    for index, items in enumerate(tqdm(preds)):
        for item in items:
            result.append((users[index], idx2item[item]))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        f"output/{model_name}_submission.csv", index=False
    )

def generate_implicit_df(pos_dataset, dataset):
    # df = pd.read_csv(data_file)
    # df = item_encoding(df, model_name)
    
    pos_dataset = pos_dataset
    dataset = dataset

    user_ids = dataset.keys()
    # item_ids = df['item_idx'].unique()

    implicit_df = dict()
    implicit_df['user_idx'] = list()
    implicit_df['item_idx'] = list()
    implicit_df['label'] = list()
    # user_dict = dict()
    # item_dict = dict()


    for user_id in tqdm(user_ids):
        # user_dict[u] = user_id
        item_ids = dataset[user_id]
        if pos_dataset:
            pos_item_ids = pos_dataset[user_id]
        for item_id in item_ids:
            # if i not in item_dict:
                # item_dict[i] = item_id
            implicit_df['user_idx'].append(user_id)
            implicit_df['item_idx'].append(item_id)
            if pos_dataset:
                if item_id in pos_item_ids:
                    implicit_df['label'].append(1)
                else:
                    implicit_df['label'].append(0)
            else:
                implicit_df['label'].append(0)

    implicit_df = pd.DataFrame(implicit_df)

    return implicit_df

def item_encoding(df, model_name):
    rating_df = df.copy()

    item_ids = rating_df['item'].unique()
    user_ids = rating_df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)

    # user, item indexing
    if model_name in ['BERT4Rec']:
        item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
    else:
        item2idx = pd.Series(data=np.arange(len(item_ids)), index=item_ids) # item re-indexing (0~num_item-1)
    
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

    # dataframe indexing
    rating_df = pd.merge(rating_df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    rating_df = pd.merge(rating_df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    rating_df.sort_values(['user_idx', 'time'], inplace=True)
    del rating_df['item'], rating_df['user']

    return rating_df

def train_valid_split(args):
    df = pd.read_csv(args.data_file)
    df = item_encoding(df, args.model_name)

    items = df.groupby("user_idx")["item_idx"].apply(list)
    # {"user_id" : [items]}
    train_set, valid_set, item_set = {} , {}, {}
    print("train_valid set split by user_idx")

    for uid, item in enumerate(tqdm(items)):

        # 유저가 소비한 item의 12.5% 또는 최대 10 으로 valid_set 아이템 구성
        # num_u_valid_set = 10
        num_u_valid_set = min(int(len(item)*0.125), 10)
        u_valid_set = np.random.choice(item, size=num_u_valid_set, replace=False)
        
        train_set[uid] = list(set(item) - set(u_valid_set))
        valid_set[uid] = u_valid_set.tolist()
        item_set[uid] = list(set(item))

    return train_set, valid_set, item_set

def negative_sampling(args, *datasets):
    df = pd.read_csv(args.data_file)
    df = item_encoding(df, args.model_name)

    users_list = df['user_idx'].unique()
    items_list = df['item_idx'].unique()

    train_set, valid_set, item_set = datasets
    neg_sample_set = {}
    # submission_set = {}

    for uid in tqdm(users_list):
        if args.neg_sampling_method == 'n_neg':
            neg_set_size = args.n_negs * len(train_set[uid])
        else:
            neg_set_size = args.neg_sample_num

        neg_sample_set[uid] = list(set(items_list) - set(item_set[uid]))
        u_neg_set = np.random.choice(neg_sample_set[uid], size=neg_set_size, replace=False)  
        train_set[uid] = list(set(train_set[uid]).union(set(u_neg_set)))
        item_set[uid] = list(set(train_set[uid]).union(set(valid_set[uid])))

        # # valid_set negative sampling
        # neg_set_size = args.valid_per_user - len(valid_set[uid])
        # valid_u_neg_set = np.random.choice(neg_sample_set[uid], size=neg_set_size, replace=False)
        # valid_set[uid] = list(set(valid_set[uid]).union(set(valid_u_neg_set)))

        # # valid_set negative sampling
        # neg_set_size = args.sub_per_user
        # sub_u_neg_set = np.random.choice(neg_sample_set[uid], size=neg_set_size, replace=False)
        # submission_set[uid] = list(set(sub_u_neg_set))

    return train_set, item_set

def negative_sampling_ncf(args, *datasets):
    df = pd.read_csv(args.data_file)
    df = item_encoding(df, args.model_name)

    users_list = df['user_idx'].unique()
    items_list = df['item_idx'].unique()

    train_set, valid_set, item_set = datasets
    neg_sample_set = {}
    submission_set = {}

    for uid in tqdm(users_list):
        if args.neg_sampling_method == 'n_neg':
            neg_set_size = args.n_negs * len(train_set[uid])
        else:
            neg_set_size = args.neg_sample_num

        neg_sample_set[uid] = list(set(items_list) - set(item_set[uid]))
        u_neg_set = np.random.choice(neg_sample_set[uid], size=neg_set_size, replace=False)  
        train_set[uid] = list(set(train_set[uid]).union(set(u_neg_set)))
        item_set[uid] = list(set(train_set[uid]).union(set(valid_set[uid])))

        # valid_set negative sampling
        neg_set_size = args.valid_per_user - len(valid_set[uid])
        valid_u_neg_set = np.random.choice(neg_sample_set[uid], size=neg_set_size, replace=False)
        valid_set[uid] = list(set(valid_set[uid]).union(set(valid_u_neg_set)))

        # submission_set negative sampling
        neg_set_size = args.sub_per_user
        sub_u_neg_set = np.random.choice(neg_sample_set[uid], size=neg_set_size, replace=False)
        submission_set[uid] = list(set(sub_u_neg_set))

    return train_set, item_set, valid_set, submission_set

def make_inter_mat(data_file, model_name, *datasets):
    df = pd.read_csv(data_file)
    df = item_encoding(df, model_name)

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    mat_list = []
    dataset_list = datasets

    for dataset in dataset_list: 
        inter_mat = np.zeros((num_users, num_items))
        for uid, items in tqdm(dataset.items()):
            for item in items:
                inter_mat[uid][item] = 1
        mat_list.append(inter_mat)

    return mat_list

def get_user_seqs(data_file, model_name):
    df = pd.read_csv(data_file)
    rating_df = item_encoding(df, model_name)
    
    lines = rating_df.groupby("user_idx")["item_idx"].apply(list)
    user_seq = []
    item_set = set()
    for line in lines:

        items = line
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    if model_name in ['SASRec']:
        num_items = max_item + 2
    else:
        num_items = max_item + 1

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    submission_rating_matrix = generate_rating_matrix_submission(
        user_seq, num_users, num_items, model_name
    )
    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(data_file, model_name):
    df = pd.read_csv(data_file)
    rating_df = item_encoding(df, model_name)

    lines = rating_df.groupby("user_idx")["item_idx"].apply(list)
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        items = line
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def mf_sgd(
    P: np.ndarray,
    Q: np.ndarray,
    b: float,
    b_u: np.ndarray,
    b_i: np.ndarray,
    samples: List[Tuple],
    learning_rate: float,
    regularization: float
) -> None:

    for user_id, item_id, rating in tqdm(samples):
    
        predicted_rating = P[user_id] @ Q[item_id].T 
        
        error = rating - b - b_u[user_id] - b_i[item_id] - predicted_rating 
        
        b_u[user_id] += learning_rate * (error - (regularization * b_u[user_id])) 
        b_i[item_id] += learning_rate * (error - (regularization * b_i[item_id])) 
        
        P[user_id, :] += learning_rate * ((error * Q[item_id, :]) - (regularization * P[user_id, :])) 
        Q[item_id, :] += learning_rate * ((error * P[user_id, :]) - (regularization * Q[item_id, :])) 

def get_predicted_full_matrix(
    P: np.ndarray,
    Q: np.ndarray,
    b: float = None,
    b_u: np.ndarray = None,
    b_i: np.ndarray = None
) -> np.ndarray:

    if b is None:
        return P @ Q.T 
    else:
        return (P @ Q.T) + np.expand_dims(b_i, axis=0) + np.expand_dims(b_u, axis=1) + b 

def als(
    F: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    C: np.ndarray,
    K: int,
    regularization: float
) -> None:

    for user_id, F_user in enumerate(tqdm(F)):
        C_u = np.diag(C[user_id])
        P[user_id] = np.linalg.solve(((Q.T @ C_u) @ Q) + (regularization * np.identity(K)), (Q.T @ C_u) @ F_user) # FILL HERE : USE np.linalg.solve()#
        
    for item_id, F_item in enumerate(tqdm(F.T)):
        C_i = np.diag(C[:, item_id])
        Q[item_id] = np.linalg.solve(((P.T @ C_i) @ P) + (regularization * np.identity(K)), (P.T @ C_i) @ F_item) # FILL HERE : USE np.linalg.solve()#

def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def get_rmse(
    R: np.ndarray,
    predicted_R: np.ndarray
) -> float:
    """
    전체 학습 데이터(실제 평가를 내린 데이터)에 대한 RMSE를 계산
    :param R: (np.ndarray) 유저-아이템 rating 매트릭스. shape: (유저 수, 아이템 수)
    :param predicted_R: (np.ndarray) 예측된 유저-아이템 rating 매트릭스. shape: (유저 수, 아이템 수)
    :return: (float) 전체 학습 데이터에 대한 RMSE
    """
    
    user_index, item_index, _ = scipy.sparse.find(R) 
    error = list()
    for user_id, item_id in tqdm(zip(user_index, item_index), total=len(user_index)):
        square_error = (predicted_R[user_id, item_id] - R[user_id, item_id]) ** 2 
        error.append(square_error)
    rmse = (sum(error) / float(len(error))) ** 0.5 
    return rmse

def get_ALS_loss(
    F: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    C: np.ndarray,
    regularization: float
) -> float:

    
    user_index, item_index = F.nonzero()
    loss = 0
    for user_id, item_id in tqdm(zip(user_index, item_index), total=len(user_index)):
        predict_error = (F[user_id, item_id] - (P[user_id].T @ Q[item_id])) ** 2 # FILL HERE #
        confidence_error = C[user_id, item_id] * predict_error # FILL HERE #
        loss += confidence_error
    for user_id in tqdm(range(F.shape[0])):
        regularization_term = regularization * np.square(np.linalg.norm(P[user_id])) # FILL HERE #
        loss += regularization_term
    for item_id in tqdm(range(F.shape[1])):
        regularization_term = regularization * np.square(np.linalg.norm(Q[item_id])) # FILL HERE #
        loss += regularization_term

    return loss
