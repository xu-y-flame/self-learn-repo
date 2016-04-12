# coding=utf-8
'''
Created on 2016年3月27日

@author: xuyan
'''

import math
from pango import Weight
import random
from operator import itemgetter

def RMSE(records):
    """
    Root Mean Square Error
        records: iterable data consisting of [u, i, rui, pui]
        rui: real score on product i by user u
        pui: prediction score on product in by user u
    """
    return math.sqrt(sum([(rui-pui)*(rui-pui) for u,i,rui,pui in records]) / float(len(records)))
                     
def MAE(records):
    """
    Mean Absolute Error
    """
    return sum([abs(rui-pui) for u,i,rui,pui in records]) / float(len(records))

def precision_recall(test, N):
    hit = 0
    n_recall = 0
    n_precision = 0
    for user, items in test.items():
        # N:     length of recommending list
        # items: real list of one user
        # rank:  predictive list of one user
        rank = recommend(user, N)
        hit += len(rank & items)
        n_recall += len(items)
        n_precision += N
    return [hit/(1.0*n_recall), hit/(1.0*n_precision)]


def Gini_Index(p):
    """
    Gini Index
        p: popularity list
    """
    j = 1
    n = len(p)
    G = 0
    for item, weight in sorted(p.items()): #, key=itemgetter(1)):
        G += (2 * j - n - 1) * Weight
    return G / float(n-1)


def split_data(data, M, k, seed):
    """
    @param data: iterable data consisting of (user, item)
    @param M: split the data into M slices on uniform distribution
    @param k: 0 <= k <= M-1 
    """
    test  = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0, M) == k:
            test.append([user, item])
        else:
            train.append([user, item])
    return train, test

def get_recommendation(user, N):
    """
    """ 
 
def recall(train, test, N):
    """
    @param N: N recommending products
    """
    hit = 0
    all = 0
    for user in train.keys():
        # Question: the user in train set should be appearing in test set?
        tu = test[user] # Product set for one user in test set
        rank = get_recommendation(user, N)   
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

def precision(train, test, N):
    """
    @param N: N recommending products
    """
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = get_recommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)
    
def coverage(train, test, N): # Question: what about test?
    """
    """
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = get_recommendation(user, N)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(len(all_items)) * 1.0)

def popularity(train, test, N):
    """
    """
    item_popu = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popu:
                item_popu[item] = 0
            item_popu[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = get_recommendation(user, N)
        for item, pui in rank:
            ret += math.log(1 + item_popu[item])
            n += 1
    ret /= (n * 1.0)
    
    return ret

def user_similarity(train):
    """
    """
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u == v:
                continue
            W[u][v] = len(train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

def user_similarity_invTable(train):
    """
    @return W: return a matrix with elements representing the similarity between two users
    """
    # Build inverse table for item_users
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
            
    # Calculate co-rated items between users
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                # If user u and v have rated the same product, 
                # add 1 to the corresponding element in the sparse matrix.
                C[u][v] += 1 
    
    # Calculate final similarity matrix W
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def recommend(user, train, W, K):
    """
    """
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items, key=itemgetter(1), reverse=True)[0:K]:
        for i, rvi in train[v].items():
            if i in interacted_items[v].items():
                continue
            rank[i] = wuv * rvi
    return rank

def user_similarity_invTable_IIF(train):
    """
    @return W: return a matrix with elements representing the similarity between two users
    
    Add penalty on number of users for one product
    
    """
    # Build inverse table for item_users
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
            
    # Calculate co-rated items between users
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                # If user u and v have rated the same product, 
                # add 1 to the corresponding element in the sparse matrix.
                C[u][v] += 1 / math.log(1 + len(users))
    
    # Calculate final similarity matrix W
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def item_similarity(train):
    """
    """
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items:
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C[i][j] += 1
                
    # Calculate final similarity matrix W
    W = dict()
    for i, related_items in C.items():
        for j, uij in related_items.items():
            W[i][j] = uij / math.sqrt(N[i] * N[j])
    return W 
    
def random_select_neg_samples(self, items, items_pool):
    """
    @param items: Dict. It contains products bought by users.
    """
    ret = dict()
    for i in items.keys():    
        ret[i] = 1
    n = 0
    
    for i in range(0, len(items) * 3):
        item = items_pool[random.randint(0, len(items_pool)-1)] # items_pool contains candidate products.
        if item in ret:
            continue
        ret[item] = 0
        n += 1
        if n > len(items):
            break
    return ret

def init_model(user_items, F):
    """
    """

def predict_score(user, item):
    """
    """
    
def Latent_Factor_Model(user_items, F, N, alpha, Lambda):
    """
    """
    [P, Q] = init_model(user_items, F) # P, Q means what ?
    for step in range(0, N):
        for user, items in user_items.items():
            samples = random_select_neg_samples(items)
            for item, rui in samples.items():
                eui = rui - predict_score(user, item)
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[item][f] - Lambda * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - Lambda * Q[item][f])
        alpha *= 0.9

def personal_rank(G, alpha, root, max_steps):
    """
    """
    rank = dict()
    rank = {x:0 for x in G.keys()}        
    rank[root] = 1
    for k in range(max_steps):
        tmp = {x:0 for x in G.keys()}
        for i, ri in G.items():
            for j, wij in ri.items():
                if j not in tmp:
                    tmp[j] = 0
                tmp[j] += alpha * rank[i] / (1.0 * len(ri))
                if j == root:
                    tmp[j] += 1 - alpha
        rank = tmp
    return rank
    
    
    
    
if __name__ == "__main__":
    """
    """
    rank = {x:1 for x in range(10)}
    print rank
#     """
#     d1 = {9:1, 1:2, 7:4, 5:6}
#     print d1.keys()
#     print d1.values()
#     print d1.items(), type(d1.items())
#     
#     print sorted(d1.items())
#     
#     m1 = {9:[1, 2, 'str'], 1:[2, 3, 3], 7:[2, 3], 5:[2, 3]}
#     
#     print m1[9][2]
#     
#     m1[9][2] = 100
#     print m1
#     
#     m2 = {9:{1:2, 3:4, 10:4}, 1:{2:1, 3:22, 22:1}, 7:{1:22, 3:23}, 5:{2:111, 3:233}}
#     print m2
#     m2[9][3] = 111
#     print m2
#     
#     
#     print m2.items()
#     for k1, va1 in m2.items():
#         for k2, va2 in va1.items():
#             print k1, k2, va2
# #         for k2, va2 in va1.items():
# #             print va2
#     
#     
# #     l1 = [(11, 2), (9, 4), (5, 6)]
# #     print sorted(l1)
# #    
# #     u = 0b110010110 # [1, 2, 3, 3, 4]
# #     v = 0b111001001 # [1, 2, 1, 1, 2]
# #     print "u&v", u & v
# #     
# #     u1 = 011
# #     u2 = 011
# #     print u1 & u2
# #     
#     
#     
#     W = dict()   
#     for i in range(5):
#         W[i] = {}
# #         for j in range(7, 10):
# #             W[i][j] = i * j 
#         for j in range(6, 9):
#             W[i][j] = j*2
#     print W
    
    
    
    
    
    
    
    
        

