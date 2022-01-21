import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from numpy.linalg import norm

#node_feat_csv = pd.read_csv('train_csvs/edges_train_A_dw_emb.csv', sep = ' ', header=None)
node_feat_csv = pd.read_csv('train_csvs/vec_1st_wo_norm_A.csv', sep = ' ', header=None)
#node_feat_csv = pd.read_csv('train_csvs/edges_train_A_hin2vec_emv_2.csv', sep = ' ', header=None)
node_feat = node_feat_csv.values[:,1:]
node_idx = node_feat_csv.values[:,0].astype('int64')

node_num = max(node_idx) + 1

node_emb_1 = np.zeros((node_num, 128))
node_emb_1[node_idx] = node_feat

node_feat_csv = pd.read_csv('train_csvs/vec_1st_wo_norm_A.csv', sep = ' ', header=None)
node_feat = node_feat_csv.values[:,1:]
node_idx = node_feat_csv.values[:,0].astype('int64')

node_num = max(node_idx) + 1

node_emb_2 = np.zeros((node_num, 128))
node_emb_2[node_idx] = node_feat

node_emb = node_emb_1

print(node_emb.shape)

test_csv = pd.read_csv(f'test_csvs/input_A_final.csv', names=['src', 'dst', 'type', 'start_at', 'end_at', 'exist'])
label = test_csv.exist.values
src = test_csv.src.values
dst = test_csv.dst.values

print(label.shape)

exist_probs = []
for i in range(label.shape[0]):
    exist_prob = np.dot(node_emb[src[i], :], node_emb[dst[i], :])/(norm(node_emb[src[i], :])*norm(node_emb[dst[i], :]))
    exist_prob_2 = np.dot(node_emb_2[src[i], :], node_emb_2[dst[i], :])/(norm(node_emb_2[src[i], :])*norm(node_emb_2[dst[i], :]))
    #print(exist_prob)
    exist_probs.append(exist_prob_2)

exist_probs = np.array(exist_probs)

AUC = roc_auc_score(label,exist_probs)
print(f'AUC is {round(AUC,5)}')


exist_probs = []
for i in range(label.shape[0]):
    exist_prob = np.dot(node_emb[src[i], :], node_emb[dst[i], :])
    #print(exist_prob)
    exist_probs.append(exist_prob)

exist_probs = np.array(exist_probs)

AUC = roc_auc_score(label,exist_probs)
print(f'AUC is {round(AUC,5)}')
