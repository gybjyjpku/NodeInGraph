import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from numpy.linalg import norm

dataset = 'B' # 'B'

node_feat_csv = pd.read_csv(f'files/vec_1st_wo_norm_{dataset}.csv', sep = ' ', header=None)
node_feat = node_feat_csv.values[:,1:129]
node_idx = node_feat_csv.values[:,0].astype('int64')

node_num = max(node_idx) + 1

node_emb_1 = np.zeros((node_num, 128))
node_emb_1[node_idx] = node_feat

node_emb_2 = np.zeros((node_num, 128))
node_emb_2[node_idx] = node_feat

node_emb = node_emb_1

print(node_emb.shape)

test_csv = pd.read_csv(f'files/input_{dataset}_initial.csv', names=['src', 'dst', 'type', 'start_at', 'end_at', 'exist'])
label = test_csv.exist.values
src = test_csv.src.values
dst = test_csv.dst.values

print(label.shape)


# dot product similarity
exist_probs = []
for i in range(label.shape[0]):
    exist_prob = np.dot(node_emb[src[i], :], node_emb[dst[i], :])
    exist_probs.append(exist_prob)

exist_probs = np.array(exist_probs)

AUC = roc_auc_score(label,exist_probs)
print(f'AUC is {round(AUC,5)}')

