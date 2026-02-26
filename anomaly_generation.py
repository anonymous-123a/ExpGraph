import datetime
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix
from sklearn.cluster import SpectralClustering

def anomaly_generation3(ini_graph_percent, anomaly_percent, data, m, max_idx1, max_idx2, datasetname=None):

    train_num = int(np.floor(ini_graph_percent * m))


    train = data[:train_num, :]

    adj = np.zeros((max_idx1 + 1, max_idx2+1))
    for edge in train:
        adj[edge[0]][edge[1]] = adj[edge[0]][edge[1]] + 1


    test = data[train_num:, :]




    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    idx_test = np.ones([np.size(test, 0) + anomaly_num, 1], dtype=np.int64)
    idx_train = np.ones([np.size(train, 0), 1], dtype=np.int64)
    if datasetname == 'amazon':
        anomaly_pos = np.random.choice(np.arange(2, np.size(idx_test, 0)), anomaly_num, replace=False)
    else:
        anomaly_pos = np.random.choice(np.arange(1, np.size(idx_test, 0)), anomaly_num, replace=False)
    idx_test[anomaly_pos] = 0



    idx_anomalies = np.nonzero(idx_test.squeeze() == 0)
    idx_normal = np.nonzero(idx_test.squeeze() == 1)
    test_aedge = np.zeros([np.size(idx_test, 0), 5], dtype=np.int64)
    test_aedge[idx_normal] = test
    test_edge = processEdges2(idx_anomalies[0], test_aedge, adj, max_idx1, max_idx2, datasetname)
    synthetic_test = np.concatenate((test_edge, idx_test), axis=1)
    synthetic_train = np.concatenate((train, idx_train), axis=1)


    return  synthetic_train, synthetic_test

def processEdges2(idx_anomalies, test_aedge, adj, max_idx1, max_idx2, datasetname):

    for idx in idx_anomalies:
        flag = 0
        th_1 = np.max(test_aedge[0:idx, 0]) + 1
        th_2 = np.max(test_aedge[0:idx, 1]) + 1

        idx_1 = np.random.choice(th_1, 1, replace=False)[0]
        idx_2 = np.random.choice(th_2, 1, replace=False)[0]
        while idx_1 == idx_2:
            idx_1 = np.random.choice(th_1, 1, replace=False)[0]
            idx_2 = np.random.choice(th_2, 1, replace=False)[0]

        while adj[idx_1][idx_2] != 0:
            idx_1 = np.random.choice(th_1, 1, replace=False)[0]
            idx_2 = np.random.choice(th_2, 1, replace=False)[0]
            while idx_1 == idx_2:
                idx_1 = np.random.choice(th_1, 1, replace=False)[0]
                idx_2 = np.random.choice(th_2, 1, replace=False)[0]
        while flag == 0:
            for edge in test_aedge[0:idx, :]:
                if idx_1 == edge[0] and idx_2 == edge[1]:
                    flag = 1
                    break
                else:
                    continue
            if flag == 0:
                test_aedge[idx, 0] = idx_1
                test_aedge[idx, 1] = idx_2
                test_aedge[idx, 4] = np.random.randint(2)
                if  datasetname == 'amazon':
                    test_aedge[idx, 3] = test_aedge[idx - 1, 3]
                else:
                    test_aedge[idx, 3] = test_aedge[idx - 1, 3] + 1
                break
            else:
                idx_1 = np.random.choice(th_1, 1, replace=False)[0]
                idx_2 = np.random.choice(th_2, 1, replace=False)[0]
                while idx_1 == idx_2:
                    idx_1 = np.random.choice(th_1, 1, replace=False)[0]
                    idx_2 = np.random.choice(th_2, 1, replace=False)[0]
                flag = 0

    return test_aedge




def anomaly_generation_yelp(anomaly_percent, data):
    ano_edges = data[data[:, 5] == 0]
    nor_edges = data[data[:, 5] == 1]


    if anomaly_percent < 0.06:
        ano_num = int(nor_edges.shape[0] * anomaly_percent)
        selected_indices = np.random.choice(ano_edges.shape[0], ano_num, replace=False)
        selected_ano_edges = ano_edges[selected_indices]
        result_edges = np.concatenate((selected_ano_edges, nor_edges))
        result_edges = result_edges[np.argsort(result_edges[:, 3])]
        return result_edges

    elif anomaly_percent > 0.11:
        nor_num = int(ano_edges.shape[0] / anomaly_percent)
        selected_indices = np.random.choice(nor_edges.shape[0], nor_num, replace=False)
        selected_nor_edges = nor_edges[selected_indices]
        result_edges = np.concatenate((selected_nor_edges, ano_edges))
        result_edges = result_edges[np.argsort(result_edges[:, 3])]
        return result_edges
