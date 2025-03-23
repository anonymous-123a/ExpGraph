import datetime
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix
from sklearn.cluster import SpectralClustering


def anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m, seed = 1):   #data的前两列必须是源节点和目标节点
    np.random.seed(seed)
    print('[#s] generating anomalous dataset...\n', datetime.datetime.now())
    # print('[#s] initial network edge percent: #.1f##, anomaly percent: #.1f##.\n', datetime.datetime.now(),
    #       ini_graph_percent * 100, anomaly_percent * 100)

    # ini_graph_percent = 0.5;
    # anomaly_percent = 0.05;


    # # select part of edges as in the training set
    # train = data[0:train_num, :]
    #
    # # select the other edges as the testing set
    # test = data[train_num:, :]

    #data to adjacency_matrix

    # vertices_id = np.unique(np.concatenate((np.expand_dims(data[:, 0].T, axis=1), np.expand_dims(data[:, 1].T, axis=1)), axis=1))
    # vertices_idx = list(range(len(vertices_id)))
    # vertices_id_to_idx_map = {vertices_id[i]: i for i in vertices_idx}
    adjacency_matrix = edgeList2Adj(data[:, 0:2])

    # clustering nodes to clusters using spectral clustering
    # kk = 42 #3#10#42#42
    # sc = SpectralClustering(kk, affinity='precomputed', n_init=10, assign_labels = 'discretize',n_jobs=-1)
    # labels = sc.fit_predict(adjacency_matrix)


    # generate fake edges that are not exist in the whole graph, treat them as
    # anamalies
    # id_1 = np.expand_dims(np.transpose(np.random.choice(np.arange(np.min(data[:, 0]), np.max(data[:, 0])), m)), axis=1)
    # id_2 = np.expand_dims(np.transpose(np.random.choice(np.arange(np.min(data[:, 1]), np.max(data[:, 1])), m)), axis=1)
    id_1 = np.expand_dims(np.transpose(np.random.choice(np.arange(int(np.min(data[:, 0])),int(np.max(data[:, 0]))), m)), axis=1)
    id_2 = np.expand_dims(np.transpose(np.random.choice(np.arange(int(np.min(data[:, 1])),int(np.max(data[:, 1]))), m)), axis=1)
    generate_edges = np.concatenate((id_1, id_2), axis=1)

    ####### genertate abnormal edges ####
    # fake_edges = np.array([x for x in generate_edges if labels[x[0]] != labels[x[1]]])
    fake_edges = generate_edges

    fake_edges = processEdges(fake_edges, data[:, 0:2])


    #anomaly_num = 12#int(np.floor(anomaly_percent * np.size(test, 0)))
    anomaly_num = int(np.floor(anomaly_percent * np.size(data, 0)))
    anomalies = fake_edges[0:anomaly_num, :]
    anomalies = np.concatenate((anomalies, np.zeros([np.size(anomalies, 0), 3])),axis=1)

    idx_data = np.ones([np.size(data, 0) + anomaly_num, 1], dtype=np.int32)
    # randsample: sample without replacement
    # it's different from datasample!

    anomaly_pos = np.random.choice(np.arange(1, np.size(idx_data, 0)), anomaly_num, replace=False)

    #anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200

    idx_data[anomaly_pos] = 0
    synthetic_data = np.concatenate((np.zeros([np.size(idx_data, 0), np.size(data, 1)], dtype=np.int32), idx_data), axis=1)

    idx_anomalies = np.nonzero(idx_data.squeeze() == 0)
    idx_normal = np.nonzero(idx_data.squeeze() == 1)

    synthetic_data[idx_anomalies, :-1] = anomalies
    synthetic_data[idx_normal, :-1] = data

    # train_num = int(np.floor(ini_graph_percent * np.size(synthetic_data, 0)))
    # syn_train = synthetic_data[0:train_num, :]
    # syn_test = synthetic_data[train_num:, :]

    # train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (syn_train[:, 0], syn_train[:, 1])),
    #                        shape=(np.max(syn_train[:, 0:2]), np.max(syn_train[:, 0:2])))
    # # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n) #TODO: node addition
    # train_mat = train_mat + train_mat.transpose()

    return synthetic_data#, train_mat

def anomaly_generation2(ini_graph_percent, anomaly_percent, data, n, m,seed = 1):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train_mat: the training network with square matrix format, the training
                        network edges for initial model training;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    # The actual generation method used for Netwalk(shown in matlab version)
    # Abort the SpectralClustering
    np.random.seed(seed)
    print('[%s] generating anomalous dataset...\n'% datetime.datetime.now())
    print('[%s] initial network edge percent: %.2f, anomaly percent: %.2f.\n'%(datetime.datetime.now(),
          ini_graph_percent , anomaly_percent ))

    # ini_graph_percent = 0.5;
    # anomaly_percent = 0.05;
    train_num = int(np.floor(ini_graph_percent * m))

    # select part of edges as in the training set
    train = data[0:train_num, :]

    # select the other edges as the testing set
    test = data[train_num:, :]

    #data to adjacency_matrix
    #adjacency_matrix = edgeList2Adj(data)

    # clustering nodes to clusters using spectral clustering
    # kk = 3 #3#10#42#42
    # sc = SpectralClustering(kk, affinity='precomputed', n_init=10, assign_labels = 'discretize',n_jobs=-1)
    # labels = sc.fit_predict(adjacency_matrix)


    # generate fake edges that are not exist in the whole graph, treat them as
    # anamalies
    # 真就直接随机生成
    idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1)

    ####### genertate abnormal edges ####
    #fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    # 移除掉self-loop以及真实边
    fake_edges = processEdges(fake_edges, data)

    #anomaly_num = 12#int(np.floor(anomaly_percent * np.size(test, 0)))
    # 按比例圈定要的异常边
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    anomalies = fake_edges[0:anomaly_num, :]

    # 按照总边数（测试正常+异常）圈定标签
    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    # randsample: sample without replacement
    # it's different from datasample!

    # 随机选择异常边的位置
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)

    #anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200
    # 选定的位置定为1
    idx_test[anomaly_pos] = 1

    # 汇总数据，按照起点，终点，label的形式填充，并且把对应的idx找出
    synthetic_test = np.concatenate((np.zeros([np.size(idx_test, 0), 2], dtype=np.int32), idx_test), axis=1)
    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)
    synthetic_test[idx_anomalies, 0:2] = anomalies
    synthetic_test[idx_normal, 0:2] = test

    # coo:efficient for matrix construction ;  csr: efficient for arithmetic operations
    # coo+to_csr is faster for small matrix, but nearly the same for large matrix (size: over 100M)
    #train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] , train[:, 1])),shape=(n, n))
    train_mat = coo_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0], train[:, 1])), shape=(n, n)).tocsr()
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n)
    train_mat = train_mat + train_mat.transpose()

    return synthetic_test, train_mat, train

def processEdges(fake_edges, data):
    """
    remove self-loops and duplicates edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # b:list->set
    # Time cost rate is proportional to the size

    # idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)
    #
    # tmp = fake_edges[idx_fake]
    # tmp[:, [0, 1]] = tmp[:, [1, 0]]
    #
    # fake_edges[idx_fake] = tmp

    idx_remove_dups = np.nonzero(np.logical_or((fake_edges[:, 0] - fake_edges[:, 1] < 0), (fake_edges[:, 0] - fake_edges[:, 1] > 0)))

    fake_edges = fake_edges[idx_remove_dups]
    a = fake_edges.tolist()
    b = data.tolist()
    c = []

    for i in a:
        if i not in b:
            c.append(i)
    fake_edges = np.array(c)
    return fake_edges


def edgeList2Adj(data):
    """
    converting edge list to graph adjacency matrix
    :param data: edge list
    :return: adjacency matrix which is symmetric
    """

    data = tuple(map(tuple, data))

    n = int(np.max(data))+1  # Get size of matrix
    matrix = np.zeros((n, n))
    for user, item in data:
        matrix[int(user)][int(item)] = 1  # Convert to 0-based index.
        matrix[int(item)][int(user)] = 1  # Convert to 0-based index.
    return matrix

def anomaly_generation3(ini_graph_percent, anomaly_percent, data, m, max_idx1, max_idx2, datasetname=None):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    # np.random.seed(1)
    print('Generating anomalous dataset...\n')
    print('Initial network edge percent: ' + str(ini_graph_percent * 100))
    print('\n')
    print('Initial anomaly percent : ' + str(anomaly_percent * 100))
    print('\n')
    train_num = int(np.floor(ini_graph_percent * m))

    # region train and test edges
    # select top train_num edges(0:train_num) as in the training set
    train = data[:train_num, :]
    # train_ = np.unique(train)
    # n_train = len(train_)
    # n = max_idx1 + max_idx2 +2
    adj = np.zeros((max_idx1 + 1, max_idx2+1))
    for edge in train:
        adj[edge[0]][edge[1]] = adj[edge[0]][edge[1]] + 1
        # adj[edge[1]][edge[0]] = adj[edge[1]][edge[0]] + 1
    # nodes=np.unique(data)

    test = data[train_num:, :]
    # test_num = test.shape[0]

    # train = data[0:train_num, :]

    # select the other edges as the testing set
    # test = data[train_num:, :]
    # endregion

    # region Read Cluster labeling from membership.txt file

    # endregion

    # region Fake Edge Generation
    # generate fake edges that are not exist in the whole graph, treat them as anamalies
    # idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    # idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) + 1, axis=1)
    # generate_edges = np.concatenate((idx_1, idx_2), axis=1)

    # genertate abnormal edges
    # fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    # remove self-loops and duplicates and order fake edges
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    idx_test = np.ones([np.size(test, 0) + anomaly_num, 1], dtype=np.int64)
    idx_train = np.ones([np.size(train, 0), 1], dtype=np.int64)
    if datasetname == 'amazon':
        anomaly_pos = np.random.choice(np.arange(2, np.size(idx_test, 0)), anomaly_num, replace=False)
    else:
        anomaly_pos = np.random.choice(np.arange(1, np.size(idx_test, 0)), anomaly_num, replace=False)
    idx_test[anomaly_pos] = 0



    # endregion

    # region Take anomaly_num edges from fake_edges as anomaly

    # anomalies = fake_edges[0:anomaly_num, :]
    # endregion

    # region Put anomaly edges in test_edges in random positions

    # randsample: sample without replacement
    # it's different from datasample!



    # anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200



    # endregion

    # region Prepare Synthetic test Edges
    idx_anomalies = np.nonzero(idx_test.squeeze() == 0)
    idx_normal = np.nonzero(idx_test.squeeze() == 1)    #1为正常
    test_aedge = np.zeros([np.size(idx_test, 0), 5], dtype=np.int64)
    test_aedge[idx_normal] = test
    # synthetic_test[idx_anomalies, 0:2] = anomalies
    test_edge = processEdges2(idx_anomalies[0], test_aedge, adj, max_idx1, max_idx2, datasetname)
    synthetic_test = np.concatenate((test_edge, idx_test), axis=1)
    synthetic_train = np.concatenate((train, idx_train), axis=1)


    # endregion

    # train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] - 1, train[:, 1] - 1)),shape=(n, n))
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n) #TODO: node addition
    # train_mat = train_mat + train_mat.transpose()

    return  synthetic_train, synthetic_test #,n_train

def processEdges2(idx_anomalies, test_aedge, adj, max_idx1, max_idx2, datasetname):
    """
    remove self-loops and duplicates and order edge
    :param fake_edges: generated edge list
    :param data: orginal edge list
    :return: list of edges
    """
    # idx_fake = np.nonzero(fake_edges[:, 0] - fake_edges[:, 1] > 0)
    #
    # tmp = fake_edges[idx_fake]
    # tmp[:, [0, 1]] = tmp[:, [1, 0]]  # 调整前后顺序，使得边信息（node1，node2）中node1<=node2
    # fake_edges[idx_fake] = tmp
    for idx in idx_anomalies:
        # print(idx)
        flag = 0
        th_1 = np.max(test_aedge[0:idx, 0]) + 1
        th_1_min = np.min(test_aedge[0:idx, 0])
        th_2 = np.max(test_aedge[0:idx, 1]) + 1
        th_2_min = np.min(test_aedge[0:idx, 1])
        # th_1 = max_idx1
        # th_2 = max_idx2
        idx_1 = np.random.choice(th_1, 1, replace=False)[0]
        idx_2 = np.random.choice(th_2, 1, replace=False)[0]
        while idx_1 == idx_2:
            idx_1 = np.random.choice(th_1, 1, replace=False)[0]
            idx_2 = np.random.choice(th_2, 1, replace=False)[0]
        # print(idx_1)
        # print(adj[idx_1][idx_2])
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
