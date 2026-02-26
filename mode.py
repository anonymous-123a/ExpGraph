import time

import torch
import torch.nn as nn
from Component import *
from torch_geometric.nn import InnerProductDecoder
from utils import *
import torch.nn.functional as F



class Model(nn.Module):
    warm_up = 0
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        config = MyConfig(k=args.neighbour_num, window_size=args.window_size, hidden_size=args.embedding_dim,
                     intermediate_size=args.embedding_dim, num_attention_heads=args.num_attention_heads,
                     weight_decay=args.weight_decay, snaps_num=args.window_size)

        self.node_encoding = NodeEncodeModel(args, config)
        self.graph_encoding2 = GraphEncodeModel_HAN2(args, config)
        self.predict = nn.Linear(args.hidden_size_HANLayer * args.num_heads_HANLayer, 1)
        self.contrastive2 = Contrastive2(args.device, args)
        self.dec = InnerProductDecoder()
        self.relational_subgraph_encoding2 = RelationalSubgraphEncoder2(args=args)
        self.landmark_matching = LandmarkMarkMatchs(args=args)
        self.sematic_predict = SemanticAttention(args.hidden_dim_GRU)

        self.m_raw = nn.Parameter(torch.randn(1))
        self.sigmoid = nn.Sigmoid()


        self.predict2 = nn.Linear(args.hidden_dim_GRU, 1)

    def forward(self, y, id_type_map, int_embedding, hop_embedding, time_embedding, type_embedding, neighbours_edges,
                 edges_snap, all_z, all_node_idx, mask_hsg_rs=None, combined_graph_data=None, edges_type=None, istest=False):


        outputs_edges = self.node_encoding(int_embedding, hop_embedding, time_embedding, type_embedding)

        edges_embeddings, hsg_combined = self.graph_encoding2(outputs_edges, type_embedding, combined_graph_data, hop_embedding)

        R_graph_embeddings_edges = self.relational_subgraph_encoding2(combined_hsg= hsg_combined, mask_hsg_rs=mask_hsg_rs, hsg_edges_size0=outputs_edges.size(0))

        predict2, sim_cs, zloss = self.landmark_matching(R_graph_embeddings_edges = R_graph_embeddings_edges, edges_embeddings = edges_embeddings, istest=istest)

        predict2 = nn.Sigmoid()(predict2)

        predict1 = self.predict(edges_embeddings)

        predict1 = nn.Sigmoid()(predict1)
        m = self.sigmoid(self.m_raw)
        predict = m * predict1 + (1-m) * predict2
        bce_loss = F.binary_cross_entropy(predict, y, reduction='none')


        all_z, all_node_idx = EdgesToNodes(edges_embeddings, edges_snap, all_z, all_node_idx)
        label_rectifier = self.dec(all_z[-1], edges_snap.t(), sigmoid=True)
        label_rectifier = label_rectifier.unsqueeze(1)
        reg_loss = torch.norm(label_rectifier - predict1, dim=1, p=2).mean()
        reg_loss = reg_loss + zloss

        if len(all_z) > self.args.window_size:
            nce_loss = self.contrastive2(all_z, all_node_idx)
        else:
            nce_loss = 0


        return bce_loss, reg_loss,nce_loss, all_z, all_node_idx, predict