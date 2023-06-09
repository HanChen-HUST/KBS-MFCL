import copy
import torch
import networkx as nx
import random
from torch_geometric.utils import degree, to_undirected
import numpy as np

def calculate_feature_weighted(x, node_centrality):
    x = copy.deepcopy(x)
    x = x.abs()
    w = x.t() @ node_centrality   
    w = w.log() 
    s = (w.max() - w) / (w.max() - w.mean())

    return s

def feature_mask(x, w, p: float = 0.4, threshold: float = 0.6):

    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = w

    drop_mask = torch.bernoulli(drop_mask).to(torch.bool)
    x = x.clone()
    x[:,drop_mask] = 0.

    return x



def cal_pagerank(edge_index, damp: float = 0.85, iter: int = 1000):
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    for edge in edges:
        G.add_edge(edge[0].item(), edge[1].item())
    try:
        pagerank_list = nx.pagerank(G, alpha=damp, max_iter=iter)
    except Exception:
        print("在默认迭代次数中没有收敛")
    pagerank_list = list(pagerank_list.values())
    x = torch.tensor(pagerank_list).to(edge_index.device).to(torch.float32)
    return x

def cal_eigenvector_centrality(edge_index):
    G = nx.Graph()
    edges = list(zip(edge_index[0], edge_index[1]))
    for edge in edges:
        G.add_edge(edge[0].item(), edge[1].item())
    #eigenvector_dict = nx.eigenvector_centrality(G)
    eigenvector_dict =nx.eigenvector_centrality_numpy(G)
    return torch.tensor(list(eigenvector_dict.values())).to(edge_index.device).to(torch.float32)

def cal_degree_centrality(edge_index):
    edge_index_ = to_undirected(edge_index)
    ind, deg = np.unique(edge_index_.cpu().numpy(), return_counts=True)
    deg_col = torch.tensor(deg)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def drop_edge(edge_index, edge_weights, p : float=0.4, threshold:float =0.6):
    edge_weights = edge_weights / edge_weights.mean() *p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, mask]


def cal_edge_pagerank(edge_index, aggr: str = 'mean'):
    pr = cal_pagerank(edge_index)
    pr_row = pr[edge_index[0]].to(torch.float32) #上一个节点的中心性
    pr_col = pr[edge_index[1]].to(torch.float32) #下一个节点的中心性
    w_row = torch.log(pr_row)
    w_col = torch.log(pr_col)
    if aggr == 'sink':
        w = w_col
    elif aggr == 'source':
        w = w_row
    elif aggr == 'mean':
        w = (w_row + w_col) * 0.5
    else:
        w = w_col
    if w.max() == w.mean():
        weights = w
    else:
        weights = (w.max() - w) / (w.max() - w.mean())

    return weights


def cal_edge_eignvetor(edge_index, aggr: str = 'mean'):
    evc = cal_eigenvector_centrality(edge_index)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    w_row, w_col = s[edge_index[0]], s[edge_index[1]]
    if aggr == 'sink':
        w = w_col
    elif aggr == 'source':
        w = w_row
    elif aggr == 'mean':
        w = (w_row + w_col) * 0.5
    else:
        w = w_col
    if w.max() == w.mean():
        weights = w
    else:
        weights = (w.max() - w) / (w.max() - w.mean())

    return weights


def cal_edge_degree(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    if s_col.max() == s_col.mean():
        weights = s_col
    else:
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights



def get_augmented_by_node_mask(batch_data):
    batch_data= copy.copy(batch_data) 
    page_rank_centrality = cal_pagerank(edge_index=batch_data.edge_index)
    feature_weighted = calculate_feature_weighted(batch_data.x, page_rank_centrality)
    batch_data.x = feature_mask(batch_data.x, feature_weighted)
    return batch_data


def get_augmented_by_node_mask_degree(batch_data):
    batch_data = copy.copy(batch_data)
    degree_centrality = cal_degree_centrality(edge_index=batch_data.edge_index)
    feature_weighted = calculate_feature_weighted(batch_data.x, degree_centrality)
    batch_data.x = feature_mask(batch_data.x, feature_weighted)
    return batch_data


def get_augmented_by_node_mask_eignvector(batch_data):
    batch_data= copy.copy(batch_data) 
    eignvector_centrality = cal_eigenvector_centrality(edge_index=batch_data.edge_index)
    print(eignvector_centrality.shape)
    print(batch_data.x.shape)
    feature_weighted = calculate_feature_weighted(batch_data.x, eignvector_centrality)
    batch_data.x = feature_mask(batch_data.x, feature_weighted, p=0.2) 
    return batch_data


def get_augmented_by_drop_edge(batch_data):
    batch_data = copy.copy(batch_data)
    edge_weighted = cal_edge_pagerank(batch_data.edge_index)
    edge_index = drop_edge(batch_data.edge_index,edge_weighted)


    edges = edge_index.detach().numpy()
    num_node, _ = batch_data.x.size()
    _, nums_edge = edges.shape
    idx_not_missing =[n for n in range(num_node) if (n in edges[0]or n in edges[1])]

    num_node_remained = len(idx_not_missing)
    batch_data.x = batch_data.x[idx_not_missing]

    batch_data.batch = batch_data.batch[idx_not_missing]
    idx_dict = {idx_not_missing[n]:n for n in range(num_node_remained)}
    edges = [[idx_dict[edges[0, n]], idx_dict[edges[1, n]]] for n in range(nums_edge) if not edges[0, n] == edges[1, n]]
    if len(edges)>0:
        batch_data.edge_index = torch.tensor(edges).transpose(0, 1)
    return batch_data


def get_augmented_by_drop_edge_eignvector(batch_data):
    batch_data = copy.copy(batch_data)
    edge_weighted = cal_edge_eignvetor(batch_data.edge_index)
    edge_index = drop_edge(batch_data.edge_index,edge_weighted)

    edges = edge_index.detach().numpy()
    num_node, _ = batch_data.x.size()
    _, nums_edge = edges.shape
    idx_not_missing =[n for n in range(num_node) if (n in edges[0]or n in edges[1])]

    num_node_remained = len(idx_not_missing)
    batch_data.x = batch_data.x[idx_not_missing]
    batch_data.batch = batch_data.batch[idx_not_missing]
    idx_dict = {idx_not_missing[n]:n for n in range(num_node_remained)}
    edges = [[idx_dict[edges[0, n]], idx_dict[edges[1, n]]] for n in range(nums_edge) if not edges[0, n] == edges[1, n]]
    if len(edges)>0:
        batch_data.edge_index = torch.tensor(edges).transpose(0, 1)
    return batch_data


def get_augmented_by_drop_edge_degree(batch_data):
    batch_data = copy.copy(batch_data)
    edge_weighted = cal_edge_degree(batch_data.edge_index)
    edge_index = drop_edge(batch_data.edge_index,edge_weighted)

    edges = edge_index.detach().numpy()
    num_node, _ = batch_data.x.size()
    _, nums_edge = edges.shape
    idx_not_missing =[n for n in range(num_node) if (n in edges[0]or n in edges[1])]

    num_node_remained = len(idx_not_missing)
    batch_data.x = batch_data.x[idx_not_missing]
    batch_data.batch = batch_data.batch[idx_not_missing]
    idx_dict = {idx_not_missing[n]:n for n in range(num_node_remained)}
    edges = [[idx_dict[edges[0, n]], idx_dict[edges[1, n]]] for n in range(nums_edge) if not edges[0, n] == edges[1, n]]
    if len(edges)>0:
        batch_data.edge_index = torch.tensor(edges).transpose(0, 1)
    return batch_data



