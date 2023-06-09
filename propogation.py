import torch
import copy
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GAT
from torch.nn import Linear
from torch.nn.functional import relu
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict
from typing import Tuple


class Encoder(torch.nn.Module):
    """
    图编码器
    """
    def __init__(self, in_channels: int=768, out_channels: int=128, activation=relu, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2 
        self.k = k
        self.skip = skip
        self.convs = torch.nn.ModuleList()

        if not self.skip:
            self.convs.append(base_model(in_channels, 2 * out_channels).jittable())
            for _ in range(1, k-1):
                self.convs.append(base_model(2 * out_channels, 2 * out_channels))
            self.convs.append(base_model(2 * out_channels, out_channels))
        else:
            self.fc_skip = Linear(in_channels, out_channels)
            self.convs.append(base_model(in_channels, out_channels))
            for _ in range(1, k):
                self.convs.append(base_model(out_channels, out_channels))

        self.activation = activation
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch):
        if not self.skip:
            init_x = copy.copy(x)
            init_xs = []
            for i in range(self.k):
                init_x = self.activation(self.convs[i](init_x, edge_index))
                init_xs.append(init_x)
            xpool = [global_mean_pool(init_x, batch) for init_x in init_xs]
            init_x = torch.cat(xpool, 1)
            return init_x, torch.cat(init_xs, 1)
        else:
            x = self.activation(self.convs[0](x, edge_index))
            xs = [self.fc_skip(x), x] 
            for i in range(1, self.k):
                u =sum(xs)
                xs.append(self.activation(self.convs[i](u,edge_index)))
            xpool = [global_mean_pool(x, batch) for x in xs[1:]]  
            x = torch.cat(xpool, 1) 
            return x, torch.cat(xs, 1)
    
    def get_embedding(self, data):
        with torch.no_grad():
            x, edge_index, batch = data.x, data.edge_index, data.batch
            graph_embed, _ = self.forward(x, edge_index, batch)
            return graph_embed


class ObjectiveLoss_Graph(torch.nn.Module):
    def __init__(self, num_hidden: int, num_gc_layers: int, tau: float = 0.07):
        super(ObjectiveLoss_Graph, self).__init__()
        self.encoder: Encoder = Encoder(768, 256, k=2)
        self.tau = tau
        
        self.num_hidden = num_hidden
        self.embedding_dim = self.num_hidden * num_gc_layers 
        self.graph_aligner = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU()
        )
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0) 


    def forward(self, batch_data, batch_data2):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        graph_embed1, _ = self.encoder(x, edge_index, batch)
        z1 = self.graph_aligner(graph_embed1)
        x, edge_index, batch = batch_data2.x, batch_data2.edge_index, batch_data2.batch
        graph_embed2, _ = self.encoder(x, edge_index, batch)
        z2 = self.graph_aligner(graph_embed2)

        return z1, z2
        
    def cal_loss(self, z1: torch.Tensor, z2: torch.Tensor): 
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.mm(z1, z2.t()) 

        batch_size, _ = z1.size() 
        nt = lambda x: torch.exp(x / self.tau) 
        
        sim_matrix = nt(sim)
        
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim) 
        loss = - torch.log(loss).mean() 
        return loss


class ObjectiveLoss_TextImage(nn.Module):
    def __init__(self, configs, tau: float = 0.07, shared_dim: int = 768, sim_dim:int=768):
        super(ObjectiveLoss_TextImage, self).__init__()
        self.text_attention = Text_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim)
        
        self.image_attention = Image_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim)
        self.tau = tau
        self.conv = nn.Conv2d(2048, 768, 1)
        self.bn = nn.BatchNorm2d(768)
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.init_emb()


    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0) 

    def forward(self, batch_data):
        root_feature, img  = batch_data.root_feature,batch_data.img

        root_length = len(root_feature)
        root_img_mask = torch.ones(root_length, 1).cuda()
        img_root_mask = torch.ones(root_length, 1).cuda()
        root_feature = torch.squeeze(root_feature, dim=1) 
        root_feature = self.text_attention(root_feature,root_img_mask)

        img = F.relu(self.bn(self.conv(img)))  
        img = img.view(img.shape[0], img.shape[1], -1)  
        img = img.permute([0, 2, 1]) 
        img = self.image_attention(img, img_root_mask)
        root_feature = torch.squeeze(root_feature,dim=1)
        img = torch.squeeze(img, dim=1)
        root_feature = self.text_aligner(root_feature)
        img = self.image_aligner(img)

        return root_feature, img
        
    def cal_loss_t(self, z1: torch.Tensor, z2: torch.Tensor): 
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        sim = torch.mm(z1, z2.t())
        batch_size, _ = z1.size() 
       
        nt = lambda x: torch.exp(x / self.tau) 
        
        sim_matrix = nt(sim)
        
        
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim) 
        loss = - torch.log(loss).mean() 

        return loss
    

class LayerNormalization(nn.Module):
    def __init__(self, features_count, epsilon=1e-6):
        super().__init__()
        self.gain = nn.Parameter(
            torch.ones(features_count), requires_grad=True)
        self.bias = nn.Parameter(
            torch.zeros(features_count), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias


class TextImage_Transformer(nn.Module):
    def __init__(self, ct: EasyDict, feature_dim: int):
        super().__init__()

        self.input_norm = LayerNormalization(feature_dim)
        input_dim = feature_dim
        self.tf_context = TransformerEncoder(
                ct.atn_ct_num_layers, input_dim, ct.atn_ct_num_heads,
                input_dim, ct.dropout)
            
        init_network(self, 0.01)

    def forward(self, features, mask, hidden_state):
        features = self.input_norm(features)
        ctx = self.tf_context(
                hidden_state, features, features, mask)
        ctx = torch.mean(ctx, dim=1,keepdim=True)
        return ctx


class Text_Transformer(nn.Module):
    def __init__(self, ct: EasyDict, feature_dim: int):
        super().__init__()

        self.input_norm = LayerNormalization(feature_dim)
        input_dim = feature_dim
        self.embedding = PositionalEncoding(
            input_dim, ct.dropout, max_len=1000)

        self.tf = TransformerEncoder(
            ct.num_layers, input_dim, ct.num_heads, input_dim,
            ct.dropout)

        init_network(self, 0.01)

    def forward(self, features, mask):
        features = self.input_norm(features)
        features = self.embedding(features)
        features = self.tf(features, features, features, mask) 

        pooled = torch.mean(features, dim=1,keepdim=True)
        return pooled
    

class Image_Transformer(nn.Module):
    def __init__(self, ct: EasyDict, feature_dim: int):
        super().__init__()
        
        self.input_norm = LayerNormalization(feature_dim)
        input_dim = feature_dim
        self.tf = TransformerEncoder(
            ct.num_layers, input_dim, ct.num_heads, input_dim,
            ct.dropout)

        init_network(self, 0.01)

    def forward(self, features, mask):
        features = self.input_norm(features)
        features = self.tf(features, features, features, mask) 
        pooled = torch.mean(features, dim=1,keepdim=True)
        return pooled

    
class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_prob=0., max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        dimension = torch.arange(0, dim).float()
        div_term = 10000 ** (2 * dimension / dim)
        pe[:, 0::2] = torch.sin(position / div_term[0::2])
        pe[:, 1::2] = torch.cos(position / div_term[1::2])
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        if step is None:
            x = x + self.pe[:x.size(1), :]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, layers_count, d_model, heads_count, d_ff, dropout_prob):
        super().__init__()
        self.d_model = d_model
        assert layers_count > 0
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads_count, d_ff, dropout_prob)
                for _ in range(layers_count)])

    def forward(self, query, key, value, mask):
        batch_size, query_len, embed_dim = query.shape
        batch_size, key_len, embed_dim = key.shape
        mask = (1 - mask.unsqueeze(1).expand(batch_size, query_len, key_len))
        mask = mask == 1
        sources = None
        for encoder_layer in self.encoder_layers:
            sources = encoder_layer(query, key, value, mask)
        return sources


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads_count, d_ff, dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention_layer = Sublayer(
            MultiHeadAttention(heads_count, d_model, dropout_prob), d_model)
        
        self.pointwise_feedforward_layer = Sublayer(
            PointwiseFeedForwardNetwork(d_ff, d_model, dropout_prob), d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, sources_mask):
        sources = self.self_attention_layer(query, key, value, sources_mask)
        sources = self.dropout(sources)
        sources = self.pointwise_feedforward_layer(sources)
        return sources


class Sublayer(nn.Module):
    def __init__(self, sublayer, d_model):
        super(Sublayer, self).__init__()
        self.sublayer = sublayer
        self.layer_normalization = LayerNormalization(d_model)

    def forward(self, *args):
        x = args[0]
        x = self.sublayer(*args) + x
        return self.layer_normalization(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads_count, d_model, dropout_prob):
        super().__init__()
        assert d_model % heads_count == 0,\
            f"model dim {d_model} not divisible by {heads_count} heads"
        self.d_head = d_model // heads_count
        self.heads_count = heads_count
        self.query_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.key_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.value_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.final_projection = nn.Linear(d_model, heads_count * self.d_head)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=3)
        self.attention = None

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, d_model = query.size()
        d_head = d_model // self.heads_count 
        query_projected = self.query_projection(query)
        key_projected = self.key_projection(key)
        value_projected = self.value_projection(value)
        batch_size, key_len, d_model = key_projected.size()
        batch_size, value_len, d_model = value_projected.size()

        query_heads = query_projected.view(
            batch_size, query_len, self.heads_count, d_head).transpose(1, 2)
        key_heads = key_projected.view(
            batch_size, key_len, self.heads_count, d_head).transpose(1, 2)
        value_heads = value_projected.view(
            batch_size, value_len, self.heads_count, d_head).transpose(1, 2)
        
        attention_weights = self.scaled_dot_product(
            query_heads, key_heads)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(attention_weights)
            attention_weights = attention_weights.masked_fill(
                mask_expanded, -1e18)
            
        
        attention = self.softmax(attention_weights)
        attention_dropped = self.dropout(attention)

        context_heads = torch.matmul(
            attention_dropped, value_heads)
        
        context_sequence = context_heads.transpose(1, 2)
        context = context_sequence.reshape(
            batch_size, query_len, d_model)
        
        final_output = self.final_projection(context)

        return final_output

    def scaled_dot_product(self, query_heads, key_heads):
        key_heads_transposed = key_heads.transpose(2, 3)#最后两维交换
        dot_product = torch.matmul(
            query_heads, key_heads_transposed)
        attention_weights = dot_product / np.sqrt(self.d_head)
        return attention_weights


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_ff, d_model, dropout_prob):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout_prob),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob))

    def forward(self, x):
        return self.feed_forward(x)

def truncated_normal_fill(
        shape: Tuple[int], mean: float = 0, std: float = 1,
        limit: float = 2) -> torch.Tensor:
    num_examples = 8
    tmp = torch.empty(shape + (num_examples,)).normal_()
    valid = (tmp < limit) & (tmp > -limit)
    _, ind = valid.max(-1, keepdim=True)
    return tmp.gather(-1, ind).squeeze(-1).mul_(std).add_(mean)

def init_weight_(w, init_gain=1):
    w.copy_(truncated_normal_fill(w.shape, std=init_gain))


def init_network(net: nn.Module, init_std: float):
    for key, val in net.named_parameters():
        if "weight" in key or "bias" in key:
            init_weight_(val.data, init_std)

    










    




        

    



