# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool

import math
import loralib as lora
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init

from .grit import *
from .model_utils import *

from itertools import groupby
from transformers import LlamaTokenizer

model_path = "/home/jzt/models/Llama-2-7B"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'left'

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 4096
    
    # adapter参数定义
    
    adapter_layer: int = 8
    adapter_dim: int = 512
    adapter_n_heads: int = 4


    num_hops: int = 2
    # lora
    lora_r: int = 16
    lora_alpha: int = 1
    lora_dropout: float = 0.2
    rrwp: int = 8
    
    n_mp_layers: int = 2
    n_encoder_layers: int = 2

    task_level: str = 'node'
    
    fans_out: Tuple[int] = (50, 50, 50)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # ε
        self.weight = nn.Parameter(torch.ones(dim)) #可学习参数γ

    def _norm(self, x):
        # RMSNorm
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 计算词向量元素两两分组以后，每组元素对应的旋转角度 
    # arange生成[0,2,4...126]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t = [0,....end]
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # t为列向量 freqs为行向量做外积
    # freqs.shape = (t.len(),freqs.len()) #shape (end,dim//2)
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 生成复数向量
    # torch.polar(abs,angle) -> abs*cos(angle) + abs*sin(angle)*j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # freqs_cis.shape  = (end,dim//2)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # ndim为x的维度数 ,此时应该为4
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # (1,x.shape[1],1,x.shape[-1])
    return freqs_cis.view(*shape)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 根据n_rep，拓展KV
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_heads = args.n_heads
        
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False,)
        
        
        self.lora_wq = LoraLinear(self.wq.in_features, lora_r=args.lora_r,
                                    lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        """self.lora_wv = LoraLinear(self.wv.in_features, lora_r=args.lora_r,
                                    lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)"""
        
        """self.lora_wq = LoraInjectedLinear(self.wq.in_features, self.wq.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)"""
        self.lora_wv = LoraInjectedLinear(self.wv.in_features, self.wv.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        
        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None

    
    def enable_cache(self):
        self.cache_enabled = True

    def disable_cache(self):
        self.cache_enabled = False
        self.cache_k, self.cache_v = None, None
    
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        graph_rep
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    
        
        xq = xq + self.lora_wq(graph_rep, x)
        # xv = xv + self.lora_wv(graph_rep, x)
        
        # xq = xq + self.lora_wq(x)
        xv = xv + self.lora_wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis) #嵌入RoPE位置编码

        keys = xk
        values = xv
        
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        if self.cache_enabled:
            if self.cache_k is None:
                assert start_pos == 0
                self.cache_k, self.cache_v = keys, values
            else:
                assert self.cache_k.size(2) >= start_pos
                self.cache_k = torch.cat([self.cache_k[:, :, :start_pos], keys], dim=2)
                self.cache_v = torch.cat([self.cache_v[:, :, :start_pos], values], dim=2)
                keys, values = self.cache_k, self.cache_v

        # 类型转换
        xq = xq.bfloat16()
        values = values.bfloat16()
        
        output = self._forward_scaled_dot_product_attention(xq, keys, values, attention_mask=mask)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)


    def _forward_scaled_dot_product_attention(self, q, k, v, attention_mask=None):
        if False and hasattr(F, "scaled_dot_product_attention"):
           return F.scaled_dot_product_attention(q, k, v, attention_mask if attention_mask is not None else None)
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
           
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_weights = torch.matmul(attn_weights, v)

        return attn_weights


class LoraLinear(nn.Module):
    def __init__(
        self, in_features, lora_r, lora_alpha, lora_dropout=0.2,
    ):
        super().__init__()

        if lora_r > in_features:
            raise ValueError(
                f"LoRA rank {lora_r} must be less or equal than {in_features}"
            )
        self.lora_r = lora_r
        self.lora_down = nn.Linear(in_features, lora_r, bias=False)  # 4096 * 16
        self.dropout = nn.Dropout(lora_dropout)
        # self.lora_up = nn.Linear(lora_r, out_features, bias=False)   # 16 * 4096
        self.scale = 1. * lora_alpha / lora_r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_up.weight)
        
    def forward(self, graph_rep, x: torch.Tensor):
        x = x.float()
        previous_dtype = x.dtype
        # x = x.to(self.lora_up.weight.dtype)
        result = self.lora_down(self.dropout(x))
        
        bsz = result.shape[0]
        
        temp_list = []
        for i in range(0, bsz):
            temp_list.append(result[i: i + 1])
        
        graph_list = []
        for i in range(0, bsz):
            graph_list.append(graph_rep[i: i + 1].squeeze())
        
        result_list = []
        for i in range(0, bsz):
            result_list.append(torch.matmul(temp_list[i], graph_list[i]) * self.scale)
        
        result = torch.cat(result_list, dim=0)
        result = result.to(previous_dtype)
        return result
    

class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, lora_r, lora_alpha, lora_dropout=0.2,
    ):
        """
        in_features: 4096
        out_features: 4096
        lora_r: 16
        """
        super().__init__()

        if lora_r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {lora_r} must be less or equal than {min(in_features, out_features)}"
            )
        self.lora_r = lora_r
        self.lora_down = nn.Linear(in_features, lora_r, bias=False)  # 4096 * 16
        self.dropout = nn.Dropout(lora_dropout)
        self.lora_up = nn.Linear(lora_r, out_features, bias=False)   # 16 * 4096
        self.scale = 1. * lora_alpha / lora_r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)


    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        x = x.to(self.lora_up.weight.dtype)
        result = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        result = result.to(previous_dtype)
        return result
    
    
# GNN-Adapter
class GriTGraphAdapter(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(GriTGraphAdapter, self).__init__()

        self.graph_adapter_layers = torch.nn.ModuleList()
        self.adapter_layer = params.adapter_layer
        # self.adapter_len = params.adapter_len

        self.dim = params.adapter_dim
        
        # 定义消息传播层
        self.mp = GritAdapterLayer(params)
        
    
        self.query_embed = nn.Parameter(torch.randn([self.adapter_layer, self.dim]))
        torch.nn.init.xavier_normal_(self.query_embed)
        # 没有decoder的过程
        
        

    def forward(self, node_configurations, input_attn, edge_index, mapping, input_node_pair_embed, batch):
        """
        :node_configurations: embedding of batched subgraph (N, seq_len, dim)
        :param input_attn: Attn (N, seq_len)
        :param edge_index: edge_index of batched subgraph
        :param mapping:
        :return:
        """
        bsz = node_configurations.shape[0]

        graph_rep_list = []

        for l in range(self.adapter_layer):
            query = self.query_embed[l: l+1].repeat(bsz, 1, 1)

            """for decoder_layer in self.decoder_layers:
                query = decoder_layer(query, node_configurations, input_attn)"""
            
            graph_rep = self.mp(query, edge_index, mapping, input_node_pair_embed, batch)
            
            graph_rep_list.append(graph_rep)

        graph_rep = torch.stack(graph_rep_list, dim=1)

        
        # 需要通过一个MLP结合node_rep和graph_rep
        # graph_rep = self.fuselayer(graph_rep, node_configurations)
        
        return graph_rep


# Grit mp层
class GritAdapterLayer(nn.Module):
    def __init__(self,
                 params: ModelArgs):
        super(GritAdapterLayer, self).__init__()
        self.dim = params.adapter_dim
        self.rrwp = params.rrwp
        self.n_mp_layers = params.n_mp_layers
        self.task_level = params.task_level
        self.params = params

        # 定义消息传播层
        self.mp_layer = nn.ModuleList()
        self.mp_layer.append(MultiHeadAttentionLayerGritSparse(0, embed_dim=self.dim, num_heads=params.adapter_n_heads,
                                                               initial_layer=True, rrwp=self.rrwp))

        for n_mp_layer in range(1, self.n_mp_layers):
            self.mp_layer.append(MultiHeadAttentionLayerGritSparse(n_mp_layer, embed_dim=self.dim, num_heads=params.adapter_n_heads,
                                                                   initial_layer=False, rrwp=self.rrwp))

        self.wp = nn.Sequential(nn.Linear(self.rrwp, self.dim),
                                nn.ReLU(),
                                nn.Linear(self.dim, self.dim))

        self.wn = nn.Sequential(nn.Linear(self.rrwp, self.dim),
                                nn.ReLU(),
                                nn.Linear(self.dim, self.dim))


        self.ind_mlp = nn.Sequential(nn.Linear(1, self.dim, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(self.dim, self.dim, bias=False))


    def forward(self, query, edge_index, mapping, input_node_pair_embed, batch):
        """
        :param input_embeds: embedding of batched subgraph (N, seq_len, dim)
        :param input_attn: Attn (N, seq_len)
        :param edge_index: edge_index of batched subgraph
        :param mapping:
        :return:
        """

        node_pos = input_node_pair_embed[edge_index[0] == edge_index[1]]
        node_pos = node_pos.view([node_pos.shape[0], 1, node_pos.shape[1]]).repeat(1, query.shape[1], 1)
        node_pos = self.wn(node_pos)
        query = query + node_pos

        out_list = []

        if self.task_level == 'graph':
            for query_idx in range(query.shape[1]):
                x_out, e_out = query[:, query_idx, :], self.wp(input_node_pair_embed)
                for layer in self.mp_layer:
                    x_out, e_out = layer(x=x_out, edge_index=edge_index,
                                         input_node_pair_embed=e_out)

                x_out = global_mean_pool(x_out, batch.to(x_out.device))
                out_list.append(x_out)

            query = torch.stack(out_list, dim=1)

        elif self.task_level == 'pair':
            for query_idx in range(query.shape[1]):

                x_out, e_out = query[:, query_idx, :], self.wp(input_node_pair_embed)
                ind = torch.zeros([mapping[1] - mapping[0], 1], dtype=query.dtype).to(query.device)
                ind[mapping[0] + 1] = 1.
                ind_emb = self.ind_mlp(ind).repeat(mapping.shape[0], 1)
                x_out = x_out + ind_emb

                for layer in self.mp_layer:
                    x_out, e_out = layer(x=x_out, edge_index=edge_index,
                                         input_node_pair_embed=e_out)

                out_list.append(x_out)

            query = torch.stack(out_list, dim=1)
            # query = torch.cat([query[mapping][:, :self.params.adapter_len//2,:], query[mapping + 1][:, self.params.adapter_len//2:, :]], dim=1)
            query = query[mapping]

        else:
            for query_idx in range(query.shape[1]):
                x_out, e_out = query[:, query_idx, :], self.wp(input_node_pair_embed)
                for layer in self.mp_layer:
                    x_out, e_out = layer(x=x_out, edge_index=edge_index,
                                         input_node_pair_embed=e_out)
                out_list.append(x_out)

            query = torch.stack(out_list, dim=1)
            query = query[mapping]

        return query



class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False,)    # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False,)    # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False,)    # up_proj
        
        

    def forward(self, x):
        
        # x.to(dtype=torch.float32)
        
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


    
    
class TransformerBlockWithAdapter(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.params = args
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    # 加入Adapter的forward
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        graph_rep: torch.Tensor
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask, graph_rep
        )
        
        h = h.to(dtype=torch.bfloat16)
        
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        # print("run ! ")
        
        return out


class ReduceDimsMLP(nn.Module):
    def __init__(self):
        super(ReduceDimsMLP, self).__init__()
        # 使用平均池化降维，从[40, 512, 768]到[40, 1, 768]
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # 一个全连接层来调整最后的大小，从[40, 1, 768]变为[40, 8, 768]
        self.fc = nn.Linear(768, 8 * 768)

    def forward(self, x):
        # x 的维度 [batch_size=40, seq_len=512, features=768]
        # 池化处理后维度变为 [batch_size=40, seq_len=1, features=768]
        x = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        # 线性层处理后扩展维度 [batch_size=40, seq_len=8, features=768]
        x = self.fc(x.squeeze(1)).view(-1, 8, 768)
        return x
    

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs,
                edge_index: torch.Tensor = None,
                node_configuration_ids: torch.Tensor = None,
                input_attention_mask: torch.Tensor = None):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.adapter_layer = params.adapter_layer
        self.rrwp = params.rrwp
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('node_configuration_ids', node_configuration_ids)
        self.register_buffer('input_attention_mask', input_attention_mask)
        
        
        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim,
        )

        # Graph-Adapter
        # self.graph_adapter = GriTGraphAdapter(params)
        
        # without GNN
        self.mlp = ReduceDimsMLP()
        
        # 降维映射 self.dim ---> self.adapter.dim 
        self.down_projection = nn.Sequential(
                    nn.Linear(self.params.dim, self.params.adapter_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.params.adapter_dim, self.params.adapter_dim, bias=False))
        
        # 升维映射 self.adaptet.dim ---> self.dim 
        self.up_projection = nn.Sequential(
                        nn.Linear(self.params.adapter_dim, self.params.adapter_dim, bias=False),
                        nn.ReLU(),
                        nn.Linear(self.params.adapter_dim, self.params.dim, bias=False))
        
        # Encoder-with-Attention
        # attention q k v
        # q即为用户需求query
        # k,v为融合了用户需求与配置表条目的向量
        self.attention_encoder = ConfigurationAttentionEncoder(params)
        
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlockWithAdapter(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False,)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
    
    
    def forward(self, input_ids, labels, node_ids, attention_mask=None, start_pos=0):
        
        _bsz, seqlen = input_ids.shape

        query_embeds = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis.to(query_embeds.device)

        position_id = torch.arange(seqlen).repeat(_bsz, 1).to(query_embeds.device)
        position_id = position_id - ((attention_mask == 0).sum(dim=-1)).unsqueeze(-1)
        position_id[position_id < 0] = 0
        freqs_cis = freqs_cis[position_id]
        
        if attention_mask is None:
            attention_mask = torch.ones(
                (_bsz, seqlen), dtype=torch.bool, device=query_embeds.device
            )
        
        
        temp_attention_mask = attention_mask

        start_pos, graph_rep, node_configurations_embeds, configurations_atte_mask = 0, None, None, None
        
        
        graph_rep = self.graphForward(node_ids=node_ids)
        graph_rep = torch.squeeze(graph_rep)
        
        graph_rep = graph_rep.float()
        graph_rep = self.up_projection(graph_rep)
        
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (_bsz, seqlen), query_embeds)
        
        
        h = query_embeds
        
        
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, attention_mask, graph_rep)
        
        h = self.norm(h)
        output = self.output(h)
        
        
        shift_outputs = output[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_outputs.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        
        
        # debug部分
        mask = shift_labels != -100
        filtered_labels = shift_labels[mask]
        
        predicted_tokens = torch.argmax(shift_logits, dim=1)
        predicted_tokens = predicted_tokens[mask]
        
        pre = tokenizer.batch_decode(predicted_tokens)
        x = tokenizer.batch_decode(filtered_labels)
        
        eval_pred = [item.split('</s>')[0] for item in pre]
        eval_pred = [item.split('\n\n###\n\n ')[-1] for item in eval_pred]
        
        eval_label = [item.split('</s>')[0] for item in x]
        eval_label = [item.split('\n\n###\n\n ')[-1] for item in eval_label]
        
        pred_strings = [''.join(group) for k, group in groupby(eval_pred, key=lambda x: x.isdigit()) if k]
        label_strings = [''.join(group) for k, group in groupby(eval_label, key=lambda x: x.isdigit()) if k]
        
        
        # 将字符串转换为整数列表
        eval_pred = [int(num) for num in pred_strings]
        eval_label = [int(num) for num in label_strings]
        
        common_elements = set(eval_pred) & set(eval_label)
        val_acc = len(common_elements) / len(eval_label)
        
        print('Pre---{}'.format(eval_pred))
        print('label---{}'.format(eval_label))
        print('Acc---{}'.format(val_acc))
        
        
        c_loss = self.criterion(shift_logits, shift_labels)

        return c_loss, val_acc
    
    
    
    @torch.inference_mode()
    def forward_inference(self, graph_rep, tokens: torch.Tensor, start_pos: int, attention_mask=None):
        _bsz, seqlen = tokens.shape

        query_embeds = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis.to(query_embeds.device)

        position_id = torch.arange(seqlen).repeat(_bsz, 1).to(query_embeds.device)
        position_id = position_id - ((attention_mask == 0).sum(dim=-1)).unsqueeze(-1)
        position_id[position_id < 0] = 0
        freqs_cis = freqs_cis[position_id]
        
        if seqlen == 1:
            attention_mask = attention_mask
            attention_mask = prepare_decoder_attention_mask(
                attention_mask, (_bsz, seqlen), query_embeds)
        elif start_pos == 0:
            # Generate first time
            if attention_mask is None:
                attention_mask = torch.ones(
                    (_bsz, seqlen), dtype=torch.bool, device=query_embeds.device
                )
            attention_mask = prepare_decoder_attention_mask(
                attention_mask, (_bsz, seqlen), query_embeds)
        else:
            raise NotImplementedError()
        
        h = query_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_cis, attention_mask, graph_rep)

        h = self.norm(h)
        output = self.output(h[:, -1, :])
        return output.float()
    
    
    
    def graphForward(self, node_ids):
        # RRWP 相对随机游走
        
        # node_num = 200000
        # print('node_num------------------')
        graph_size = 20
        
        if self.params.task_level == 'graph':
            subset, edge_index_sub, mapping, batch = batch_subgraph_graph_level(self.edge_index, node_ids,
                                                                        num_nodes=self.node_configuration_ids.shape[0],)

        elif self.params.task_level == 'pair':
            subset, edge_index_sub, mapping, batch = batch_subgraph_pair_level(self.edge_index, node_ids,
                                                                        num_nodes=self.node_configuration_ids.shape[0],)

        else:
            subset, edge_index_sub, mapping, batch = batch_subgraph(self.edge_index, node_ids,
                                                                        num_nodes=self.node_configuration_ids.shape[0],
                                                                        num_hops=self.params.num_hops,
                                                                        fans_out=self.params.fans_out
                                                                        )


        edge_index_full, input_node_pair_embed = add_full_rrwp(edge_index_sub, num_nodes=len(subset),
                                                                   walk_length=self.params.rrwp
                                                                   )


        node_configurations_ids, adapter_input_attn = self.node_configuration_ids[subset], self.input_attention_mask[subset]
        node_configurations_embeds = self.tok_embeddings(node_configurations_ids)

        node_configurations_embeds = node_configurations_embeds.float()
        
        node_configurations_embeds = self.down_projection(node_configurations_embeds)
            
            
        node_configurations_embeds = self.attention_encoder(node_configurations_embeds, adapter_input_attn)
        
        print("node_embed-----{}".format(node_configurations_embeds.shape))
        
        # without GNN [40, 512, 768] -> [2, 8, 1, 768]
        """graph_rep = self.graph_adapter(node_configurations_embeds, adapter_input_attn, edge_index_full, mapping,
                                           input_node_pair_embed, batch)"""
                                           
        graph_rep = self.mlp(node_configurations_embeds)
        
        # print("graph111-----{}".format(graph_rep.shape))
        bsz = len(node_ids)
        reshaped_tensor = graph_rep.view(bsz, graph_size, self.adapter_layer, self.params.adapter_dim)
        # reshaped_tensor = graph_rep.view(2, 20, 8, 768)
        graph_rep = torch.mean(reshaped_tensor, dim=1)
        graph_rep = graph_rep.unsqueeze(2)
        
        # print("graph222-----{}".format(graph_rep.shape))
        
        return graph_rep

    
    def generate(self, node_ids, input_ids, attention_mask,
            max_new_tokens: int,
            temperature: float = -1.,
            top_p: float = 1.,
            pad_token_id = 0
    ):
        bsz, prompt_size = input_ids.shape
        params = self.params
        self.enable_cache()

        total_len = prompt_size + max_new_tokens
        tokens = torch.full((bsz, total_len), pad_token_id).to(input_ids.device).long()
        tokens[:,:prompt_size] = input_ids

        start_pos = prompt_size
        prev_pos = 0

        _bsz, _ = tokens.shape
        
        # 得到graph_rep
        graph_rep = None
        graph_rep = self.graphForward(node_ids=node_ids)
        graph_rep = torch.squeeze(graph_rep)
        graph_rep = graph_rep.float()
        graph_rep = self.up_projection(graph_rep)
        
        for cur_pos in range(start_pos, total_len):
            logits = self.forward_inference(graph_rep, tokens[:, prev_pos:cur_pos], start_pos=prev_pos, attention_mask=attention_mask)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            attention_mask = torch.cat([attention_mask, torch.ones((bsz, 1)).to(attention_mask.device)], dim=-1)

        self.disable_cache()
        return tokens


    def enable_cache(self):
        for layer in self.layers:
            layer.attention.enable_cache()


    def disable_cache(self):
        for layer in self.layers:
            layer.attention.disable_cache()
            
    
    def set_trainable_params_new(self):
        param_adapter, param_lora  = [],  []
        
        # adapter = ["graph_adapter", "up_projection", "down_projection", "attention_encoder"]
        adapter = ["mlp", "up_projection", "down_projection", "attention_encoder"]
        
        for name, param in self.named_parameters():
            if any(n in name for n in adapter):
                param.requires_grad = True
                param.data = param.data.float()
                param_adapter.append(param)
            elif "lora" in name:
                param.requires_grad = True
                param.data = param.data.float()
                param_lora.append(param)
            else:
                param.requires_grad = False

        return param_adapter, param_lora


    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params


        return trainable_params, all_param



class ConfigurationAttentionEncoder(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(ConfigurationAttentionEncoder, self).__init__()
        self.n_encoder_layers = params.n_encoder_layers
        self.encoder = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_encoder_layers)])

    def forward(self, query, input_attn):
        for layer in self.encoder:
            query = layer(query, input_attn)
        return query



class EncoderLayer(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(EncoderLayer, self).__init__()
        self.dim = params.adapter_dim
        self.attention = EncoderSelfAttention(params)
        self.feed_forward = nn.Sequential(nn.Linear(self.dim, self.dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(2 * self.dim, self.dim))

        self.attention_norm = nn.LayerNorm(self.dim, eps=params.norm_eps)
        self.ffn_norm = nn.LayerNorm(self.dim, eps=params.norm_eps)

    def forward(self, query, input_attn):
        
        query = query + self.attention(query, input_attn)
        query = self.attention_norm(query)
        query = query + self.feed_forward(query)
        query = self.ffn_norm(query)
        return query



class EncoderSelfAttention(nn.Module):
    def __init__(self,
                 params: ModelArgs,
                 ):
        super(EncoderSelfAttention, self).__init__()

        self.n_heads = params.adapter_n_heads
        self.dim = params.adapter_dim
        self.head_dim = self.dim // self.n_heads
        # self.adapter_len = params.adapter_len

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, )
        self.wk = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, )
        self.wv = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, )
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, )

        self.freqs_cis = precompute_freqs_cis(self.dim // self.n_heads, params.max_seq_len)


    def forward(self, query, input_attn):
        _bsz, query_len, _ = query.shape
        freqs_cis = self.freqs_cis.to(query.device)

        position_id = torch.arange(query_len).repeat(_bsz, 1).to(query.device)
        position_id = position_id - ((input_attn == 0).sum(dim=-1)).unsqueeze(-1)
        position_id[position_id < 0] = 0
        freqs_cis = freqs_cis[position_id]


        xq, xk, xv = self.wq(query), self.wk(query), self.wv(query)

        xq = xq.view(_bsz, query_len, self.n_heads, self.head_dim)
        xk = xk.view(_bsz, query_len, self.n_heads, self.head_dim)
        xv = xv.view(_bsz, query_len, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)


        # Full Attention

        input_attn = input_attn.view(_bsz, 1, 1, query_len).repeat(1, 1, query_len, 1)
        input_attn = 1.0 - input_attn
        input_attn = input_attn.masked_fill(input_attn.to(torch.bool), torch.finfo(query.dtype).min).float()

        
        output = F.scaled_dot_product_attention(xq, xk, xv, input_attn)

        output = output.transpose(1, 2).contiguous().view(_bsz, query_len, -1)
        return self.wo(output)
    
    