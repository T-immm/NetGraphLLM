import pickle
from config import module_path
import torch
from transformers import LlamaTokenizer
from torch.utils.data import TensorDataset
import datasets


def load_pretrain_config():
    dataset = datasets.load_from_disk(f"{module_path}/dataset/pretrain")
    # split = pickle.load(open(f"{module_path}/dataset/config/split_20000.pkl", 'rb'))
    # split = pickle.load(open(f"{module_path}/dataset/config/split_temp.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/pretrain/edge_pretrain.pkl", 'rb'))
    
    edge_attr = pickle.load(open(f"{module_path}/dataset/pretrain/edge_attr.pkl", 'rb'))
    
    return dataset, edge_index, edge_attr


def load_textual_config():
    dataset = datasets.load_from_disk(f"{module_path}/dataset/config")
    split = pickle.load(open(f"{module_path}/dataset/config/split.pkl", 'rb'))
    # split = pickle.load(open(f"{module_path}/dataset/config/split_temp.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/config/edge_index.pkl", 'rb'))
    edge_attr = pickle.load(open(f"{module_path}/dataset/config/edge_attr.pkl", 'rb'))
    return dataset, split, edge_index, edge_attr


def load_update_config():
    dataset = datasets.load_from_disk(f"{module_path}/dataset/update_config")
    # split = pickle.load(open(f"{module_path}/dataset/update_config_mini/split_1580.pkl", 'rb'))
    split = pickle.load(open(f"{module_path}/dataset/update_config/split.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/update_config/edge_index.pkl", 'rb'))
    edge_attr = pickle.load(open(f"{module_path}/dataset/update_config/edge_attr.pkl", 'rb'))
    return dataset, split, edge_index, edge_attr



