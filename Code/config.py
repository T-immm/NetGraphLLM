from os import path
import argparse

module_path = path.dirname(path.abspath(__file__))

from dataset_process.dataset import *
from dataset_process.dataset_preprocess import *


load_dataset = {
    'pretrain':load_pretrain_config,
    'network_understand': load_textual_config,
    'config_update': load_update_config,
}

preprocess_original_dataset = {
    'pretrain': preprocess_function_pretrain_config,
    'network_understand': preprocess_function_original_understand,
    'config_update': preprocess_function_original_update
}

preprocess_train_dataset = {
    'network_understand': preprocess_function_generator_network_understand,
    'config_update': preprocess_function_generator_config_update,
}

preprocess_test_dataset = {
    'network_understand': preprocess_test_function_generator_network_understand,
    'config_update': preprocess_test_function_generator_config_update,
}

instruction_len = {
    'pretrain': 50,
    'network_understand': 50,
    'config_update': 320,
}

original_len = {
    'pretrain': 512,
    'network_understand': 1200,
    'config_update': 512,
}

task_level = {
    'pretrain': 'graph',
    'network_understand': 'pair',
    'config_update': 'pair',
}


def parse_args_llama():
    parser = argparse.ArgumentParser(description="XXXX")
    parser.add_argument("--project", type=str, default="AAAA")
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument("--model_name", type=str, default='LLaMA-7B-2')
    parser.add_argument("--dataset", type=str, default='config_path')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--adapter_len", type=int, default=1)
    parser.add_argument("--adapter_dim", type=int, default=768)
    parser.add_argument("--adapter_n_heads", type=int, default=6)
    parser.add_argument("--n_decoder_layers", type=int, default=4)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_mp_layers", type=int, default=4)

    # Model Training
    # parser.add_argument("--batch_size", type=int, default=16)
    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_steps", type=int, default=2)
    
    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # RRWP
    parser.add_argument("--rrwp", type=int, default=8)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=32)

    args = parser.parse_args()
    return args




