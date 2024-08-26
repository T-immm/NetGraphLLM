import json
import pathlib
import pickle
import transformers
import torch
import os
import copy
import wandb
import gc
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from config import *
from transformers import default_data_collator
from accelerate import DistributedDataParallelKwargs

import os
from dataset_process import *
from model.model import Transformer, ModelArgs
# from model.model_without_gnn import Transformer, ModelArgs
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs


model_path = "/home/jzt/models/Llama-2-7B"

torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

wandb.init(mode="dryrun")

def main(args, SEED):
    group = f"{args.dataset}"
    accelerator.init_trackers(project_name=f"{args.project}",
                              init_kwargs={"wandb":
                                               {"tags": [args.dataset, args.model_name],
                                                "group": group,
                                                "name": f"{args.dataset}_EXP{SEED}",
                                                "config": args}
                                           },
                              )


    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    
    
    seed_everything(seed=SEED)
    accelerator.print(args)


    with accelerator.main_process_first():
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = 'left'
        
        dataset, split, edge_index, edge_attr = load_dataset[args.dataset]()
        
        # 通过llama分词器将节点文本信息的desc转化为嵌入表示
        # 节点文本信息
        original_dataset = dataset.map(
            preprocess_original_dataset[args.dataset](tokenizer=tokenizer, max_length=original_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")

        # 训练集 包含prompt
        clm_dataset_train = dataset.map(
            preprocess_train_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")

        # 测试集 包含prompt，label信息 
        clm_dataset_test = dataset.map(
            preprocess_test_dataset[args.dataset](tokenizer=tokenizer, max_length=instruction_len[args.dataset]),
            batched=True,
            batch_size=None,
            remove_columns=[i for i in dataset.column_names if i not in ['node_ids', 'label', 'text_label']],
            keep_in_memory=True,
            writer_batch_size=10000,
            num_proc=1,
        ).with_format("torch")





    accelerator.wait_for_everyone()

    train_dataset = clm_dataset_train.select(split['train'])
    val_dataset = clm_dataset_train.select(split['valid'])
    val_dataset_eval = clm_dataset_test.select(split['valid'])
    
    test_dataset = clm_dataset_train.select(split['test'])
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               pin_memory=True, shuffle=True, collate_fn=default_data_collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True,
                                             shuffle=False, collate_fn=default_data_collator)
    val_loader_eval = torch.utils.data.DataLoader(val_dataset_eval, batch_size=args.batch_size, drop_last=False,
                                                  pin_memory=True, shuffle=False, collate_fn=default_data_collator)
    
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False,
                                              pin_memory=True, shuffle=False, collate_fn=default_data_collator)
    
    with open(model_path + "/params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(# w_lora=False,
                                      # w_adapter=True,
                                      # adapter_layer = 10,
                                      adapter_dim=args.adapter_dim,
                                      # adapter_len=args.adapter_len,
                                      lora_alpha=16,
                                      lora_r=8,
                                      num_hops=3,
                                      n_mp_layers=args.n_mp_layers,
                                      rrwp=args.rrwp,
                                      n_encoder_layers=args.n_encoder_layers,
                                      # n_decoder_layers=args.n_decoder_layers,
                                      adapter_n_heads=args.adapter_n_heads,
                                      task_level=task_level[args.dataset],
                                      **params)


    model_args.vocab_size = tokenizer.vocab_size
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    
    # torch.distributed.init_process_group()
    
    # 输入模型 node_configuration_ids
    base_model: Transformer = Transformer(params=model_args, edge_index=edge_index, # 边连接关系输入
                                          edge_attr=edge_attr,
                                          node_configuration_ids=original_dataset['input_ids'], # 节点嵌入式表示的文本信息
                                          input_attention_mask=original_dataset['attention_mask'],
                                          )
    torch.set_default_tensor_type(torch.FloatTensor)


    # ckpt = Path(model_path + "/consolidated.00.pth")
    # ckpt = torch.load(ckpt, map_location='cpu')
    # base_model.load_state_dict(ckpt, strict=False)
    device = torch.device('cuda')
    # base_model.load_state_dict(torch.load('/home/jzt/Temp/result/forward_path/gnn/model_state.pth'))
    base_model.load_state_dict(torch.load('model_state.pth'))
    base_model.to(device)

    accelerator.print(model_args)

    # Step 4 Set Optimizer
    param_adapter, param_lora = base_model.set_trainable_params_new()


    lr_group = {
        'adapter': args.lr,
        'lora': args.lr,
    }

    wd_group = {
        'adapter': args.wd,
        'lora': args.wd,
    }

    accelerator.print(lr_group)
    accelerator.print(wd_group)

    optimizer = torch.optim.AdamW(
        [
            {'params': param_adapter, 'lr': lr_group['adapter'], 'weight_decay': wd_group['adapter']},
            {'params': param_lora, 'lr': lr_group['lora'], 'weight_decay': wd_group['lora']},
        ],
        betas=(0.9, 0.95))

    trainable_params, all_param = base_model.print_trainable_params()
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    model, train_loader, val_loader, val_loader_eval, optimizer = accelerator.prepare(base_model, train_loader,
                                                                                      val_loader, val_loader_eval,
                                                                                      optimizer)

    print(model)

    # Step 5. Evaluating
    model, test_loader = accelerator.prepare(model, test_loader)

    model.eval()
    val_loss = 0.
    total_acc = 0

    with torch.no_grad():
            print("TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            print("Len--------------------{}".format(len(test_loader)))
            for step, batch in enumerate(test_loader):
                loss, temp_acc = model(**batch)
                val_loss += loss.item()
                total_acc += temp_acc
                
            accelerator.print(f"Test Loss: {val_loss / len(test_loader)}")
            accelerator.print(f"Test Acc: {total_acc / len(test_loader)}")
            print("TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")


if __name__ == "__main__":

    args = parse_args_llama()
    for exp, SEED in enumerate(range(args.exp_num)):
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        transformers.logging.set_verbosity_error()
        accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs, init_kwargs],
                                  gradient_accumulation_steps=args.grad_steps)

        main(args, SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()