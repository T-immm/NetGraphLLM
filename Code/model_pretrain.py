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

        
        
        
        dataset, edge_index, edge_info = load_dataset[args.dataset]()
        # print(dataset[0])
        # task_list = dataset["task"]
        # print(len(task_list))
        
        
        
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

        




    accelerator.wait_for_everyone()

    # Step 2: Build Node Classification Dataset
    
    train_loader = torch.utils.data.DataLoader(original_dataset, batch_size=args.batch_size, drop_last=True,
                                               pin_memory=True, shuffle=True, collate_fn=default_data_collator)
    
    # print(train_loader.dataset)

    
    
    
    
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
                                          
                                          edge_info = edge_info, # 网络动态信息
                                          
                                          node_configuration_ids=original_dataset['input_ids'], # 节点嵌入式表示的文本信息
                                          input_attention_mask=original_dataset['attention_mask'],
                                          )
    torch.set_default_tensor_type(torch.FloatTensor)


    ckpt = Path(model_path + "/consolidated.00.pth")
    ckpt = torch.load(ckpt, map_location='cpu')
    base_model.load_state_dict(ckpt, strict=False)
    
    

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

    model, train_loader, optimizer = accelerator.prepare(base_model, train_loader, optimizer)

    print(model)
    # print("train_loader------{}".format(train_loader))
    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss, best_val_acc = float('inf'), -float('inf')

    
    
    count_train = 0
    
    
    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            count_train = count_train + 1
            # print('batch------------------------------------')
            # print(batch)
            
            
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss, train_acc = model(**batch)
                
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                accelerator.clip_grad_norm_(optimizer.param_groups[1]['params'], 0.1)

                if (step + 1) % args.grad_steps == 0:
                    adjust_learning_rate(optimizer.param_groups[0], lr_group['adapter'], step / len(train_loader) + epoch,
                                         args)
                    adjust_learning_rate(optimizer.param_groups[1], lr_group['lora'], step / len(train_loader) + epoch,
                                         args)

                optimizer.step()
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            
            
            if (step + 1) % args.grad_steps == 0:
                adapter_lr = optimizer.param_groups[0]["lr"]
                lora_lr = optimizer.param_groups[1]["lr"]

                accelerator.log({'Adapter Lr': adapter_lr, 'Lora Lr': lora_lr})
                accelerator.log({'Accum Loss': accum_loss / args.grad_steps})
                accelerator.print(f"Accum Loss: {accum_loss / args.grad_steps}")
                
                accum_loss = 0.

            progress_bar.update(1)
            

        accelerator.print(
            f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        accelerator.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

    # 保存模型参数
    torch.save(model.state_dict(), 'model_state.pth')
    
    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    accelerator.wait_for_everyone()
    


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