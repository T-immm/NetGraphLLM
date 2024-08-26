# nohup accelerate launch --config_file accelerate_config.yaml model_pretrain.py --num_epochs 1 --dataset pretrain
nohup accelerate launch --config_file accelerate_config.yaml model_train.py --num_epochs 4 --dataset network_understand
# nohup accelerate launch --config_file accelerate_config.yaml model_train.py --num_epochs 4 --dataset config_update
