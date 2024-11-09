# CUDA_VISIBLE_DEVICES=0 python pretrain.py --fold 0 --config_file config/pretrain_mrnet_sag.yaml --dataset mrnet
# CUDA_VISIBLE_DEVICES=1 python pretrain.py --fold 0 --config_file config/pretrain_inhouse.yaml --dataset inhouse 
# CUDA_VISIBLE_DEVICES=0 python finetune.py --fold 0 --config_file config/finetune_mrnet.yaml --dataset mrnet --ckpt results/checkpoints/pretrain_mrnet/KGNet/0-2024-11-09-20-58-28-best.pth
CUDA_VISIBLE_DEVICES=1 python finetune.py --fold 0 --config_file config/finetune_inhouse.yaml --dataset inhouse --ckpt results/checkpoints/pretrain_inhouse/KGNet/0-2024-11-09-21-05-21-best.pth
