
############ 

# Physics-informed Loss

# python train.py --config 'configs/experiment/base_config.yaml' --model 'NodeEdgeGNN'
# python train.py --config 'configs/experiment/global_loss_config.yaml' --model 'NodeEdgeGNN'
# python train.py --config 'configs/experiment/local_loss_config.yaml' --model 'NodeEdgeGNN'

python test.py --config 'configs/experiment/base_config.yaml' --model 'NodeEdgeGNN' --model_path 'physics_informed_stc\model\NodeEdgeGNN_2025-08-02_01-03-06_base.pt'
python test.py --config 'configs/experiment/global_loss_config.yaml' --model 'NodeEdgeGNN' --model_path 'physics_informed_stc\model\NodeEdgeGNN_2025-08-02_02-48-18_global.pt'
python test.py --config 'configs/experiment/local_loss_config.yaml' --model 'NodeEdgeGNN' --model_path 'physics_informed_stc\model\NodeEdgeGNN_2025-08-01_15-47-37_local.pt'

#############

# python train.py --config 'configs\experiment_attn\config_1200_attn.yaml' --model 'NodeEdgeGNNAttn'

# python test_normal_edge.py --config 'configs\experiment_attn\config_1200_attn.yaml' --model 'NodeEdgeGNNAttn' --model_path 'attention\model\NodeEdgeGNNAttn_2025-08-01_01-59-47.pt'
