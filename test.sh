# python convert_mrnet_to_nii.py --mrnet_file "data/npy_mrnet_example/0001/sag_org.npy" --ref_file "data/nifti_inhouse_example/00001/sag_org.nii.gz" --view "sag" --save_path "data/nifti_mrnet_example/0001_sag/sag_org.nii.gz"

# python construct_graph.py --subject_folder "data/nifti_mrnet_example/0001_sag" --bone_index "1,2,3" --main_view "sag" --save_path "data/graph_mrnet_example/0001/sag.npz"

# python construct_graph.py --subject_folder "data/nifti_inhouse_example/00001" --bone_index "1,4,6" --main_view "sag" --save_path "data/graph_inhouse_example/00001.npz"

# CUDA_VISIBLE_DEVICES=0 python pretrain.py --fold 0 --config_file config/pretrain_mrnet_sag.yaml --dataset mrnet
# CUDA_VISIBLE_DEVICES=0 python pretrain.py --fold 0 --config_file config/pretrain_inhouse.yaml --dataset inhouse 
CUDA_VISIBLE_DEVICES=0 python finetune.py --fold 0 --config_file config/finetune_mrnet.yaml --dataset mrnet 
# CUDA_VISIBLE_DEVICES=0 python finetune.py --fold 0 --config_file config/finetune_inhouse.yaml --dataset inhouse 