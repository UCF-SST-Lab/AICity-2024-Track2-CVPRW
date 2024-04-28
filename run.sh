#################################### Training with BDD #################################### 

python train.py --cfg_path cfgs/bdd_ped_clip_pdvcl.yml --gpu_id=0 --no_self_iou
#python train.py --cfg_path cfgs/bdd_veh_clip_pdvcl.yml --gpu_id=1 --no_self_iou


#################################### Pretraining with BDD ################################# 

#python train.py --cfg_path cfgs/train_bdd_ped_clip_pdvcl_pretrain.yml --gpu_id=0 --no_self_iou
#python train.py --cfg_path cfgs/train_bdd_veh_clip_pdvcl_pretrain.yml --gpu_id=1 --no_self_iou

#################################### Finetuning with WTS ################################## 
####################### specify the model load path with saved checkpointpoint folder
#python train.py --cfg_path cfgs/train_wts_ped_event_pdvcl_finetune.yml --gpu_id=0 --no_self_iou --load=save/bdd_ped_clip_pdvcl_pretrain/ --load_vocab data/vocabulary/vocabulary_bdd_pedestrian.json
#python train.py --cfg_path cfgs/train_wts_ped_normal_pdvcl_finetune.yml --gpu_id=1 --no_self_iou --load=save/bdd_ped_clip_pdvcl_pretrain/ --load_vocab data/vocabulary/vocabulary_bdd_pedestrian.json

#python train.py --cfg_path cfgs/train_wts_veh_event_pdvcl_finetune.yml --gpu_id=0 --no_self_iou --load=save/bdd_veh_clip_pdvcl_pretrain/ --load_vocab data/vocabulary/vocabulary_bdd_vehicle.json
#python train.py --cfg_path cfgs/train_wts_veh_normal_pdvcl_finetune.yml --gpu_id=1 --no_self_iou --load=save/bdd_veh_clip_pdvcl_pretrain/ --load_vocab data/vocabulary/vocabulary_bdd_vehicle.json


##########################################################################################
#################################### Evaluation ########################################## 

#python train.py --cfg_path cfgs/wts_veh_event_eval_pdvcl.yml --gpu_id=0 --epoch=0 --no_self_iou --load=save/train_wts_veh_event_pdvcl --load_vocab data/vocabulary/vocabulary_bdd_vehicle.json
#python train.py --cfg_path cfgs/wts_ped_event_eval_pdvcl.yml --gpu_id=0 --epoch=0 --no_self_iou --load=save/train_wts_ped_event_pdvcl --load_vocab data/vocabulary/vocabulary_bdd_pedestrian.json
#python train.py --cfg_path cfgs/wts_veh_normal_eval_pdvcl.yml --gpu_id=0 --epoch=0 --no_self_iou --load=save/train_wts_veh_normal_pdvcl --load_vocab data/vocabulary/vocabulary_bdd_vehicle.json
#python train.py --cfg_path cfgs/wts_ped_normal_eval_pdvcl.yml --gpu_id=0 --epoch=0 --no_self_iou --load=save/train_wts_ped_normal_pdvcl --load_vocab data/vocabulary/vocabulary_bdd_pedestrian.json

#python train.py --cfg_path cfgs/bdd_veh_eval_pdvcl.yml --gpu_id=0 --epoch=0 --no_self_iou --load=save/bdd_veh_clip_pdvcl --load_vocab data/vocabulary/vocabulary_bdd_vehicle.json
#python train.py --cfg_path cfgs/bdd_ped_eval_pdvcl.yml --gpu_id=0 --epoch=0 --no_self_iou --load=save/bdd_ped_clip_pdvcl --load_vocab data/vocabulary/vocabulary_bdd_pedestrian.json








