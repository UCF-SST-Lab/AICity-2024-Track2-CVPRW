#################################### Training/Pretraining with BDD

python train.py --cfg_path cfgs/bdd_ped_clip_pdvcl.yml --gpu_id=0 --no_self_iou
#python train.py --cfg_path cfgs/bdd_veh_clip_pdvcl.yml --gpu_id=1 --no_self_iou

#################################### Finetuning with WTS
#python train.py --cfg_path cfgs/train_wts_veh_event_pdvcl_finetune.yml --gpu_id=2 --no_self_iou --load=save/bdd_veh_clip_pdvcl/ --load_vocab data/vocabulary/vocabulary_bdd_vehicle.json
#python train.py --cfg_path cfgs/train_wts_veh_normal_pdvcl_finetune.yml --gpu_id=3 --no_self_iou --load=save/bdd_veh_clip_pdvcl/ --load_vocab data/vocabulary/vocabulary_bdd_vehicle.json

#python train.py --cfg_path cfgs/train_wts_ped_event_pdvcl_finetune.yml --gpu_id=4 --no_self_iou --load=save/bdd_ped_clip_pdvcl/ --load_vocab data/vocabulary/vocabulary_bdd_pedestrian.json
#python train.py --cfg_path cfgs/train_wts_ped_normal_pdvcl_finetune.yml --gpu_id=5 --no_self_iou --load=save/bdd_ped_clip_pdvcl/ --load_vocab data/vocabulary/vocabulary_bdd_pedestrian.json



#################################### Evaluation
#python train.py --cfg_path cfgs/wts_veh_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_event_pdvcl_v_2024-03-21-22-53-28 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_vehicle_event_wc1.json

#python train.py --cfg_path cfgs/wts_ped_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_ped_event_pdvcl_v_2024-03-21-22-56-00 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_pedestrian_event_wc1.json

#python train.py --cfg_path cfgs/wts_veh_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_normal_pdvcl_v_2024-03-21-22-50-20 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_vehicle_normal_wc1.json

#python train.py --cfg_path cfgs/wts_ped_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_ped_normal_pdvcl_v_2024-03-21-22-55-15 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_pedestrian_normal_wc1.json

#python train.py --cfg_path cfgs/bdd_veh_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/wts_challenge_clip_pdvcl_v_2024-03-21-22-58-56 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/bdd_ped_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/wts_challenge_clip_pdvcl_v_2024-03-21-22-58-04 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json

################# submit-06
#python train.py --cfg_path cfgs/bdd_veh_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_veh_clip_pdvcl --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json
#python train.py --cfg_path cfgs/bdd_ped_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_ped_clip_pdvcl_v_2024-03-22-13-42-54 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json

#python train.py --cfg_path cfgs/bdd_ped_testing_validation.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/eval-ped-bdd-wts_challenge_clip_pdvcl_v_2024-03-20-18-08-52 --load_vocab /home/do868987/nlp_research/VidChapters/PDVC/data/vocabulary_bdd_wts.json
#python train.py --cfg_path cfgs/bdd_veh_testing_validation.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/eval-veh-bdd-wts_challenge_clip_pdvcl_v_2024-03-20-18-02-20 --load_vocab /home/do868987/nlp_research/VidChapters/PDVC/data/vocabulary_bdd_wts.json

################# submit-07
#python train.py --cfg_path cfgs/wts_veh_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_event_pdvcl_v_2024-03-23-14-28-43/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/wts_veh_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_normal_pdvcl_v_2024-03-23-14-29-01 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/wts_ped_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_ped_event_pdvcl_v_2024-03-23-15-27-29/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json

#python train.py --cfg_path cfgs/wts_ped_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_ped_normal_pdvcl_v_2024-03-23-15-30-20/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json


################# submit-08
#python train.py --cfg_path cfgs/bdd_veh_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_veh_clip_pdvcl_v_2024-03-22-18-05-59/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json
#python train.py --cfg_path cfgs/bdd_ped_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_ped_clip_pdvcl_v_2024-03-22-13-42-54 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json

#python train.py --cfg_path cfgs/bdd_ped_testing_validation.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/eval-ped-bdd-wts_challenge_clip_pdvcl_v_2024-03-20-18-08-52 --load_vocab /home/do868987/nlp_research/VidChapters/PDVC/data/vocabulary_bdd_wts.json
#python train.py --cfg_path cfgs/bdd_veh_testing_validation.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/eval-veh-bdd-wts_challenge_clip_pdvcl_v_2024-03-20-18-02-20 --load_vocab /home/do868987/nlp_research/VidChapters/PDVC/data/vocabulary_bdd_wts.json


################# submit-12
#python train.py --cfg_path cfgs/wts_veh_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_event_pdvcl_v_2024-03-25-13-04-15/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_vehicle_event_wc1.json

#python train.py --cfg_path cfgs/wts_veh_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_normal_pdvcl_v_2024-03-25-12-59-49 --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_vehicle_normal_wc1.json

#python train.py --cfg_path cfgs/wts_ped_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_ped_event_pdvcl_v_2024-03-25-12-59-32/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_pedestrian_event_wc1.json

#python train.py --cfg_path cfgs/wts_ped_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_ped_normal_pdvcl_v_2024-03-25-13-02-18/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_wts_pedestrian_normal_wc1.json


#python train.py --cfg_path cfgs/bdd_veh_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_veh_clip_pdvcl_v_2024-03-24-02-51-42/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/bdd_ped_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_ped_clip_pdvcl_v_2024-03-24-02-54-10/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json


python train.py --cfg_path cfgs/wts_veh_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_event_pdvcl_v_2024-03-25-19-51-12/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/wts_ped_event_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/wts_ped_event_eval_pdvcl_v_2024-03-25-20-15-27/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json

#python train.py --cfg_path cfgs/wts_veh_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_veh_normal_pdvcl_v_2024-03-25-20-00-14/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json


#python train.py --cfg_path cfgs/wts_ped_normal_eval_pdvcl.yml --gpu_id=4 --epoch=0 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/train_wts_ped_normal_pdvcl_v_2024-03-25-20-08-27/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json



#################################### Training
#python train.py --cfg_path cfgs/wts_challenge_clip_pdvcl.yml --gpu_id=3 --epoch=30 --no_self_iou --lr=1e-3
#python train.py --cfg_path cfgs/bdd_veh_clip_pdvcl.yml --gpu_id=3 --epoch=30 --no_self_iou 
#python train.py --cfg_path cfgs/train_wts_ped_normal_pdvcl.yml --gpu_id=3 --epoch=61 --no_self_iou
#python train.py --cfg_path cfgs/train_wts_ped_event_pdvcl.yml --gpu_id=4 --epoch=61 --no_self_iou
#python train.py --cfg_path cfgs/train_wts_veh_normal_pdvcl.yml --gpu_id=5 --epoch=61 --no_self_iou
#python train.py --cfg_path cfgs/train_wts_veh_event_pdvcl.yml --gpu_id=5 --epoch=31 --no_self_iou


#python train.py --cfg_path cfgs/train_wts_veh_normal_pdvcl.yml --gpu_id=5 --epoch=61 --no_self_iou
#python train.py --cfg_path cfgs/train_wts_veh_event_pdvcl.yml --gpu_id=4 --epoch=61 --no_self_iou
#python train.py --cfg_path cfgs/train_wts_ped_normal_pdvcl.yml --gpu_id=3 --epoch=61 --no_self_iou
#python train.py --cfg_path cfgs/train_wts_ped_event_pdvcl.yml --gpu_id=2 --epoch=101 --no_self_iou


#python train.py --cfg_path cfgs/bdd_veh_clip_pdvcl.yml --gpu_id=2 --epoch=101 --no_self_iou
#python train.py --cfg_path cfgs/bdd_ped_clip_pdvcl.yml --gpu_id=0 --epoch=101 --no_self_iou


#python train.py --cfg_path cfgs/train_wts_ped_event_pdvc_large.yml --gpu_id=3 --epoch=41 --no_self_iou --lr=1e-3
#python train.py --cfg_path cfgs/train_wts_veh_event_pdvc_large.yml --gpu_id=4 --epoch=41 --no_self_iou --lr=1e-3
#python train.py --cfg_path cfgs/train_wts_ped_normal_pdvc_large.yml --gpu_id=5 --epoch=31 --no_self_iou --lr=1e-3
#python train.py --cfg_path cfgs/train_wts_veh_normal_pdvc_large.yml --gpu_id=2 --epoch=31 --no_self_iou --lr=1e-3


#python train.py --cfg_path cfgs/bdd_veh_clip_pdvcl.yml --gpu_id=2 --epoch=101 --no_self_iou
#python train.py --cfg_path cfgs/bdd_ped_clip_pdvcl.yml --gpu_id=1 --epoch=101 --no_self_iou




#################################### Finetuning
#python train.py --cfg_path cfgs/train_wts_veh_event_pdvcl_finetune.yml --gpu_id=5 --epoch=30 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_veh_clip_pdvcl_v_2024-03-22-13-59-57/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json
#python train.py --cfg_path cfgs/train_wts_veh_normal_pdvcl_finetune.yml --gpu_id=4 --epoch=30 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_veh_clip_pdvcl_v_2024-03-22-13-59-57/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/train_wts_ped_event_pdvcl_finetune.yml --gpu_id=4 --epoch=30 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_ped_clip_pdvcl_v_2024-03-22-13-42-54/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json
#python train.py --cfg_path cfgs/train_wts_ped_normal_pdvcl_finetune.yml --gpu_id=5 --epoch=30 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_ped_clip_pdvcl_v_2024-03-22-13-42-54/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json




#python train.py --cfg_path cfgs/train_wts_veh_event_pdvcl_finetune.yml --gpu_id=1 --epoch=20 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_veh_eval_pdvcl_v_2024-03-25-18-19-20/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/train_wts_veh_normal_pdvcl_finetune.yml --gpu_id=6 --epoch=20 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_veh_eval_pdvcl_v_2024-03-25-18-19-20/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_vehicle_wc1.json

#python train.py --cfg_path cfgs/train_wts_ped_event_pdvcl_finetune.yml --gpu_id=7 --epoch=20 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_ped_clip_pdvcl_v_2024-03-24-02-54-10/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json

#python train.py --cfg_path cfgs/train_wts_ped_normal_pdvcl_finetune.yml --gpu_id=3 --epoch=20 --no_self_iou --load=/home/do868987/nlp_research/VidChapters/PDVC/save/bdd_ped_clip_pdvcl_v_2024-03-24-02-54-10/ --load_vocab /home/do868987/nlp_research/vocabulary/vocabulary_bdd_pedestrian_wc1.json
















