# Enhancing Traffic Safety with Parallel Dense Video Captioning for End-to-End Event Analysis

This repository contains code to reproduce the results for our paper Enhancing Traffic Safety with Parallel Dense Video Captioning for End-to-End Event Analysis (CVPRW 2024) 


**Table of Contents:**
* [Preparation](#preparation)
* [Training and Validation](#training-and-validation)
  + [Download Video Features](#download-video-features)
  + [Dense Video Captioning](#dense-video-captioning)
  + [Video Paragraph Captioning](#video-paragraph-captioning)
* [Performance](#performance)
  + [Dense video captioning](#dense-video-captioning)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)



## Preparation

1. Clone the repo
```bash
git clone --recursive https://github.com/UCF-SST-Lab/AICity2024CVPRW.git
```

2. Create vitual environment by conda
```bash
conda create -n PDVC python=3.7
source activate PDVC
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install ffmpeg
pip install -r requirement.txt
```

3. Compile the deformable attention layer (requires GCC >= 5.4). 
```bash
cd pdvc/ops
sh make.sh
```

## Training Dense Video Captioning

```
# Training
config_path=cfgs/bdd_veh_clip_pdvcl.yml
python train.py --cfg_path ${config_path} --gpu_id ${GPU_ID} --epoch=30
# The script will evaluate the model for every epoch. The results and logs are saved in `./save`.

# Evaluation
eval_folder=bdd_eval # specify the folder to be evaluated
python eval.py --eval_folder ${eval_folder} --eval_transformer_input_type queries --gpu_id ${GPU_ID}
```

## Performance
### Dense video captioning (with learnt proposals)

|  Model | Features | config_path |   Url   |   BLEU4   | METEOR | ROUGE-L |  CIDEr |
|  ----  |  ----    |   ----  |  ----  |  ----   |  ----  |  ----  |  ---- |
| PDVC_light   | CLIP  | cfgs/anet_c3d_pdvcl.yml |  


## Citation
If you find this repo helpful, please consider citing:
```
@article{shoman2024enhancing,
  title={Enhancing Traffic Safety with Parallel Dense Video Captioning for End-to-End Event Analysis},
  author={Shoman, Maged and Wang, Dongdong and Aboah, Armstrong and Abdel-Aty, Mohamed},
  journal={arXiv preprint arXiv:2404.08229},
  year={2024}
}
```
```
@article{wang20248th,
  title={The 8th AI City Challenge},
  author={Wang, Shuo and Anastasiu, David C and Tang, Zheng and Chang, Ming-Ching and Yao, Yue and Zheng, Liang and Rahman, Mohammed Shaiqur and Arya, Meenakshi S and Sharma, Anuj and Chakraborty, Pranamesh and others},
  journal={arXiv preprint arXiv:2404.09432},
  year={2024}
}
```

## Acknowledgement

The implementation of PDVC is modified based on [PDVC](https://github.com/ttengwang/PDVC). <br>
The implementation of video Feature extraction is modified based on [FrozenBiLM](https://github.com/antoyang/FrozenBiLM). <br>
The implementation of Deformable Transformer is mainly based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). <br>
The implementation of the captioning head is based on [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch).
We thanks the authors for their efforts.
