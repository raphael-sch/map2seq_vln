Code for ACL 2022 paper:  [Schumann and Riezler, "Analyzing Generalization of Vision and Language Navigation to Unseen Outdoor Areas "](https://aclanthology.org/2022.acl-long.518.pdf)

## Results for Model Weights in this Repository

| Model                             |  TC   |  SPD  |  SED  |  TC   |  SPD  |  SED  |
|-----------------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|                                   |  dev  |  dev  |  dev  | test  | test  | test  |
| touchdown seen:                   |       |       |       |       |       |       |
| - no image                        | 15.38 | 25.66 | 15.03 | 13.13 | 27.52 | 12.58 |
| - 4th-to-last                     | 30.05 | 11.12 | 29.46 | 29.60 | 11.79 | 28.89 |
| - 4th-to-last no head. & no junc. | 24.08 | 13.63 | 23.48 | 24.49 | 14.19 | 23.98 |
| touchdown unseen:                 |       |       |       |       |       |       |
| - no image                        | 11.50 | 25.89 | 10.72 | 9.62  | 27.71 | 9.04  |
| - 4th-to-last                     | 16.88 | 19.59 | 16.21 | 15.06 | 20.52 | 14.32 |
| map2seq seen:                     |       |       |       |       |       |       |
| - no image                        | 42.25 | 8.02  | 41.52 | 39.25 | 7.90  | 38.42 |
| - pre-final                       | 49.88 | 5.87  | 48.96 | 47.75 | 6.53  | 46.76 |
| - pre-final no head. & no junc.   | 49.62 | 7.36  | 48.80 | 44.88 | 8.79  | 44.21 |
| map2seq unseen:                   |       |       |       |       |       |       |
| - no image                        | 27.88 | 10.71 | 27.05 | 30.25 | 11.52 | 29.02 |
| - 4th-to-last                     | 27.62 | 11.81 | 26.96 | 29.62 | 13.16 | 28.91 |
| merged seen:                      |       |       |       |       |       |       |
| - no image                        | 27.80 | 18.48 | 27.15 | 24.63 | 19.51 | 23.91 |
| - 4th-to-last                     | 38.20 | 9.18  | 37.42 | 36.22 | 9.55  | 35.42 |
| merged unseen:                    |       |       |       |       |       |       |
| - no image                        | 22.19 | 17.85 | 21.46 | 19.85 | 21.20 | 19.10 |
| - 4th-to-last                     | 25.31 | 15.17 | 24.24 | 24.10 | 16.48 | 23.46 |

## Workflow without Images 
(no need to download and preprocess panoramas)

### Preparation
```
pip install -r requirements.txt
```

### Inference and Evaluation
```
python vln/main.py --test True --dataset map2seq_unseen --config outputs/map2seq_unseen/noimage/config/noimage.yaml --exp_name noimage --resume SPD_best
```

### Train from Scratch:
```
python vln/main.py --dataset touchdown_unseen --config configs/noimage.yaml --exp_name no_image
```






#
## Workflow with Images

### Panorama Preprocessing
Unfortunately we are not allowed to share the panorama images or the ResNet features derived from them. You have to request to download the images here: https://sites.google.com/view/streetlearn/dataset  
Then change into the `panorama_preprocessing/last_layer` or `panorama_preprocessing/fourth_layer` folder and use the `extract_features.py` script. 

### Preparation
```
pip install -r requirements.txt
```

### Test
```
python vln/main.py --test True --dataset touchdown_seen --img_feat_dir 'path_to_features_dir' --config link_to_config --exp_name 4th-to-last --resume SPD_best
```

The `path_to_features_dir` should contain the `resnet_fourth_layer.pickle` and `resnet_last_layer.pickle` file created in the pano preprocessing step.


### Train from Scratch:
```
python vln/main.py --dataset touchdown_seen --img_feat_dir 'path_to_features_dir' --config configs/4th-to-last.yaml --exp_name 4th-to-last
```

References
=========

Code based on https://github.com/VegB/VLN-Transformer  
Touchdown splits based on: https://github.com/lil-lab/touchdown  
map2seq splits based on: https://map2seq.schumann.pub  
Panorama images can be downloaded here: https://sites.google.com/view/streetlearn/dataset

Citation
=========
Please cite the following paper if you use this code:

```
@inproceedings{schumann-riezler-2022-analyzing,
    title = "Analyzing Generalization of Vision and Language Navigation to Unseen Outdoor Areas",
    author = "Schumann, Raphael and Riezler, Stefan",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2022",
    address = "Dublin, Ireland",
    pages = "7519--7532"
}
```