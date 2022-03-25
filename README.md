Code based on https://github.com/VegB/VLN-Transformer

Touchdown splits based on: https://github.com/lil-lab/touchdown

map2seq splits based on: https://map2seq.schumann.pub

Panorama images can be downloaded here: https://sites.google.com/view/streetlearn/dataset




# Workflow without Images 
(no need to download and preprocess panoramas)

## Preparation
```
pip install -r requirements.txt
```

## Test
```
python vln/main.py --test True --dataset map2seq_unseen --config outputs/map2seq_unseen/noimage/config/noimage.yaml --exp_name noimage --resume SPD_best
```

## Train from Scratch:
```
python vln/main.py --dataset touchdown_unseen --config configs/noimage.yaml --exp_name no_image
```






#
#
# Workflow with Images

## Preparation
```
pip install -r requirements.txt
```

image features dir should contain .pickle files named e.g. "pre-final.pickle"

pickle object format: dict: panoid: heading: numpy array (5x2048)

e.g. dict('HgFMRzAguxKiBHkwCQ_TgQ': dict(297: np(5x2048), 118: np(5x2048)), 'lvziEd_sT6RjF5Jpxo9_Fg': dict(151: np(5x2048), 208: np(5x2048), 28: np(5x2048)))

## Test
```
python vln/main.py --test True --dataset touchdown_seen --img_feat_dir 'path_to_features_dir' --config link_to_config --exp_name 4th-to-last --resume SPD_best
```


## Train from Scratch:
```
python vln/main.py --dataset touchdown_seen --img_feat_dir 'path_to_features_dir' --config configs/4th-to-last.yaml --exp_name 4th-to-last
```