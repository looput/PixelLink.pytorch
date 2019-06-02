# PixelLink

This code is based on PixlLink and PseNet, the performance is not satisfactory

## Requirements
* Python 3.6
* PyTorch v0.4.1+
* opencv-python 3.4

## Introduction
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ic15.py
```

## Testing
```
CUDA_VISIBLE_DEVICES=0 python test_ic15.py --scale 1 --resume [path of model]
```

## Eval script for ICDAR 2015
```
cd eval
sh eval_ic15.sh
```


## Performance
| Dataset | Pretrained | Precision (%) | Recall (%) | F-measure (%) | FPS (1080Ti) | Input |
| - | - | - | - | - | - | - |
| ICDAR2015 | No | 81.2 | 75 | 78 | 5 | 1280*768 |

## TODO
- [ ] Find the bug of low performance 
- [ ] Accomplish the code with better config file and more datasets