from __future__ import print_function
from pprint import pprint
import numpy as np
import util
import pixel_link
# slim = tf.contrib.slim

#=====================================================================
#====================Pre-processing params START======================
# VGG mean parameters.
r_mean = 123.
g_mean = 117.
b_mean = 104.
rgb_mean = [r_mean, g_mean, b_mean]

# scale, crop, filtering and resize parameters
use_rotation = True
rotation_prob = 0.5
max_expand_scale = 1
expand_prob = 0
min_object_covered = 0.1          # Minimum object to be cropped in random crop.
bbox_crop_overlap = 0.2         # Minimum overlap to keep a bbox after cropping.
crop_aspect_ratio_range = (0.5, 2.)  # Distortion ratio during cropping.
area_range = [0.1, 1]
flip = False
using_shorter_side_filtering=True
min_shorter_side = 10
max_shorter_side = np.infty
#====================Pre-processing params END========================
#=====================================================================


#=====================================================================
#====================Post-processing params START=====================
decode_method = pixel_link.DECODE_METHOD_join
min_area = 300
min_height = 10
#====================Post-processing params END=======================
#=====================================================================



#=====================================================================
#====================Training and model params START =================
dropout_ratio = 0
max_neg_pos_ratio = 3


pixel_neighbour_type = pixel_link.PIXEL_NEIGHBOUR_TYPE_8
#pixel_neighbour_type = pixel_link.PIXEL_NEIGHBOUR_TYPE_4


feat_layers = ['conv3_3', 'conv4_3', 'conv5_3', 'fc7']
strides = [1]

pixel_cls_weight_method = pixel_link.PIXEL_CLS_WEIGHT_bbox_balanced
bbox_border_width = 1
pixel_cls_border_weight_lambda = 1.0
pixel_cls_loss_weight_lambda = 2.0
pixel_link_neg_loss_weight_lambda = 1.0
pixel_link_loss_weight = 1.0
#====================Training and model params END ==================
#=====================================================================


#=====================================================================
#====================do-not-change configurations START===============
num_classes = 2
ignore_label = -1
background_label = 0
text_label = 1
data_format = 'NHWC'
train_with_ignored = False
#====================do-not-change configurations END=================
#=====================================================================

global weight_decay

global train_image_shape
global image_shape
global score_map_shape

global batch_size
global batch_size_per_gpu
global gpus
global num_clones
global clone_scopes

global num_neighbours

global pixel_conf_threshold
global link_conf_threshold

def _set_weight_decay(wd):
    global weight_decay
    weight_decay = wd

def _set_image_shape(shape):
    h, w = shape
    global train_image_shape
    global score_map_shape
    global image_shape
    
    assert w % 4 == 0
    assert h % 4 == 0
    
    train_image_shape = [h, w]
    score_map_shape = (int(h / strides[0]), int(w / strides[0]))
    image_shape = train_image_shape

def _set_batch_size(bz):
    global batch_size
    batch_size = bz

def _set_seg_th(pixel_conf_th, link_conf_th):
    global pixel_conf_threshold
    global link_conf_threshold
    
    pixel_conf_threshold = pixel_conf_th
    link_conf_threshold = link_conf_th
    
    
def  _set_train_with_ignored(train_with_ignored_):
    global train_with_ignored    
    train_with_ignored = train_with_ignored_

    
def init_config(image_shape, batch_size = 1, 
                weight_decay = 0.0005, 
                num_gpus = 1, 
                pixel_conf_threshold = 0.6,
                link_conf_threshold = 0.9):
    _set_seg_th(pixel_conf_threshold, link_conf_threshold)
    _set_weight_decay(weight_decay)
    _set_image_shape(image_shape)

    global num_neighbours
    num_neighbours = pixel_link.get_neighbours_fn()[1]

init_config((320, 320), 4,
            pixel_conf_threshold=0.8,
            link_conf_threshold=0.8,)
# init_config((640, 640), 4)
