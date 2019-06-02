# dataloader add 3.0 scale
# dataloader add filer text
import random
import sys
# sys.path.extend('/workspace/lupu/pixellink.pytorch')

import cv2
import numpy as np
import Polygon as plg
import pyclipper
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

import util
from pixel_link import cal_gt_for_single_image

from .transfrom import build_transfrom

ic15_root_dir = './data/icdar2015/'
ic15_train_data_dir = ic15_root_dir + 'train_images/'
ic15_train_gt_dir = ic15_root_dir + 'train_gts/'
ic15_test_data_dir = ic15_root_dir + 'ch4_test_images/'
ic15_test_gt_dir = ic15_root_dir + 'ch4_test_images_gts/'

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img

def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        line = util.str.remove_all(line, '\ufeff')
        gt = util.str.split(line, ',')
        if gt[-1][0] == '#':
            tags.append(-1)
        else:
            tags.append(1)
        box = [int(gt[i]) for i in range(8)]
        # box = np.asarray(box) / ([w * 1.0, h * 1.0] * 4)
        box = np.asarray(box)
        bboxes.append(box)
    return np.array(bboxes), np.array(tags)

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs
    
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    
    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox)[0]
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

class IC15Loader(data.Dataset):
    def __init__(self, is_transform=False, img_size=None):
        self.is_transform = is_transform
        
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)

        data_dirs = [ic15_train_data_dir]
        gt_dirs = [ic15_train_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        self.transfrom=build_transfrom()
        
        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
                
                gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)
        
        if self.is_transform:
            # img = random_scale(img, self.img_size[0])
            # img = cv2.resize(img, (self.img_size[0],self.img_size[1]))
            # TODO use transfrom here
            bboxes=np.reshape(bboxes,(-1,4,2)).astype(np.float)
            img,bboxes,tags=self.transfrom(img,bboxes,tags)
        
        # show image
    
        h,w,_=img.shape
        bboxes[:,:,0]/=w
        bboxes[:,:,1]/=h
        pixel_cls_label, pixel_cls_weight, \
        pixel_link_label, pixel_link_weight = cal_gt_for_single_image(bboxes[:,:,0], bboxes[:,:,1], tags)

        # '''
        if self.is_transform:
            img = Image.fromarray(img.astype(np.uint8))
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        cls_label = torch.from_numpy(pixel_cls_label).float()
        cls_weight = torch.from_numpy(pixel_cls_weight)
        link_label = torch.from_numpy(pixel_link_label).float()
        link_weight = torch.from_numpy(pixel_link_weight)

        return img, cls_label,  cls_weight,  link_label,  link_weight


def test():
    import matplotlib
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    dataloader=IC15Loader(is_transform=True)
    
    for i in range(100):
        img, cls_label,  cls_weight,  link_label,  link_weight=dataloader.__getitem__(i)

        ax1=plt.subplot(2,2,1)
        ax1.imshow(np.transpose((img.numpy()*255).astype(np.uint8),(1,2,0)))
        ax2=plt.subplot(2,2,2)
        ax2.imshow(cls_label.numpy())
        ax3=plt.subplot(2,2,3)
        ax3.imshow(cls_weight.numpy())

        plt.show()      