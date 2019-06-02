import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

from dataset import IC15TestLoader
import models
import util

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt 

from pixel_link import decode_batch,mask_to_bboxes

def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print(idx, '/', len(img_paths), img_name)
    # plt.imshow(res)
    # plt.show()
    cv2.imwrite(output_root + img_name, res)

def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def to_bboxes(image_data, pixel_pos_scores, link_pos_scores):
    link_pos_scores=np.transpose(link_pos_scores,(0,2,3,1))    
    mask = decode_batch(pixel_pos_scores, link_pos_scores,0.6,0.9)[0, ...]
    bboxes = mask_to_bboxes(mask, image_data.shape)
    return mask,bboxes

def test(args):
    data_loader = IC15TestLoader(long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=18, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=18, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=18, scale=args.scale)
    elif args.arch == "vgg16":
        model = models.vgg16(pretrained=True,num_classes=18)
    
    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()
    
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print(("Loading model and optimizer from checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in list(checkpoint['state_dict'].items()):
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print(("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])))
            sys.stdout.flush()
        else:
            print(("No checkpoint found at '{}'".format(args.resume)))
            sys.stdout.flush()

    model.eval()
    
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img) in enumerate(test_loader):
        print(('progress: %d / %d'%(idx, len(test_loader))))
        sys.stdout.flush()

        img = img.cuda()
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        torch.cuda.synchronize()
        start = time.time()

        cls_logits,link_logits = model(img)

        outputs=torch.cat((cls_logits,link_logits),dim=1)
        shape=outputs.shape
        pixel_pos_scores=F.softmax(outputs[:,0:2,:,:],dim=1)[:,1,:,:]
        # pixel_pos_scores=torch.sigmoid(outputs[:,1,:,:])
        # FIXME the dimention should be changed
        link_scores=outputs[:,2:,:,:].view(shape[0],2,8,shape[2],shape[3])
        link_pos_scores=F.softmax(link_scores,dim=1)[:,1,:,:,:]

        mask,bboxes=to_bboxes(org_img,pixel_pos_scores.cpu().numpy(),link_pos_scores.cpu().numpy())

        score = pixel_pos_scores[0,:,:]
        score = score.data.cpu().numpy().astype(np.float32)   

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print(('fps: %.2f'%(total_frame / total_time)))
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, bboxes, 'outputs/submit_ic15/')
        
        text_box = cv2.resize(text_box, (org_img.shape[1], org_img.shape[0]))
        score_s = cv2.resize(np.repeat(score[:,:,np.newaxis]*255,3,2).astype(np.uint8), (org_img.shape[1], org_img.shape[0]))
        mask = cv2.resize(np.repeat(mask[:,:,np.newaxis],3,2).astype(np.uint8), (org_img.shape[1], org_img.shape[0]))
        
        link_score=(link_pos_scores[0,0,:,:]).cpu().numpy()*(score>0.5).astype(np.float)
        link_score = cv2.resize(np.repeat(link_score[:,:,np.newaxis]*255,3,2).astype(np.uint8), (org_img.shape[1], org_img.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box,score_s],[link_score,mask]], 'outputs/vis_ic15/')

    cmd = 'cd %s;zip -j %s %s/*'%('./outputs/', 'submit_ic15.zip', 'submit_ic15');
    print(cmd)
    sys.stdout.flush()
    util.cmd.cmd(cmd)
    cmd_eval='cd eval;sh eval_ic15.sh'
    sys.stdout.flush()
    util.cmd.cmd(cmd_eval)

if __name__ == '__main__':
    # import crash_on_ipy
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='vgg16')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=1280,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    
    args = parser.parse_args()
    test(args)
