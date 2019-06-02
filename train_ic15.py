import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil
from tensorboardX import SummaryWriter
import torchvision

from torch.autograd import Variable
from torch.utils import data
import os

from dataset import IC15Loader
from metrics import runningScore
import models
from util import Logger, AverageMeter
import time
import util

def ohem_single(score, n_pos, neg_mask):
    if n_pos == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = neg_mask
        return selected_mask
    
    neg_num = neg_mask.view(-1).sum()
    neg_num = (min(n_pos * 3, neg_num)).to(torch.int)
    
    if neg_num == 0:
        selected_mask = neg_mask
        return selected_mask

    neg_score=torch.masked_select(score,neg_mask)*-1
    value,_=neg_score.topk(neg_num)
    threshold=value[-1]

    selected_mask= neg_mask*(score<=-threshold)
    return selected_mask

def ohem_batch(neg_conf, pos_mask, neg_mask):
    selected_masks = []
    for img_neg_conf,img_pos_mask,img_neg_mask in zip(neg_conf,pos_mask,neg_mask):
        n_pos=img_pos_mask.view(-1).sum()
        selected_masks.append(ohem_single(img_neg_conf, n_pos, img_neg_mask))

    selected_masks = torch.stack(selected_masks, 0).to(torch.float)

    return selected_masks

def dice_loss(input, target, mask):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    mask = mask.contiguous().view(mask.size()[0], -1)
    
    input = input * mask
    target = target * mask

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text >  0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def train(train_loader, model, criterion, optimizer, epoch,writer=None):
    import config 
    cls_loss_lambda=config.pixel_cls_loss_weight_lambda
    link_loss_lambda=config.pixel_link_loss_weight

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_cls = AverageMeter()
    running_metric_text = runningScore(2)

    device=torch.device('cuda:0')
    end = time.time()
    for batch_idx, (imgs,cls_label, cls_weight, link_label, link_weight) in enumerate(train_loader):
        data_time.update(time.time() - end)

        imgs=imgs.to(device)
        cls_label, cls_weight, link_label, link_weight = cls_label.to(
            device), cls_weight.to(device), link_label.to(device), link_weight.to(device)
        
        link_label = link_label.transpose(2,3).transpose(1,2) # [b, 8, h, w]
        link_weight = link_weight.transpose(2,3).transpose(1,2) # [b, 8, h, w]

        # outputs=model(imgs)

        # pixel_cls_logits = outputs[:, 0:2, :, :]
        # pixel_link_logits = outputs[:, 2:, :, :]

        pixel_cls_logits,pixel_link_logits=model(imgs)

        pos_mask=(cls_label>0)
        neg_mask=(cls_label==0)

        # train_mask=pos_mask+neg_mask
        # pos_logits=pixel_cls_logits[:,1,:,:]

        # dice_loss=criterion(pos_logits,pos_mask.to(torch.float),train_mask.to(torch.float))
        # for text class loss
        pixel_cls_loss=F.cross_entropy(pixel_cls_logits,pos_mask.to(torch.long),reduce=False)
        
        pixel_cls_scores=F.softmax(pixel_cls_logits,dim=1)
        pixel_neg_scores=pixel_cls_scores[:,0,:,:]
        selected_neg_pixel_mask=ohem_batch(pixel_neg_scores,pos_mask,neg_mask)
        
        n_pos=pos_mask.view(-1).sum()
        n_neg=selected_neg_pixel_mask.view(-1).sum()

        pixel_cls_weights=(cls_weight+selected_neg_pixel_mask).to(torch.float)

        cls_loss=(pixel_cls_loss*pixel_cls_weights).view(-1).sum()/(n_pos+n_neg)

        # for link loss
        if n_pos==0:
            link_loss=(pixel_link_logits*0).view(-1).sum()
            shape=pixel_link_logits.shape
            pixel_link_logits_flat=pixel_link_logits.contiguous().view(shape[0],2,8,shape[2],shape[3])
        else:
            shape=pixel_link_logits.shape
            pixel_link_logits_flat=pixel_link_logits.contiguous().view(shape[0],2,8,shape[2],shape[3])
            link_label_flat=link_label

            pixel_link_loss=F.cross_entropy(pixel_link_logits_flat,link_label_flat.to(torch.long),reduce=False)

            def get_loss(label):
                link_mask=(link_label_flat==label)
                link_weight_mask=link_weight*link_mask.to(torch.float)
                n_links=link_weight_mask.view(-1).sum()
                loss=(pixel_link_loss*link_weight_mask).view(-1).sum()/n_links
                return loss
            
            neg_loss = get_loss(0)
            pos_loss = get_loss(1)

            neg_lambda=1.0
            link_loss=pos_loss+neg_loss*neg_lambda
        
        # loss_text = criterion(texts, gt_texts, selected_masks)
        loss = cls_loss_lambda*cls_loss+link_loss_lambda*link_loss
        loss_cls.update(cls_loss.cpu().item(), imgs.size(0))
        losses.update(loss.cpu().item(), imgs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # score_text = cal_text_score(F.softmax(pixel_cls_logits,dim=1)[:,1,:,:], pos_mask, cls_label>-1, running_metric_text)
        score_text = cal_text_score(F.softmax(pixel_cls_logits,dim=1)[:,1,:,:], pos_mask, cls_label>-1, running_metric_text)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 20 == 0:
            if batch_idx%40==0:
                grid=torchvision.utils.make_grid(imgs[:2,:,:,:],4,normalize=True)
                writer.add_image("image",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

                pos_score=pixel_cls_scores[:,1:,:,:]
                grid=torchvision.utils.make_grid(pos_score[:2,:,:,:],4)
                writer.add_image("pos_score",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')
                grid=torchvision.utils.make_grid(pos_mask[:2,None,:,:].to(torch.float),4,normalize=True)
                writer.add_image("pos_mask",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

                grid=torchvision.utils.make_grid(link_label[:2,0:1,:,:].to(torch.float),4,normalize=True)
                writer.add_image("link_label_0",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

                link_score=F.softmax(pixel_link_logits_flat,dim=1)[:2,1,0:1,:,:]*pos_mask[:2,None,:,:].to(torch.float)
                grid=torchvision.utils.make_grid(link_score,4,normalize=True)
                writer.add_image("link_score_0",grid,len(train_loader)*epoch+batch_idx,dataformats='CHW')

            writer.add_scalar("cls_loss",cls_loss.cpu().item(),len(train_loader)*epoch+batch_idx)
            writer.add_scalar("link_loss",link_loss.cpu().item(),len(train_loader)*epoch+batch_idx)

            output_log  = '({batch}/{size}) Batch: {bt:.3f}s | TOTAL: {total:.0f}min | ETA: {eta:.0f}min | Loss: {loss:.4f}| Loss_cls: {loss_cls:.4f} | Acc_t: {acc: .4f} | IOU_t: {iou_t: .4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader),
                bt=batch_time.avg,
                total=batch_time.avg * batch_idx / 60.0,
                eta=batch_time.avg * (len(train_loader) - batch_idx) / 60.0,
                loss=losses.avg,
                loss_cls=loss_cls.avg,
                acc=score_text['Mean Acc'],
                iou_t=score_text['Mean IoU'])
            print(output_log)
            sys.stdout.flush()

    return (losses.avg, score_text['Mean Acc'], score_text['Mean IoU'])

def adjust_learning_rate(args, optimizer, epoch):
    global state
    if epoch in args.schedule:
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main(args):
    if args.checkpoint == '':
        args.checkpoint = "checkpoints/ic15_%s_bs_%d_ep_%d"%(args.arch, args.batch_size, args.n_epoch)
    if args.pretrain:
        if 'synth' in args.pretrain:
            args.checkpoint += "_pretrain_synth"
        else:
            args.checkpoint += "_pretrain_ic17"

    print(('checkpoint path: %s'%args.checkpoint))
    print(('init lr: %.8f'%args.lr))
    print(('schedule: ', args.schedule))
    sys.stdout.flush()

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    writer=SummaryWriter(args.checkpoint)

    kernel_num=18
    start_epoch = 0

    data_loader = IC15Loader(is_transform=True, img_size=args.img_size)
    train_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=True)

    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=kernel_num)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=kernel_num)
    elif args.arch == "vgg16":
        model = models.vgg16(pretrained=False,num_classes=kernel_num)
    
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        # NOTE 这个地方的momentum对训练影响相当之大，使用0.99时训练crossentropy无法收敛.
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    title = 'icdar2015'
    if args.pretrain:
        print('Using pretrained model.')
        assert os.path.isfile(args.pretrain), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint['state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])
    elif args.resume:
        print('Resuming from checkpoint.')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        print('Training from scratch.')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss','Train Acc.', 'Train IOU.'])

    for epoch in range(start_epoch, args.n_epoch):
        adjust_learning_rate(args, optimizer, epoch)
        print(('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.n_epoch, optimizer.param_groups[0]['lr'])))

        train_loss, train_te_acc, train_te_iou = train(train_loader, model, dice_loss, optimizer, epoch,writer)
        if epoch %40==39:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': args.lr,
                    'optimizer' : optimizer.state_dict(),
                }, checkpoint=args.checkpoint,filename='checkpoint_%d.pth'%epoch)

        logger.append([optimizer.param_groups[0]['lr'], train_loss, train_te_acc, train_te_iou])
    logger.close()

if __name__ == '__main__':
    # import crash_on_ipy
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--img_size', nargs='?', type=int, default=640, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=600, 
                        help='# of the epochs')
    parser.add_argument('--schedule', type=int, nargs='+', default=[600],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16,
                        help='Batch Size')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--pretrain', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

    args = parser.parse_args()

    main(args)