import numpy as np

from pixel_link import decode_batch,mask_to_bboxes

def to_bboxes(image_data, pixel_pos_scores, link_pos_scores):
    link_pos_scores=np.transpose(link_pos_scores,(0,2,3,1))    
    mask = decode_batch(pixel_pos_scores, link_pos_scores,0.6,0.9)[0, ...]
    bboxes = mask_to_bboxes(mask, image_data.shape)
    return mask,bboxes

