import cv2
from PIL import Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing
import os
from os.path import exists

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,  
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']


data_path = './'
train_lst_path = data_path + 'data/train_cls.txt'
sal_path = data_path + 'data/saliency_aug/'
att_path = data_path + 'runs/exp2/attention/'
last_att_path = data_path + 'runs/exp1/attention/'
save_path = './pseudo_labels/'

if not exists(save_path):
	os.makedirs(save_path)
		
with open(train_lst_path) as f:
    lines = f.readlines()

def gen_gt(index):
    line = lines[index]
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    bg_name = sal_path + name + '.png'
    if not os.path.exists(bg_name):
        return
    sal = cv2.imread(bg_name, 0)
    height, width = sal.shape
    gt = np.zeros((21, height, width), dtype=np.float32)
    added_gt = np.zeros((21, height, width), dtype=np.float32)                 
    added_gt[0] = 0.5                                                         
    sal = np.array(sal, dtype=np.float32)
    
    # some thresholds. 
    conflict = 0.9
    fg_thr = 0.3
    # the below two values are used for generating uncertainty pixels
    bg_thr = 32
    att_thr = 0.8

    # use saliency map to provide background cues
    gt[0] = (1 - (sal / 255))
    init_gt = np.zeros((height, width), dtype=float) 
    sal_att = sal.copy()  
    
    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])
        att_name = att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            continue
        
        # normalize attention to [0, 1] 
        att = cv2.imread(att_name, 0)
        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        gt[cls+1] = att.copy()
        sal_att = np.maximum(sal_att, (att > att_thr) *255)
    
    
    # throw low confidence values for all classes
    gt[gt < fg_thr] = 0
    
    # conflict pixels with multiple confidence values
    bg = np.array(gt > conflict, dtype=np.uint8)  
    bg = np.sum(bg, axis=0)
    gt = gt.argmax(0).astype(np.uint8)
    gt[bg > 1] = 255
    
    # pixels regarded as background but confidence saliency values 
    bg = np.array(sal_att >= bg_thr, dtype=np.uint8) * np.array(gt == 0, dtype=np.uint8)
    gt[bg > 0] = 255  

    #POM
    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])
        att_name = last_att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            continue
        
        # normalize attention to [0, 1] 
        att = cv2.imread(att_name, 0)
        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        position = [gt==(cls+1)]

        temp = att[tuple(position)]
        if np.sum(temp)!=0: 
            flt_thr = np.median(temp) 
        else:                         
            position = [att > 0.3]
            if np.sum(position) != 0:
                temp = att[tuple(position)]
                temp_median = np.median(temp)  
                position = [att > temp_median]
                temp = att[tuple(position)]
                flt_thr = np.median(temp) 
            else:
                flt_thr = 1
        
        select_position = np.where(att > flt_thr, 1, 0)
        added_gt[cls+1] = select_position

    ignore = np.sum(added_gt, axis=0)
    added_gt = np.zeros((height, width), dtype=np.uint8)
    added_gt[ignore > 0.6] = 255                     # if there is a class, the background should be ignored

    flag = ((gt==0) & (added_gt == 255))

    gt = np.where(flag, 255, gt)
    
    # we ignore the whole image for an image with a small ratio of semantic objects
    out = gt 
    valid = np.array((out > 0) & (out < 255), dtype=int).sum()
    ratio = float(valid) / float(height * width)
    if ratio < 0.01:
        out[...] = 255

    # output the proxy labels using the VOC12 label format
    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)

### Parallel Mode
pool = multiprocessing.Pool(processes=16)
pool.map(gen_gt, range(len(lines)))
#pool.map(gen_gt, range(100))
pool.close()
pool.join()

