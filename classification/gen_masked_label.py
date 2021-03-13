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

seg_path = '/your_dir/segmentation/data/scores/voc12/deeplabv2_resnet101_msc/train_aug/trainaug_pred/'
sal_label_path = data_path + 'pseudo_labels/'
sal_path = data_path + 'data/saliency_aug/'

save_path = './masked_labels/'

if not exists(save_path):
	os.makedirs(save_path)
		
with open(train_lst_path) as f:
    lines = f.readlines()

# generate proxy ground-truth
def gen_gt(index):
    line = lines[index]
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    seg_name = seg_path + name + '.png'
    sal_label_name = sal_label_path + name + '.png'
    bg_name = sal_path + name + '.png'
    
    if not os.path.exists(seg_name):
        print('seg_name is wrong')
        return
    if not os.path.exists(sal_label_name):
        print('sal_label_name is wrong')
        return
    if not os.path.exists(bg_name):
        print('bg_name is wrong')
        return
    gt = np.asarray(Image.open(seg_name), dtype=np.int32)
    sal_label = np.asarray(Image.open(sal_label_name), dtype=np.int32)
    sal = cv2.imread(bg_name, 0)
    sal = np.array(sal, dtype=np.float32)

    height, width = gt.shape

    if len(fields) > 2:
        flag = (((sal_label > 0) & (sal_label < 255)) & (gt==0))
        gt = np.where(flag, sal_label, gt)

        init_mask = np.zeros((height, width), dtype=float) 
        flag = ((gt != 0) | (sal != 0))
        init_mask = np.where(flag, 1, init_mask)
        kernel = np.ones((30, 30), np.uint8)
        last_mask = cv2.dilate(init_mask, kernel, iterations=1)
        gt[last_mask==0] = 255

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

# Loop Mode
#for i in range(len(lines)):
#    gen_gt(i)
