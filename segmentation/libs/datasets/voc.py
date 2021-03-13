#!/usr/bin/env python
# coding: utf-8
#


from __future__ import absolute_import, print_function

import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from .base import _BaseDataset


class VOC(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset
    """

    def __init__(self, year=2012, **kwargs):
        self.year = year
        super(VOC, self).__init__(**kwargs)

    def _set_files(self):
        self.root = osp.join(self.root, "VOC{}".format(self.year))
        self.image_dir = osp.join(self.root, "JPEGImages")
        self.label_dir = osp.join(self.root, "SegmentationClass")

        if self.split in ["train", "trainval", "val", "test"]:
            file_list = osp.join(
                 "/y_dir/segmentation/list", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.image_dir, image_id + ".jpg")
        label_path = osp.join(self.label_dir, image_id + ".png")
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image_id, image, label


class VOCAug(_BaseDataset):
    """
    PASCAL VOC Segmentation dataset with extra annotations
    """

    def __init__(self, year=2012, **kwargs):
        self.year = year
        super(VOCAug, self).__init__(**kwargs)

    def _set_files(self):
        self.root = osp.join(self.root, "VOC{}".format(self.year))

        if self.split in ["train", "train_aug", "trainval", "trainval_aug", "val"]:
            file_list = osp.join(
                 "/your_dir/segmentation/list", self.split + ".txt"
            )
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip().split(" ") for id_ in file_list]
            self.files, self.labels = list(zip(*file_list))
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index].split("/")[-1].split(".")[0]
        image_path = osp.join(self.root, self.files[index][1:])
        label_path = osp.join(self.root, self.labels[index][1:])
        #label_path = osp.join("/your_dir/classification/pseudo_labels", self.labels[index][1:].split("/")[-1])
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image_id, image, label

