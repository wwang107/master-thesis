from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN
import yacs

_C = CN()

# DATASET related params
_C.DATASET = CN()
_C.DATASET.COCOROOT = './dataset/coco'
_C.DATASET.CMUROOT = './dataset/panoptic'
_C.DATASET.DATASET = 'coco_kpt'
_C.DATASET.DATASET_TEST = 'coco'
_C.DATASET.NUM_JOINTS = 17
_C.DATASET.TRAIN = 'train2017'
_C.DATASET.TEST = 'val2017'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.INPUT_SIZE = 256
_C.DATASET.OUTPUT_SIZE = (128, 128)

# CMU DATASET SPECIFIC
_C.DATASET.NUM_VIEW = 5
_C.DATASET.NUM_FRAME_PER_SUBSEQ = 1
_C.DATASET.MAX_NUM_PEOPLE = 10

# DATASET augmentation related params
_C.DATASET.MIN_SCALE=0.8
_C.DATASET.MAX_SCALE = 1.2
_C.DATASET.MAX_TRANSLATE = 50
_C.DATASET.MAX_ROTATION = 60
# heatmap generator (default is OUTPUT_SIZE/64)
_C.DATASET.SIGMA =  3

# TRAINING PARAMS
_C.DATASET.CMU_BATCH_SIZE = 32
_C.DATASET.CMU_NUM_WORKERS = 8

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
