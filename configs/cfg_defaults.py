# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from yacs.config import CfgNode as CN

# This is the default config for LSTR
# ---------------------------------------------------------------------------- #
# Config Definition
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Metadata
# ---------------------------------------------------------------------------- #
_C.SEED = 3

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = ''
_C.MODEL.CHECKPOINT = ''
_C.MODEL.STREAMING = False


# Network Architecture

_C.MODEL.ARCH = CN()
_C.MODEL.ARCH.POINTCLOUD = True
_C.MODEL.ARCH.POINTCLOUD_BACKBONE = 'PointConv'
_C.MODEL.ARCH.POINTCLOUD_MOTION = True
_C.MODEL.ARCH.RGB = False
_C.MODEL.ARCH.RGB_BACKBONE = 'ResNet50'
_C.MODEL.ARCH.WITH_HAND = True

# Positional Encoding
_C.MODEL.ARCH.POS_ENCODING = None

# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #
_C.DATA = CN()

_C.DATA.DATA_ROOT = '/scratch/zf540/EgoPAT3Dv2/prediction/data/dataset.hdf5'
_C.DATA.NUM_POINTS = 8192
_C.DATA.ENHANCED = False
_C.DATA.ENHANCED_ANNO = '/scratch/zf540/EgoPAT3Dv2/prediction/data/annotation_transformation.hdf5'
# Data Loader
_C.DATA.DATA_LOADER = CN()
_C.DATA.DATA_LOADER.BATCH_SIZE = 8
_C.DATA.DATA_LOADER.NUM_WORKERS = 10
_C.DATA.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------------- #
_C.TRAINING = CN()
_C.TRAINING.NUM_EPOCHS = 30

# Optimizer
_C.TRAINING.OPTIMIZER = 'Adam'
_C.TRAINING.LEARNING_RATE = 0.005
_C.TRAINING.DECAY_RATE = 1e-4

# Loss
_C.TRAINING.LOSS = 'Ori'

# ---------------------------------------------------------------------------- #
# Testing
# ---------------------------------------------------------------------------- #
_C.TESTING = CN()
_C.TESTING.BATCH_SIZE = 1
_C.TESTING.SEEN = True
_C.TESTING.ENHANCED = False
_C.TESTING.DATASET = None
# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
_C.VERBOSE = False


def get_cfg():
    return _C.clone()
