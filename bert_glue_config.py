import argparse
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'bert-base-uncased'
_C.MODEL.NUM_CLASSES_RTE = 2
_C.MODEL.NUM_CLASSES_STSB = 1

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 16
_C.DATA.MAX_LENGTH = 128

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS_RTE = 3
_C.TRAIN.EPOCHS_STSB = 3
_C.TRAIN.LEARNING_RATE = 2e-5
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.WARMUP_STEPS = 500
_C.TRAIN.OUTPUT_DIR_RTE = './results/rte'
_C.TRAIN.OUTPUT_DIR_STSB = './results/stsb'
_C.TRAIN.LOGGING_DIR_RTE = './logs/rte'
_C.TRAIN.LOGGING_DIR_STSB = './logs/stsb'
_C.TRAIN.LOGGING_STEPS = 10

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP = True
_C.SEED = 42


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
