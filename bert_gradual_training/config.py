import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 16
_C.DATA.MAX_LENGTH = 128
_C.DATA.NUM_WORKERS = 4
_C.DATA.PIN_MEMORY = True

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'bert'
_C.MODEL.NAME = 'bert-base-uncased'
_C.MODEL.RESUME = ''
_C.MODEL.PRETRAINED_QUAD_PATH = ''  # Path to pre-trained model with quad/2quad functions
_C.MODEL.NUM_CLASSES_RTE = 2
_C.MODEL.NUM_CLASSES_STSB = 1
_C.MODEL.DROP_RATE = 0.1
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.0
_C.MODEL.TRANSITION_TOTAL_STEPS = 300000
_C.MODEL.TRANSITION_WARMUP_STEPS = 0
_C.MODEL.TRANSITION_R0 = 1.0

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS_RTE = 300
_C.TRAIN.EPOCHS_STSB = 300
_C.TRAIN.RECALIBRATION_EPOCHS = 10
_C.TRAIN.WARMUP_EPOCHS = 0.1
_C.TRAIN.COOLDOWN_EPOCHS = 0
_C.TRAIN.BASE_LR = 2e-5
_C.TRAIN.LEARNING_RATE = 2e-5  # Keep for backwards compatibility
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# Legacy parameters for backwards compatibility
_C.TRAIN.WARMUP_STEPS = 500
_C.TRAIN.OUTPUT_DIR_RTE = './results/rte_gradual'
_C.TRAIN.OUTPUT_DIR_STSB = './results/stsb_gradual'
_C.TRAIN.LOGGING_DIR_RTE = './logs/rte_gradual'
_C.TRAIN.LOGGING_DIR_STSB = './logs/stsb_gradual'
_C.TRAIN.LOGGING_STEPS = 10

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP = True
_C.OUTPUT = ''
_C.TAG = 'default'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 42
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    config.defrost()
    if hasattr(args, 'opts') and args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if hasattr(args, 'batch_size') and args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume
    if hasattr(args, 'pretrained_quad_model') and args.pretrained_quad_model:
        config.MODEL.PRETRAINED_QUAD_PATH = args.pretrained_quad_model
    if hasattr(args, 'use_checkpoint') and args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if hasattr(args, 'amp') and args.amp:
        config.AMP = args.amp
    if hasattr(args, 'output') and args.output:
        config.OUTPUT = args.output
    if hasattr(args, 'tag') and args.tag:
        config.TAG = args.tag
    if hasattr(args, 'eval') and args.eval:
        config.EVAL_MODE = True
    if hasattr(args, 'throughput') and args.throughput:
        config.THROUGHPUT_MODE = True

    # output folder
    if config.OUTPUT:
        config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()

def get_config(args=None):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    if args is not None:
        update_config(config, args)
    return config 