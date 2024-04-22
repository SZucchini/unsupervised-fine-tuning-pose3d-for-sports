from yacs.config import CfgNode as CN

_C = CN()
_C.DEVICE = 'cuda'
_C.FRAMES = 27
_C.PRETRAINED_CKPT = './libs/MotionAGFormer/checkpoint/best_epoch.pth.tr'
_C.SEED = 42
_C.WORKSPACE = ''

_C.CALIB = CN()
_C.CALIB.BA = False
_C.CALIB.BA_LAMBDA1 = 1.0
_C.CALIB.BA_LAMBDA2 = 10.0
_C.CALIB.RANSAC = False
_C.CALIB.RANSAC_ITER = 100
_C.CALIB.TH_2D = 555.0
_C.CALIB.TH_3D = 1.0

_C.DATA = CN()
_C.DATA.PSEUDO_PATH = './data/ASP-27/calib_valid'
_C.DATA.TEST_PATH = './data/ASP-27/valid'
_C.DATA.TRAIN_PATH = './data/ASP-27/valid'

_C.TRAIN = CN()
_C.TRAIN.BATCHSIZE = 16
_C.TRAIN.EPOCHS = 60
_C.TRAIN.LEARNING_RATE = 0.0005
_C.TRAIN.LR_DECAY = 0.99
_C.TRAIN.SCALE_WEIGHT = 0.5
_C.TRAIN.VELOCITY_WEIGHT = 20.0
_C.TRAIN.WEIGHT_DECAY = 0.01


_C.TUNING = CN()
_C.TUNING.ITERATIONS = 5


def get_cfg_defaults():
    return _C.clone()
