from yacs.config import CfgNode as CN


_C = CN()
_C.DEVICE = 'cuda'
_C.FRAMES = 27
_C.MOTIONAGF_CKPT = './libs/pose/MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr'
_C.SEED = 42
_C.WORKSPACE = './workspace/default'

_C.CALIB = CN()
_C.CALIB.BA = False
_C.CALIB.BA_LAMBDA1 = 1.0
_C.CALIB.BA_LAMBDA2 = 10.0
_C.CALIB.RANSAC = False
_C.CALIB.RANSAC_ITER = 100
_C.CALIB.TH_2D = 555.0
_C.CALIB.TH_3D = 1.0
_C.CALIB.INTRINSIC_C1 = './data/Runner/intrinsic/iphone13.npz'
_C.CALIB.INTRINSIC_C2 = './data/Runner/intrinsic/iphone11pro.npz'
_C.CALIB.INTRINSIC_C3 = './data/Runner/intrinsic/iphone13.npz'

_C.MMPOSE = CN()
_C.MMPOSE.DET_CONFIG = './common/mmpose/mmdet_cfg/rtmdet_m_640-8xb32_coco-person.py'
_C.MMPOSE.DET_CKPT = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
_C.MMPOSE.VITPOSE_CONFIG = './common/mmpose/mmpose_cfg/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
_C.MMPOSE.VITPOSE_CKPT = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'

_C.MMPOSE.DET_ID = 0
_C.MMPOSE.DET_BOX_THR = 0.5
_C.MMPOSE.DET_NMS_THR = 0.5
_C.MMPOSE.OKS_TRACKING = True
_C.MMPOSE.TRACKING_THR = 0.3
_C.MMPOSE.NORM_POSE2D = True

_C.NEPTUNE = CN()
_C.NEPTUNE.PROJECT = 'username/project'
_C.NEPTUNE.TOKEN_PATH = './token/neptune.txt'

_C.TUNING = CN()
_C.TUNING.DATA_DIRS = [
    './data/Runner/run_001/video',
    './data/Runner/run_002/video',
    './data/Runner/run_004/video'
]
_C.TUNING.DATA2D_PATH = './data/Runner/data2d/all.pkl'
_C.TUNING.TRAIN_DATA = ['run_001', 'run_002', 'run_004']
_C.TUNING.EVAL_DATA = ['run_001', 'run_002', 'run_004']
_C.TUNING.TRAIN_RATIO = 0.9
_C.TUNING.BATCHSIZE = 16
_C.TUNING.LEARNING_RATE = 0.0005
_C.TUNING.LR_DECAY = 0.99
_C.TUNING.WEIGHT_DECAY = 0.01
_C.TUNING.EPOCHS = 100
_C.TUNING.ITERATIONS = 5


def get_cfg_defaults():
    return _C.clone()
