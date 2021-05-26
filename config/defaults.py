from yacs.config import CfgNode

_C = CfgNode()

# Train Params
_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.CE_CLASS_WEIGHT = [1., 1., 2.5, 1., 2.5, 1., 1., 1., 1., 2.5] 
# _C.TRAIN.CE_CLASS_WEIGHT = []
_C.TRAIN.LOAD_WEIGHT_PATH = ""
_C.TRAIN.LOSS_WEIGHT = []
_C.TRAIN.LEARNING_RATE = 1e-4
_C.TRAIN.MODEL_TYPE = ""
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NUM_EPOCHS = 1
_C.TRAIN.NUM_GPUS = 1
_C.TRAIN.OPTIMIZER_TYPE = ""
_C.TRAIN.RUN_NAME = ""
_C.TRAIN.SEED = 1

# Data Params
_C.DATA = CfgNode()
_C.DATA.CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

_C.DATA.IMG_SIZE = (32, 32)
_C.DATA.ROOT_PATH = "./dataset"
# _C.DATA.TRAIN_CLASS_COUNTS = [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
_C.DATA.TRAIN_CLASS_COUNTS = [4500, 4500, 2000, 4500, 2000, 4500, 4500, 4500, 4500, 2000]
_C.DATA.VAL_CLASS_COUNTS = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500] 
_C.DATA.TRANSFORM_LIST = ['resize', 'to_tensor', 'random_horizontal_flip']
_C.DATA.NUM_WORKERS = 32
