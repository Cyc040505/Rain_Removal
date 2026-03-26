from typing import Any, List
from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'ps128_bs1'

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 16
        self._C.OPTIM.NUM_EPOCHS = 250
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 1
        self._C.TRAINING.SAVE_AFTER_EVERY = 10
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False
        self._C.TRAINING.TRAIN_DIR = 'images_dir/train'
        self._C.TRAINING.VAL_DIR = 'images_dir/val'
        self._C.TRAINING.SAVE_MODEL_DIR = '../model'
        self._C.TRAINING.TRAIN_PS = 128
        self._C.TRAINING.VAL_PS = 64

        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        self._C.freeze()

    def dump(self, file_path: str):
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
