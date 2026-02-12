import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info
。

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = '/root/RTDETR-main/runs/UDD/weights/best.pt'
    model = RTDETR(model_path)
    result = model.val(data='/root/RTDETR-main/dataset/UDD/UDD.yaml',
                      split='test',
                      imgsz=640,
                      batch=1,
                      project='runs/val',
                      name='exp_P2_val',
                      )
