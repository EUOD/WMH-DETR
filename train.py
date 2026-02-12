import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/root/RTDETR-main/ultralytics/cfg/models/rt-detr/rtdetr-p2.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='/root/RTDETR-main/dataset/UDOD/UDOD.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16, 
                workers=8, 
                device='0', 
                project='runs/UDD',
                name='train',
                )
