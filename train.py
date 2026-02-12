import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('/root/RTDETR-main/ultralytics/cfg/models/rt-detr/rtdetr.yaml')
    model.train(data='/root/RTDETR-main/dataset/UDD/UDD.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=16, 
                workers=8, 
                device='0', 
                project='runs/UDD',
                name='train',
                )
