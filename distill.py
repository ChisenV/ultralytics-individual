import warnings
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'G:\github\ultralyticshub\ultralytics-individual\ultralytics\cfg\models\v8\yolov8n.yaml',
        'data': r'./ultralytics/cfg/datasets/dota8-bubble.yaml',
        'imgsz': 1024,
        'epochs': 800,
        'batch': 8,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        # 'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project': 'BUBBLE-distill',
        'name': 'yolov8n-distill',

        # distill
        'prune_model': False,
        'teacher_weights': r'G:\github\ultralyticshub\ultralytics-individual\DETECT-BUBBLE\20250607-144203\weights\best.pt',
        'teacher_cfg': r'G:\github\ultralyticshub\ultralytics-individual\ultralytics\cfg\models\v8\yolov8m-detect.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',

        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,

        'teacher_kd_layers': '15,18,21',
        'student_kd_layers': '15,18,21',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()
