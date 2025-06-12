import os
import time
import wandb
from ultralytics import YOLO


def train():
    project_name = "DETECT-BUBBLE"
    name = time.strftime("%Y%m%d-%H%M%S")
    model = "./ultralytics/cfg/models/v8/yolov8n.yaml"
    data = "./ultralytics/cfg/datasets/dota8-bubble.yaml"

    model = YOLO(model, task="detect")

    # Train the model
    train_results = model.train(
        project=project_name,
        name=name,
        data=data,  # path to dataset YAML
        batch=8,  # batch size
        epochs=800,  # number of epochs
        imgsz=1024,  # training image size
        patience=200,  # early stop patience (epochs without improvement)
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        hsv_h=0.015,  # (float) image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # (float) image HSV-Saturation augmentation (fraction)
        hsv_v=0.0,  # (float) image HSV-Value augmentation (fraction)
        degrees=10.0,  # (float) image rotation (+/- deg)
        translate=0.1,  # (float) image translation (+/- fraction)
        scale=0.5,  # (float) image scale (+/- gain)
        shear=0.0,  # (float) image shear (+/- deg)
        perspective=0.0,  # (float) image perspective (+/- fraction), range 0-0.001
        flipud=0.5,  # (float) image flip up-down (probability)
        fliplr=0.5,  # (float) image flip left-right (probability)
    )


def predict(model_path, image_path, task='detect', save_path=""):
    model = YOLO(model_path, task=task)  # load a pretrained YOLOv8n OBB model

    # image_path = r'E:\python_project\ultralytics\datasets\yanshichengxu\images\val'
    images = os.listdir(image_path)
    images = [os.path.join(image_path, image) for image in images]
    # Run batched inference on a list of images
    results = model(images, save=True, save_txt=True, show_labels=False, save_conf=True)  # return a list of Results objects

    # Process results list
    # for i, result in enumerate(results):
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for class ification outputs
    #     result.show()  # display to screen
    #     result.save(filename=str(os.path.join(save_path, f'result{i}.jpg')))  # save to disk


def export(path, informat='pt', outformat='onnx', task='detect'):
    if os.path.isfile(path):
        model = YOLO(path, task=task)
        success = model.export(format=outformat, imgsz=640, batch=1, dynamic=False)
        print(success)
    else:
        models = [os.path.join(path, p) for p in os.listdir(path) if p.endswith(informat)]
        for i, model_path in enumerate(models):
            model = YOLO(model_path, task=task)
            success = model.export(format=outformat, imgsz=640, batch=1, dynamic=False)
            print(success)


if __name__ == '__main__':
    # train()
    predict(
        model_path=r"G:\github\ultralyticshub\ultralytics-individual\DETECT-BUBBLE\20250608-103946\weights\best.pt",
        image_path=r"H:\dataset\BUBBLE\BUBBLE-2\images_crop_1024\images"
    )
