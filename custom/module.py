import torch


fish_classification = torch.hub.load('../yolov5_fish_classification', 'custom', path='../yolov5_fish_classification/runs/train/custom_model/weights/best.pt', source='local')  # local repo
detect_head_tail = torch.hub.load('../yolov5_detect_head_tail', 'custom', path='../yolov5_detect_head_tail/runs/train/custom_model/weights/best.pt', source='local')  # local repo



def model_fish_classification(img):
    return fish_classification(img, size=640)

def model_detect_head_tail(img):
    return detect_head_tail(img, size=640)