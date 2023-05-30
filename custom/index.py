import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import math
from module import model_fish_classification

image_fish = 'fish4.jpg'
# Images
img = cv2.imread(image_fish)

image_length = max(img.shape[0], img.shape[1])

# Inference
results = model_fish_classification(img)  

boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)


fish_classes = results.pandas().xyxy[0]['name'].tolist()
xmin = boxes['xmin'].tolist()
ymin = boxes['ymin'].tolist()
xmax = boxes['xmax'].tolist()
ymax = boxes['ymax'].tolist()

# Đọc ảnh vào một mảng NumPy
image_array = np.array(Image.open(image_fish))
# Lặp qua danh sách bounding boxes
fish_objects = []
for i in range(len(xmin)):
    # Chuyển đổi tọa độ bounding box sang số nguyên
    xmin_i, ymin_i, xmax_i, ymax_i = int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i])

    print(xmin_i, ymin_i, xmax_i, ymax_i)
    # Cắt ảnh chứa vật thể
    object_array = image_array[ymin_i:ymax_i, xmin_i:xmax_i]

    # Tạo đối tượng Image từ mảng NumPy
    object_image = Image.fromarray(object_array)

    # Lưu ảnh cắt từng con cá
    image_path = f'object_{i+1}.jpg'
    object_image.save(image_path)

    # Lưu thông tin tọa độ, tên loài cá và đường dẫn ảnh vào danh sách các đối tượng
    fish_object = {
        'species': fish_classes[i],  # Thay bằng tên loài cá tương ứng
        'xmin': xmin_i,
        'ymin': ymin_i,
        'xmax': xmax_i,
        'ymax': ymax_i,
        'image_length': image_length,
        'image': image_path
    }
    fish_objects.append(fish_object)

# In danh sách các đối tượng cá với thông tin tọa độ và đường dẫn ảnh
for fish_object in fish_objects:
    print('species:', fish_classes[i])
    print('Bounding Box:', fish_object['xmin'], fish_object['ymin'], fish_object['xmax'], fish_object['ymax'])
    print('Image:', fish_object['image'])
    print('---')

# Lưu danh sách các đối tượng cá vào file (ví dụ: JSON)
import json

output_file = 'fish_objects.json'
with open(output_file, 'w') as f:
    json.dump(fish_objects, f)

print('Saved fish objects to', output_file)