import json
import cv2
from PIL import Image
import math
from module import model_detect_head_tail

# Load file JSON chứa thông tin tên cá và đường dẫn ảnh
json_file = 'fish_objects.json'
with open(json_file, 'r') as f:
    fish_objects = json.load(f)

# Duyệt qua từng đối tượng cá
for fish_object in fish_objects:
    # Đọc ảnh cá từ đường dẫn
    image_path = fish_object['image']

    image_length = fish_object['image_length']

    img = cv2.imread(image_path)


    # Detect đầu đuôi của cá
    results = model_detect_head_tail(img)  # Gọi mô hình detect đầu đuôi cá
    # Xử lý kết quả detect
        
    head_box = []
    tail_box = []
    fish_box = []

    for i in range(len(results.pandas().xyxy[0])):
        name = results.pandas().xyxy[0].iloc[i][6]
        box = results.pandas().xyxy[0].iloc[i]

        if name == 'đầu':
            head_box.append(box)
        elif name == 'đuôi':
            tail_box.append(box)
        else:
            fish_box.append(box)

    if(len(head_box) == 1 & len(tail_box) == 1 & len(fish_box) == 1):
    
        # Chuyển đổi tọa độ bounding box sang số nguyên
        head_xmin, head_ymin, head_xmax, head_ymax = map(int, head_box[0][['xmin', 'ymin', 'xmax', 'ymax']].values)
        tail_xmin, tail_ymin, tail_xmax, tail_ymax = map(int, tail_box[0][['xmin', 'ymin', 'xmax', 'ymax']].values)
    

        # Hiển thị ảnh cá và vẽ bounding box cho đầu đuôi cá
        cv2.rectangle(img, (head_xmin, head_ymin), (head_xmax, head_ymax), (0, 255, 0), 2)  # Vẽ bounding box cho đầu cá
        cv2.rectangle(img, (tail_xmin, tail_ymin), (tail_xmax, tail_ymax), (0, 0, 255), 2)  # Vẽ bounding box cho đuôi cá
        cv2.imshow('Fish Image', img)
        cv2.waitKey(0)

        head_center_x = (head_xmin + head_xmax) / 2
        head_center_y = (head_ymin + head_ymax) / 2

        # Tọa độ tâm của bounding box đuôi cá
        tail_center_x = (tail_xmin + tail_xmax) / 2
        tail_center_y = (tail_ymin + tail_ymax) / 2


        if head_center_x < tail_center_x:
            head_edge2_x = head_xmin
        else:
            head_edge2_x = head_xmax
        head_edge2_y = int(head_center_y + (head_edge2_x - head_center_x) * (tail_center_y - head_center_y) / (tail_center_x - head_center_x))

        # Tính toán tọa độ điểm cắt trên cạnh thứ hai của tail_box
        if tail_center_x < head_center_x:
            tail_edge2_x = tail_xmin
        else:
            tail_edge2_x = tail_xmax
        tail_edge2_y = int(tail_center_y + (tail_edge2_x - tail_center_x) * (head_center_y - tail_center_y) / (head_center_x - tail_center_x))

        cv2.line(img, (int(head_edge2_x), head_edge2_y), (int(tail_edge2_x), tail_edge2_y), (0, 255, 0), 2)

        fish_length = math.sqrt((tail_edge2_x - head_edge2_x)**2 + (tail_edge2_y - head_edge2_y)**2)

        # Hiển thị hình ảnh với đường thẳng
        cv2.imshow('Image with Lines', img)
        cv2.waitKey(0)


        print('Độ dài từ tâm bounding box đầu cá đến tâm bounding box đuôi cá:', fish_length)
        # Tính tỉ lệ đường với cạnh dài của bounding box
        ratio = fish_length / image_length

        print('Tỉ lệ đường với cạnh dài của bounding box:', ratio)

        # Lưu ảnh đã detect đầu đuôi cá (tùy chọn)
        image_name = image_path.split('.')[0]  # Tên ảnh gốc (loại bỏ phần mở rộng định dạng)
        image_with_detection_path = f'{image_name}_detection.jpg'
        cv2.imwrite(image_with_detection_path, img)

        # Cập nhật thông tin đầu và đuôi cá vào đối tượng cá
        fish_object['head'] = {'xmin': head_xmin, 'ymin': head_ymin, 'xmax': head_xmax, 'ymax': head_ymax}
        fish_object['tail'] = {'xmin': tail_xmin, 'ymin': tail_ymin, 'xmax': tail_xmax, 'ymax': tail_ymax}

# Lưu danh sách đối tượng cá đã được cập nhật thông tin đầu đuôi vào file JSON
output_file = 'fish_objects_updated.json'
with open(output_file, 'w') as f:
    json.dump(fish_objects, f)

print('Updated fish objects with head and tail information saved to', output_file)