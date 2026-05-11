import os
import cv2 as cv
import numpy as np

def load_data(base_dir, split="train"):
    categories = {"NORMAL": -1, "PNEUMONIA": 1} 
    images = []
    labels = []

    for category, label in categories.items():
        path = os.path.join(base_dir, split, category)
    
        if not os.path.exists(path):
            print(f"Cảnh báo: Không tìm thấy thư mục {path}")
            continue
            
        print(f"Đang tải dữ liệu từ: {category}...")
        
        for img_file in os.listdir(path):
            try:
                img_path = os.path.join(path, img_file)
                image = cv.imread(img_path)
                
                if image is None: continue 
                image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                image = cv.resize(image, (128, 128), interpolation=cv.INTER_NEAREST)
                image = image / 255.0
                images.append(image.flatten())
                labels.append(label)
                
            except Exception as e:
                print(f"Lỗi khi xử lý file {img_file}: {e}")

    
    return np.array(images), np.array(labels)