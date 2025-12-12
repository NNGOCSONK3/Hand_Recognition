import cv2
import os
import csv
import numpy as np
import hand_improved as htm

# --- CẤU HÌNH ĐƯỜNG DẪN MỚI ---
DATASET_ROOT = "Dataset"
MODELS_ROOT = "Models"

IMAGE_DATASET_DIR = os.path.join(DATASET_ROOT, "dataset_hand")
OUTPUT_CSV = os.path.join(MODELS_ROOT, "hand_landmarks_data.csv")

if not os.path.exists(MODELS_ROOT): os.makedirs(MODELS_ROOT)

detector = htm.HandDetector(detectionCon=0.5, maxHands=1) 

print(f"[INFO] Đang khởi tạo file {OUTPUT_CSV}...")
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['label']
    for i in range(21): header.extend([f'x{i}', f'y{i}'])
    writer.writerow(header)

if not os.path.exists(IMAGE_DATASET_DIR):
    print(f"[LỖI] Không tìm thấy thư mục {IMAGE_DATASET_DIR}")
    exit()

classes = os.listdir(IMAGE_DATASET_DIR)
total = 0; success = 0

for label in classes:
    class_path = os.path.join(IMAGE_DATASET_DIR, label)
    if not os.path.isdir(class_path): continue
    
    print(f" -> Đang xử lý lớp: {label}")
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        total += 1
        detector.findHands(img, draw=False)
        lmList, _ = detector.findPosition(img, draw=False)
        
        if lmList:
            base_x, base_y = lmList[0][1], lmList[0][2]
            max_dist = 0
            for lm in lmList:
                dist = np.hypot(lm[1] - base_x, lm[2] - base_y)
                if dist > max_dist: max_dist = dist
            if max_dist == 0: max_dist = 1
            
            row = [label]
            for lm in lmList:
                norm_x = (lm[1] - base_x) / max_dist
                norm_y = (lm[2] - base_y) / max_dist
                row.extend([norm_x, norm_y])
            
            with open(OUTPUT_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            success += 1

print(f"[HOÀN TẤT] Tổng ảnh: {total} | Trích xuất thành công: {success}")