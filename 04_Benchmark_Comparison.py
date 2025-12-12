import cv2
import numpy as np
import os
import time
import pickle
import sys
from tensorflow.keras.models import load_model

# Import các thư viện tự code
try: from hand_improved import HandDetector
except: pass
try: from knn_manual import KNN_Manual 
except: pass
# LBP không dùng nữa trong benchmark này nên có thể bỏ qua hoặc để try/except
try: from lbp_manual import LocalBinaryPatterns
except: pass

# --- CẤU HÌNH ---
MODELS_ROOT = "Models"
DATASET_ROOT = "Dataset"
TEST_DATA_DIR = os.path.join(DATASET_ROOT, "dataset_hand") 

# Đường dẫn model
KNN_PATH = os.path.join(MODELS_ROOT, "knn_manual.pkl") 
SVM_SKELETON_PATH = os.path.join(MODELS_ROOT, "svm_skeleton.pkl") # Quay lại dùng SVM Skeleton
CNN_PATH = os.path.join(MODELS_ROOT, "cnn_lbp.h5") # Hoặc cnn_hand.h5 tùy file bạn có
SCALER_PATH = os.path.join(MODELS_ROOT, "scaler.pkl")

# --- LOAD MODELS ---
print("[INFO] Loading Models...")
models_loaded = {}

# 1. Setup chung
detector = HandDetector(detectionCon=0.5, maxHands=1)
if os.path.exists(SCALER_PATH):
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
else: scaler = None

# 2. KNN Manual
if os.path.exists(KNN_PATH):
    models_loaded['KNN'] = pickle.load(open(KNN_PATH, 'rb'))

# 3. SVM (Skeleton) - SỬA LẠI: Dùng model Skeleton
if os.path.exists(SVM_SKELETON_PATH):
    models_loaded['SVM'] = pickle.load(open(SVM_SKELETON_PATH, 'rb'))
    # Không cần load LBP descriptor nữa

# 4. CNN
if os.path.exists(CNN_PATH):
    models_loaded['CNN'] = load_model(CNN_PATH)

IMG_SIZE_CNN = (64, 64)

# --- BIẾN THỐNG KÊ ---
stats = {
    'Geometric': {'correct': 0, 'time': 0},
    'KNN Manual': {'correct': 0, 'time': 0},
    'SVM (Skel)': {'correct': 0, 'time': 0}, # Đổi tên hiển thị
    'CNN (DL)':   {'correct': 0, 'time': 0}
}
total_samples = 0

print(f"[INFO] Bắt đầu Benchmark trên tập dữ liệu: {TEST_DATA_DIR}")
if not os.path.exists(TEST_DATA_DIR):
    print("[LỖI] Không tìm thấy thư mục dataset.")
    exit()

classes = sorted(os.listdir(TEST_DATA_DIR))

for label in classes:
    class_path = os.path.join(TEST_DATA_DIR, label)
    if not os.path.isdir(class_path): continue
    
    if not label.isdigit(): continue
    true_label = int(label)
    
    print(f" -> Testing Class: {label}...")
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img_origin = cv2.imread(img_path)
        if img_origin is None: continue
        
        total_samples += 1
        
        # --- 1. GEOMETRIC, KNN, SVM (SKELETON) ---
        # Cả 3 phương pháp này đều cần MediaPipe tìm xương tay trước
        start = time.time()
        img_geo = img_origin.copy()
        detector.findHands(img_geo, draw=False)
        lmList, _ = detector.findPosition(img_geo, draw=False)
        
        # > Geometric Predict
        try:
            geo_pred = detector.fingersUp().count(1)
        except: geo_pred = -1
        stats['Geometric']['time'] += (time.time() - start)
        if geo_pred == true_label: stats['Geometric']['correct'] += 1
        
        # Chuẩn bị dữ liệu xương (cho KNN và SVM)
        input_vec = None
        if lmList and scaler:
            base_x, base_y = lmList[0][1], lmList[0][2]
            max_dist = max([np.hypot(lm[1]-base_x, lm[2]-base_y) for lm in lmList]) or 1
            row = []
            for lm in lmList:
                row.extend([(lm[1]-base_x)/max_dist, (lm[2]-base_y)/max_dist])
            input_vec = scaler.transform([row])

        # > KNN Predict
        start = time.time()
        if 'KNN' in models_loaded and input_vec is not None:
            knn_pred = models_loaded['KNN'].predict(input_vec)[0]
            if knn_pred == true_label: stats['KNN Manual']['correct'] += 1
        stats['KNN Manual']['time'] += (time.time() - start)

        # > SVM Predict (Skeleton)
        start = time.time()
        if 'SVM' in models_loaded and input_vec is not None:
            svm_pred = models_loaded['SVM'].predict(input_vec)[0]
            if svm_pred == true_label: stats['SVM (Skel)']['correct'] += 1
        stats['SVM (Skel)']['time'] += (time.time() - start)

        # --- 2. CNN (IMAGE) ---
        start = time.time()
        if 'CNN' in models_loaded:
            img_resize = cv2.resize(img_origin, IMG_SIZE_CNN)
            # Chuẩn hóa ảnh cho CNN
            img_cnn = img_resize.astype('float32') / 255.0
            img_cnn = np.expand_dims(img_cnn, axis=0) # (1, 64, 64, 3)
            
            pred_prob = models_loaded['CNN'].predict(img_cnn, verbose=0)
            cnn_pred = np.argmax(pred_prob)
            if cnn_pred == true_label: stats['CNN (DL)']['correct'] += 1
        stats['CNN (DL)']['time'] += (time.time() - start)

# --- IN KẾT QUẢ ---
print("\n" + "="*65)
print(f"KẾT QUẢ SO SÁNH TRÊN {total_samples} ẢNH TEST")
print("="*65)
print(f"{'Method':<20} | {'Accuracy (%)':<15} | {'Avg Time (ms)':<15}")
print("-" * 65)

for name, data in stats.items():
    if total_samples > 0:
        acc = (data['correct'] / total_samples) * 100
        avg_time = (data['time'] / total_samples) * 1000 # ms
        print(f"{name:<20} | {acc:<15.2f} | {avg_time:<15.4f}")
    else:
        print(f"{name:<20} | N/A             | N/A")

print("="*65)

