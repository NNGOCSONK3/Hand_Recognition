import cv2
import numpy as np
import hand_improved as htm
from tensorflow.keras.models import load_model
import pickle
import os
import sys

# --- IMPORT ---
# Cố gắng import KNN Manual từ cả root và folder Models
try:
    from knn_manual import KNN_Manual 
except ImportError:
    try:
        from Models.knn_manual import KNN_Manual
    except:
        pass # Nếu không tìm thấy file code thì thôi

# --- CẤU HÌNH ĐƯỜNG DẪN ---
MODELS_ROOT = "Models"
DATASET_ROOT = "Dataset"

# Đường dẫn file model
KNN_PATH = os.path.join(MODELS_ROOT, "knn_manual.pkl") 
SVM_PATH = os.path.join(MODELS_ROOT, "svm_skeleton.pkl")
DNN_PATH = os.path.join(MODELS_ROOT, "dnn_skeleton.h5")
SCALER_PATH = os.path.join(MODELS_ROOT, "scaler.pkl")
FINGERS_DIR = os.path.join(DATASET_ROOT, "Fingers")

# Load Models
knn_model = None
if os.path.exists(KNN_PATH):
    try:
        knn_model = pickle.load(open(KNN_PATH, 'rb'))
        print("[INFO] KNN Manual Loaded.")
    except: print("[ERR] KNN Load Failed")

svm_model = None
if os.path.exists(SVM_PATH):
    try:
        svm_model = pickle.load(open(SVM_PATH, 'rb'))
        print("[INFO] SVM Loaded.")
    except: print("[ERR] SVM Load Failed")

scaler = pickle.load(open(SCALER_PATH, 'rb')) if os.path.exists(SCALER_PATH) else None
dnn_model = load_model(DNN_PATH) if os.path.exists(DNN_PATH) else None

# Load ảnh minh họa
overlayList = []
if os.path.exists(FINGERS_DIR):
    imgList = sorted(os.listdir(FINGERS_DIR))
    for imPath in imgList:
        image = cv2.imread(os.path.join(FINGERS_DIR, imPath))
        if image is not None:
            image = cv2.resize(image, (150, 200)) 
            overlayList.append(image)
    print(f"[INFO] Loaded {len(overlayList)} finger images.")

detector = htm.HandDetector(detectionCon=0.7, maxHands=1)

wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

REQUIRED_FRAMES = 8 
class VerifyState:
    def __init__(self):
        self.curr = None; self.cnt = 0; self.ver = "..."; self.col = (0, 255, 255)
        self.is_verified = False
    def update(self, val):
        if val == self.curr: self.cnt += 1
        else: self.curr = val; self.cnt = 1
        
        if self.cnt >= REQUIRED_FRAMES:
            self.ver = self.curr; self.col = (0, 255, 0)
            self.is_verified = True
            return f"{self.ver} (OK)"
        
        self.is_verified = False
        return f"{self.curr} ({self.cnt})"

# Khởi tạo trạng thái cho 4 phương pháp
st_geo = VerifyState()
st_knn = VerifyState()
st_svm = VerifyState()
st_dnn = VerifyState()

print("[INFO] System Ready. Press 'q' to exit.")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    
    img = detector.findHands(img, draw=True)
    lmList, bbox = detector.findPosition(img, draw=False)
    
    r_geo, r_knn, r_svm, r_dnn = "?", "?", "?", "?"
    
    if lmList and scaler:
        # 1. Geometric
        fingers = detector.fingersUp()
        r_geo = str(fingers.count(1))
        
        # 2. Prepare Data
        base_x, base_y = lmList[0][1], lmList[0][2]
        max_dist = 0
        for lm in lmList:
            dist = np.hypot(lm[1]-base_x, lm[2]-base_y)
            if dist > max_dist: max_dist = dist
        if max_dist == 0: max_dist = 1
        
        row = []
        for lm in lmList:
            n_x = (lm[1] - base_x) / max_dist
            n_y = (lm[2] - base_y) / max_dist
            row.extend([n_x, n_y])
            
        input_data = scaler.transform([row])
        
        # 3. Predict KNN (Manual)
        if knn_model:
            try:
                pred = knn_model.predict(input_data)[0]
                r_knn = str(pred)
            except: pass

        # 4. Predict SVM (Library)
        if svm_model:
            try:
                pred = svm_model.predict(input_data)[0]
                r_svm = str(pred)
            except: pass
            
        # 5. Predict DNN (Deep Learning)
        if dnn_model:
            try:
                pred = dnn_model.predict(input_data, verbose=0)
                if np.max(pred) > 0.6: r_dnn = str(np.argmax(pred))
            except: pass

    # Update Verify States
    d_geo = st_geo.update(r_geo)
    d_knn = st_knn.update(r_knn)
    d_svm = st_svm.update(r_svm)
    d_dnn = st_dnn.update(r_dnn)
    
    # Hiển thị ảnh minh họa (Ưu tiên: DNN -> SVM -> KNN -> Geo)
    best_result = None
    if st_dnn.is_verified and r_dnn.isdigit(): best_result = int(r_dnn)
    elif st_svm.is_verified and r_svm.isdigit(): best_result = int(r_svm)
    elif st_knn.is_verified and r_knn.isdigit(): best_result = int(r_knn)
    elif st_geo.is_verified and r_geo.isdigit(): best_result = int(r_geo)
        
    if best_result is not None and 0 <= best_result < len(overlayList):
        overlay_img = overlayList[best_result]
        h_ol, w_ol, _ = overlay_img.shape
        x_pos = wCam - w_ol - 10; y_pos = 10
        try:
            img[y_pos:y_pos+h_ol, x_pos:x_pos+w_ol] = overlay_img
            cv2.rectangle(img, (x_pos, y_pos), (x_pos+w_ol, y_pos+h_ol), (0, 255, 0), 3)
            # cv2.putText(img, "Match", (x_pos, y_pos - 10), 2, 1, (0, 255, 0), 2)
        except: pass

    # UI Bảng Thông Số (Hiển thị 4 dòng riêng biệt)
    # Tăng chiều cao bảng để chứa đủ 4 dòng
    cv2.rectangle(img, (10, 10), (400, 230), (0,0,0), -1)
    
    cv2.putText(img, f"Geometric: {d_geo}", (20, 50), 2, 0.8, st_geo.col, 2)
    cv2.putText(img, f"KNN (ML)): {d_knn}", (20, 100), 2, 0.8, st_knn.col, 2) # Dòng KNN riêng
    cv2.putText(img, f"SVM (ML):  {d_svm}", (20, 150), 2, 0.8, st_svm.col, 2) # Dòng SVM riêng
    cv2.putText(img, f"DNN (DL):   {d_dnn}", (20, 200), 2, 0.8, st_dnn.col, 2)

    cv2.imshow("Hand Recognition Full Comparison", img)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()