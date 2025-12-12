import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from knn_manual import KNN_Manual # Import thư viện tự code

MODELS_ROOT = "Models"
if not os.path.exists(MODELS_ROOT): os.makedirs(MODELS_ROOT)

DATA_FILE = os.path.join(MODELS_ROOT, "hand_landmarks_data.csv")
KNN_MODEL_PATH = os.path.join(MODELS_ROOT, "knn_manual.pkl")
DNN_MODEL_PATH = os.path.join(MODELS_ROOT, "dnn_skeleton.h5")
SCALER_PATH = os.path.join(MODELS_ROOT, "scaler.pkl")

# 1. Load Data
if not os.path.exists(DATA_FILE):
    print(f"[LỖI] Không tìm thấy file {DATA_FILE}")
    exit()

df = pd.read_csv(DATA_FILE).dropna()
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values.astype(int)

# 2. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Train KNN Manual
print("[INFO] Training KNN Manual...")
knn = KNN_Manual(k=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)
acc_knn = np.mean(preds == y_test)
print(f" -> KNN Manual Accuracy: {acc_knn * 100:.2f}%")
with open(KNN_MODEL_PATH, 'wb') as f: pickle.dump(knn, f)

# 4. Train DNN
print("[INFO] Training DNN...")
num_classes = len(np.unique(y))
y_train_dl = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_dl = tf.keras.utils.to_categorical(y_test, num_classes)

model = models.Sequential([
    layers.Input(shape=(42,)), 
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_dl, epochs=100, batch_size=32, verbose=0)

loss, acc = model.evaluate(X_test, y_test_dl, verbose=0)
print(f" -> DNN Accuracy: {acc*100:.2f}%")
model.save(DNN_MODEL_PATH)
print(f"[INFO] Done. Models saved in '{MODELS_ROOT}' folder.")