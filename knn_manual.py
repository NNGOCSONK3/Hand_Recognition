import numpy as np
from collections import Counter

class KNN_Manual:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    # Hàm Training: Thực chất chỉ là lưu dữ liệu lại (Lazy Learning)
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Hàm Predict: Tính toán khoảng cách để tìm láng giềng
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # 1. Tự tính khoảng cách Euclidean (Toán học)
        distances = [np.sqrt(np.sum((x_train - x)**2)) for x_train in self.X_train]
        
        # 2. Tìm K điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # 3. Bầu chọn (Voting)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]