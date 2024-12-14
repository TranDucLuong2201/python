from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Tải dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target

# 2. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Chuẩn hóa dữ liệu (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Khởi tạo mô hình KNN với k=5
knn = KNeighborsClassifier(n_neighbors=5)

# 5. Huấn luyện mô hình
knn.fit(X_train, y_train)

# 6. Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# 7. Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình:", accuracy)
print("Báo cáo phân loại:\n", classification_report(y_test, y_pred))

# 8. Dự đoán một mẫu mới
new_sample = [[5.0, 3.5, 1.5, 0.2]]
new_sample_scaled = scaler.transform(new_sample)
predicted_class = knn.predict(new_sample_scaled)
print("Lớp dự đoán cho mẫu mới:", iris.target_names[predicted_class[0]])
