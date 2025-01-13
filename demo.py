import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
# สมมุติว่าคุณมีไฟล์ "diabetes.csv" ที่มีข้อมูล
data = pd.read_csv("diabetes.csv")

# 2. Visualizing the data
print(data.describe())  # สถิติเบื้องต้น
print(data.isnull().sum())  # ตรวจสอบข้อมูลที่หายไป (Missing data)

# 3. Handling missing data (ถ้ามี)
# แทนค่า missing data ด้วยค่าเฉลี่ย (ตัวอย่าง)
data.fillna(data.mean(), inplace=True)

# 4. Visualizing the data อีกครั้ง (กราฟตัวอย่าง)
sns.pairplot(data, hue="Outcome")
plt.show()

# 5. Selecting features and splitting the data
X = data.drop("Outcome", axis=1)  # Features (ตัวแปรอิสระ)
y = data["Outcome"]  # Target (ผลลัพธ์)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Train the SVM model with different kernels
kernels = ["linear", "poly", "rbf"]  # Linear, Polynomial, RBF kernels
for kernel in kernels:
    print(f"Training SVM with {kernel} kernel")
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    # 7. Test the SVM model
    y_pred = model.predict(X_test)

    # 8. Evaluate the model
    print(f"Kernel: {kernel}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# 9. Visualizing SVM margin (for 2D features only)
# ตัวอย่างแสดง margin ถ้าใช้แค่ 2 features
X_vis = X_train.iloc[:, :2]  # ใช้แค่ 2 feature แรก
model_vis = SVC(kernel="linear")
model_vis.fit(X_vis, y_train)

# สร้าง plot สำหรับ SVM
def plot_svm_boundary(X, y, model):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors="k", marker="o")
    plt.title("SVM Decision Boundary")
    plt.show()

plot_svm_boundary(X_vis, y_train, model_vis)

# 10. รายงานผล
print("Finished training and evaluation.")
