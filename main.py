import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import zscore

# Load dataset
data = pd.read_csv("diabetes.csv")  # เปลี่ยนชื่อไฟล์ตามจริง


# 1. Visualizing the data to find missing data

print("Descriptive Statistics:")
print(data.describe())

# ตรวจหาค่า 0 ในคอลัมน์ที่ไม่ควรมีค่า 0
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    print(f"{col} missing (0 values): {sum(data[col] == 0)}")


# 2. Handling the missing data

# แทนค่า 0 ด้วยค่าเฉลี่ยในคอลัมน์ที่กำหนด
for col in cols_with_zero:
    data[col].replace(0, data[col].mean(), inplace=True)


# 3. Visualizing the data (descriptive statistics)

# Distribution plots
for col in data.columns[:-1]:  # ไม่รวม Outcome
    sns.histplot(data[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# 4. Selecting the features for training the model

X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]              # Target


# 5. Split the data into train and test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 6. Train SVM and visualize the SVM margin with different kernels

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

kernels = ["linear", "poly", "rbf"]
models = {}

# ฟังก์ชันสำหรับการแสดงผล decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

# เทรน SVM และแสดง decision boundary สำหรับแต่ละ kernel
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train_2d, y_train)
    models[kernel] = model
    plot_decision_boundary(X_train_2d, y_train, model, f"SVM Decision Boundary ({kernel} Kernel)")


# 7. Test SVM

for kernel, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Kernel: {kernel}")
    print("Accuracy:", accuracy_score(y_test, y_pred))


# 8. Evaluate the SVM model with scikit-learn's evaluation metrics

for kernel, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Kernel: {kernel}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# 9. Report the findings

print("Best performing kernel:")
best_kernel = max(models, key=lambda k: accuracy_score(y_test, models[k].predict(X_test)))
print(f"Kernel: {best_kernel}, Accuracy: {accuracy_score(y_test, models[best_kernel].predict(X_test))}")


# 10. Optional: Manage the outliers

z_scores = np.abs(zscore(data))
threshold = 3  # ค่าเกณฑ์สำหรับ outliers
outliers = np.where(z_scores > threshold)
print(f"Number of outliers: {len(outliers[0])}")

# outliers
data_no_outliers = data[(z_scores < threshold).all(axis=1)]
print(f"Data shape after removing outliers: {data_no_outliers.shape}")
