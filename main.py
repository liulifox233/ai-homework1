# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "scikit-learn",
# ]
# ///
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data
y_true = iris.target

sse = []
silhouette = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    if k >= 2:
        silhouette.append(silhouette_score(X, kmeans.labels_))
    else:
        silhouette.append(None)

plt.rcParams['font.sans-serif'] = ['Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False
dpi = 300

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, sse, 'bo-')
plt.xlabel('K值')
plt.ylabel('SSE')
plt.title('肘部法则分析')

# 绘制轮廓系数图（从K=1开始）
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette[1:], 'go-')
plt.xlabel('K值')
plt.ylabel('轮廓系数')
plt.title('轮廓系数分析')
plt.tight_layout()
plt.savefig('k_analysis.png', dpi=dpi)
plt.close()

# 选择最佳K值
best_k = 3

# 应用最佳K值聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# 计算评估指标
ari = adjusted_rand_score(y_true, y_pred)

# 标签对齐处理混淆矩阵
def adjust_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = dict(zip(col_ind, row_ind))
    return np.array([label_mapping[x] for x in y_pred])

y_pred_adjusted = adjust_labels(y_true, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred_adjusted)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=dpi)
plt.close()

# 可视化聚类结果（PCA降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('聚类结果可视化（PCA降维）')
plt.savefig('cluster_visualization.png', dpi=dpi)
plt.close()

# 打印关键指标
print(f"最佳K值选择: {best_k}")
print(f"调整兰德指数: {ari:.3f}")
print("混淆矩阵已保存为 confusion_matrix.png")
print("分析图表已保存为 k_analysis.png 和 cluster_visualization.png")
