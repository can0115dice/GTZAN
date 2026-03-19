import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('features_3_sec.csv')
print(df.head())

df = pd.read_csv('features_3_sec.csv')

# 初步检查
print("数据集形状:", df.shape)

print("前五行数据预览:")
print(df.head())

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum().sum())

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# 1. 移除无关列
# filename 是字符串，length 是常量（都是30000），对模型分类没有意义
cols_to_drop = ['filename', 'length']
data = df.drop(columns=cols_to_drop)

# 2. 对标签(Label)进行数字编码
# 将 'blues', 'rock' 等转换为 0, 1, 2...
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# 打印编码对照关系，方便后续结果分析
label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
print("流派编码对照表:", label_mapping)

# 3. 准备特征矩阵 X 和 目标向量 y
X = data.drop(columns=['label'])
y = data['label']

# 1. 划分训练集和测试集 (80% 训练, 20% 测试)
# random_state=42
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 特征标准化 (Feature Scaling)
# SVM 和神经网络对数值范围非常敏感
scaler = StandardScaler()

# 仅在训练集上拟合(fit)，然后在训练集和测试集上转换(transform)
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"预处理完成！")
print(f"训练集维度: {X_train.shape}, 测试集维度: {X_test.shape}")

# ==========================================
# Exploratory Data Analysis (EDA)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# 类别分布图 (Class Distribution) ---
# 确认每个流派的样本量是否均衡
plt.figure(figsize=(10, 5))
sns.countplot(x=df['label'], hue=df['label'], palette='viridis', legend=False)
plt.title('Distribution of Music Genres', fontsize=15)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 特征相关性热图 (Correlation Heatmap) ---
# 检查特征之间是否存在多重共线性 (Multicollinearity)
plt.figure(figsize=(16, 10))
# 只取前 20 个特征进行展示，否则图片会太拥挤
corr = data.iloc[:, :20].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Audio Features (Top 20)', fontsize=15)
plt.show()

# 关键特征箱线图 (Boxplots for Key Features) ---
# 检查这些列是否存在于 df 中，防止报错
key_features = ['tempo', 'spectral_centroid_mean', 'chroma_stft_mean', 'rms_mean']
available_features = [f for f in key_features if f in df.columns]

if available_features:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for i, feature in enumerate(available_features):
        sns.boxplot(x='label', y=feature, hue='label', data=df, ax=axes[i], palette='Set3', legend=False)
        axes[i].set_title(f'{feature} Distribution by Genre', fontsize=14)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# 均值分布密度图 (Density Plot) ---
plt.figure(figsize=(10, 6))
if 'spectral_centroid_mean' in df.columns:
    sns.kdeplot(data=df, x='spectral_centroid_mean', hue='label',
                fill=True, common_norm=False, palette='tab10')
    plt.title('Density Plot of Spectral Centroid Mean by Genre', fontsize=15)
    plt.show()

import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC

# 创建模型保存目录
import os
os.environ["JOBLIB_TEMP_FOLDER"] = "E:\DATA"
if not os.path.exists('models'):
    os.makedirs('models')

# --- 5.1 Support Vector Machines (SVM) ---
print("\n[1/3] 正在运行 SVM 网格搜索 (GridSearchCV)...")

# 定义参数网格
params_svm = {
    "cls__C": [1, 10, 50],
    "cls__kernel": ['rbf', 'poly'],
    "cls__gamma": ['scale', 'auto']
}

# 这里的 Pipeline 不需要再次 StandardScaler，因为你之前的步骤已经做过了
pipe_svm = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.1)), # 移除方差过小的特征
    ('cls', SVC(probability=True))
])

# 使用 cv=5 进行交叉验证
grid_svm = GridSearchCV(pipe_svm, params_svm, scoring='accuracy', n_jobs=-1, cv=5, verbose=1)
grid_svm.fit(X_train, y_train)

# 评估 SVM
preds_svm = grid_svm.predict(X_test)
print(f"SVM 验证集最佳准确率: {grid_svm.best_score_:.4f}")
print(f"SVM 测试集准确率: {accuracy_score(y_test, preds_svm):.4f}")
joblib.dump(grid_svm, "models/pipe_svm.joblib")

# 评估 SVM
preds_svm = grid_svm.predict(X_test)
print(f"SVM 验证集最佳准确率: {grid_svm.best_score_:.4f}")
print(f"SVM 测试集准确率: {accuracy_score(y_test, preds_svm):.4f}")
joblib.dump(grid_svm, "models/pipe_svm.joblib")


# --- 5.2 XGBoost (Extreme Gradient Boosting) ---
print("\n[2/3] 正在运行 XGBoost 网格搜索...")

params_xgb = {
    "cls__max_depth": [4, 6],
    "cls__n_estimators": [100, 200],
    "cls__learning_rate": [0.05, 0.1]
}

pipe_xgb = Pipeline([
    ('cls', xgb.XGBClassifier(objective='multi:softmax', num_class=10, verbosity=0))
])

grid_xgb = GridSearchCV(pipe_xgb, params_xgb, scoring='accuracy', n_jobs=-1, cv=3, verbose=1)
grid_xgb.fit(X_train, y_train)

preds_xgb = grid_xgb.predict(X_test)
print(f"XGBoost Best accuracy on the validation set: {grid_xgb.best_score_:.4f}")
print(f"XGBoost Accuracy of the test set: {accuracy_score(y_test, preds_xgb):.4f}")
joblib.dump(grid_xgb, "models/pipe_xgb.joblib")

# LightGBM (快速基准模型) ---
print("\n[3/3] 正在训练 LightGBM...")

# LightGBM 通常在默认参数下就有很好的表现
model_lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1)
model_lgbm.fit(X_train, y_train)

preds_lgbm = model_lgbm.predict(X_test)
print(f"LightGBM 测试集准确率: {accuracy_score(y_test, preds_lgbm):.4f}")
joblib.dump(model_lgbm, "models/model_lgbm.joblib")


# ==========================================
# 6. Performance Analysis (全模型详细评估)
# ==========================================
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 结果汇总对比表
print("\n" + "=" * 40)
print("模型最终表现总览 (Final Accuracy Summary)")
print("-" * 40)
results = {
    "SVM": accuracy_score(y_test, preds_svm),
    "XGBoost": accuracy_score(y_test, preds_xgb),
    "LightGBM": accuracy_score(y_test, preds_lgbm)
}

# 按照准确率从高到低排序打印
for model_name, acc in sorted(results.items(), key=lambda item: item[1], reverse=True):
    print(f"{model_name:12}: {acc:.4f}")
print("=" * 40)

# 2. 循环输出每个模型的详细报告和混淆矩阵
all_preds = {
    "SVM": preds_svm,
    "XGBoost": preds_xgb,
    "LightGBM": preds_lgbm
}

for model_name, preds in all_preds.items():
    print(f"\n\n" + "#" * 50)
    print(f"### {model_name} 详细评估报告 ###")
    print("#" * 50)

    # 打印 Precision, Recall, F1-score 等
    print(classification_report(y_test, preds, target_names=le.classes_))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix: {model_name} (Acc: {results[model_name]:.4f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# 3. 额外加分项：最佳模型总结
best_model = max(results, key=results.get)
print(f"\n结论：表现最佳的模型是 {best_model}，准确率为 {results[best_model]:.4f}")