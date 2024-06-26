

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:07:45 2024

@author: Movement Rehab Lab
"""

import numpy as np

def fill_matrix(vector, shape=(7, 5), indices=None):
    """
    Fills the vector with a matrix of the specified shape at the given index.

    Args:
    vector
    shape (7, 5)
    indices 10-20 system

    Returns:
    numpy.ndarray: modified matrix
    """
    if indices is None:
        # 默认的填充位置索引
        indices = [
            (0, 1), (0, 3), 
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (2, 3),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
            (4, 1), (4, 2), (4, 3),
            (5,0),(5,1),(5,2),(5,3),(5,4),
            (6,1),(6,2),(6,3)
        ]
    
    # 初始化矩阵
    matrix = np.zeros(shape, dtype=float)

    # 使用循环将向量填充到矩阵中指定位置
    for idx, val in zip(indices, vector):
        matrix[idx] = val

    return matrix

# 测试 fill_matrix 函数
test_vector = np.arange(1, 27)
test_matrix = fill_matrix(test_vector)
print(test_matrix)

# 定义一个 72*4*26 的样本集合
sample_set = np.load('~/demo_dataset.npy')
sample_labels = np.load('~/sample_labels.npy')

# 初始化一个空的列表用于存储填充后的矩阵
filled_matrices_list = []

# 对每个样本进行处理
for sample in sample_set:
    filled_sample = []
    # 对样本中的每个向量进行填充并添加到列表中
    for vec in sample:
        filled_matrix = fill_matrix(vec)
        filled_sample.append(filled_matrix)
    filled_matrices_list.append(filled_sample)
# 打印填充后的矩阵列表长度
print(len(filled_matrices_list))
filled_matrices_list=np.array(filled_matrices_list)

from keras.models import Sequential
from keras.layers import Conv3D, Flatten, Dense
from keras.regularizers import l2


def create_3d_cnn(input_shape):
    model = Sequential()
    model.add(Conv3D(128, kernel_size=(2, 2, 2), activation='relu',input_shape=input_shape, kernel_regularizer=l2(0.005)))
    model.add(Conv3D(128, kernel_size=(2, 2, 2),  activation='relu',kernel_regularizer=l2(0.005)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3个类别的输出
    
    return model

# 样本形状为4*7*5
input_shape = (4, 7, 5, 1)  # 最后一个维度1表示单通道灰度图像
model = create_3d_cnn(input_shape)

np.random.seed(42)  # 设置随机种子，确保每次运行得到相同的随机结果
shuffle_indices = np.random.permutation(len(filled_matrices_list))
filled_matrices_list_shuffled = filled_matrices_list[shuffle_indices]
sample_labels_shuffled = sample_labels[shuffle_indices]-1



from keras.optimizers import Nadam
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
# Binarize labels
n_classes = len(np.unique(sample_labels_shuffled))
binarized_true_labels = label_binarize(sample_labels_shuffled, classes=np.arange(n_classes))


predicted_labels=[]
accuracies = []
sensitivities = []
specificities = []
f1_scores = []
kappa_scores = []

# Initialize lists to store tpr and fpr for each fold
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
y_scores=[]
y_tests= []

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5)

# Iterate over each fold
for i, (train_index, test_index) in enumerate(kf.split(filled_matrices_list_shuffled, sample_labels_shuffled)):
    X_train, X_test = filled_matrices_list_shuffled[train_index], filled_matrices_list_shuffled[test_index]
    y_train, y_test = binarized_true_labels[train_index], binarized_true_labels[test_index]
    
    model = create_3d_cnn(input_shape=(4, 7, 5, 1))
    optimizer = Nadam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) # use binary crossentropy for binary labels
    model.fit(X_train, y_train, epochs=100, batch_size=36, verbose=0)
    
    # Compute ROC curve and area under the curve
    y_score = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    y_scores.append(y_score)
    y_tests.append(y_test)
    aucs.append(roc_auc)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    
    y_pred = np.argmax(y_score, axis=1)  # Convert predicted probabilities to class labels
    predicted_labels.append(y_pred) 
    
    # Compute evaluation metrics
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    sensitivity = recall_score(np.argmax(y_test, axis=1), y_pred, average='macro')
    specificity = recall_score(np.argmax(y_test, axis=1), y_pred, average='macro', labels=[0, 2])
    f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='macro')
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    
    accuracies.append(accuracy)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    f1_scores.append(f1)
    kappa_scores.append(kappa)
   
# Plot random guessing
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guessing')

# Plot mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='r', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2)

# Plot settings
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Mean of 5 Folds')
plt.legend(loc='lower right')
plt.show()

print("Average Accuracy:", np.mean(accuracies))
print("Average Sensitivity:", np.mean(sensitivities))
print("Average Specificity:", np.mean(specificities))
print("Average F1 Score:", np.mean(f1_scores))
print("Average Kappa Score:", np.mean(kappa_scores))

y_tests = np.concatenate(y_tests)
y_scores = np.concatenate(y_scores)



# from sklearn.metrics import roc_auc_score

# micro_roc_auc_ovr = roc_auc_score(
#     y_tests,
#     y_scores,
#     multi_class="ovr",
#     average="micro",
# )

# print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{micro_roc_auc_ovr:.3f}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
y_tests2 =  np.argmax(y_tests,axis=1)
predicted_labels = np.concatenate(predicted_labels)
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_tests2, predicted_labels)

# 绘制热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=False, cmap='Blues', fmt='d', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

from pycm import *

actual_vector= y_tests2
predict_vector=predicted_labels
cm = ConfusionMatrix(actual_vector= actual_vector, predict_vector= predict_vector)
print(cm)


