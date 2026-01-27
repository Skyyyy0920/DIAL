import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# 读取数据
scn_data = scipy.io.loadmat(r'W:\Brain Analysis\data\PPMI\scnHCPD.mat')
fcn_data = scipy.io.loadmat(r'W:\Brain Analysis\data\PPMI\fcnHCPD.mat')
lab_data = scipy.io.loadmat(r'W:\Brain Analysis\data\PPMI\labHCPD.mat')

# 提取数据
scn_corr = scn_data['scn_corr']  # (88, 1)
fcn_corr = fcn_data['fcn_corr']  # (88, 1)
labels = lab_data['labels'].flatten()  # 转为1维数组 (88,)

# 将每个样本的矩阵提取出来
scn_matrices = np.array([scn_corr[i, 0] for i in range(len(scn_corr))])  # (88, 90, 90)
fcn_matrices = np.array([fcn_corr[i, 0] for i in range(len(fcn_corr))])  # (88, 90, 90)

print(f"原始数据形状:")
print(f"  SCN: {scn_matrices.shape}")
print(f"  FCN: {fcn_matrices.shape}")
print(f"  Labels: {labels.shape}")
print(f"  标签分布 - 类别0: {np.sum(labels == 0)}, 类别1: {np.sum(labels == 1)}")

# 按6:4分割，保持正负样本比例
scn_train, scn_test, fcn_train, fcn_test, y_train, y_test = train_test_split(
    scn_matrices,
    fcn_matrices,
    labels,
    test_size=0.4,
    random_state=42,
    stratify=labels  # 保持正负样本比例
)

# 打印分割后的信息
print("\n训练集:")
print(f"  样本数: {len(y_train)}")
print(f"  标签分布 - 类别0: {np.sum(y_train == 0)}, 类别1: {np.sum(y_train == 1)}")
print(f"  比例: {np.sum(y_train == 0) / len(y_train):.2%} : {np.sum(y_train == 1) / len(y_train):.2%}")

print("\n测试集:")
print(f"  样本数: {len(y_test)}")
print(f"  标签分布 - 类别0: {np.sum(y_test == 0)}, 类别1: {np.sum(y_test == 1)}")
print(f"  比例: {np.sum(y_test == 0) / len(y_test):.2%} : {np.sum(y_test == 1) / len(y_test):.2%}")

# 保存为pickle格式
train_dict = {
    'scn': scn_train,
    'fcn': fcn_train,
    'labels': y_train
}

test_dict = {
    'scn': scn_test,
    'fcn': fcn_test,
    'labels': y_test
}

# 保存训练集
with open(r'W:\Brain Analysis\data\PPMI\train_data.pkl', 'wb') as f:
    pickle.dump(train_dict, f)

# 保存测试集
with open(r'W:\Brain Analysis\data\PPMI\test_data.pkl', 'wb') as f:
    pickle.dump(test_dict, f)
