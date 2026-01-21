from qiskit.circuit.library import TwoLocal, ZZFeatureMap
from qiskit_algorithms.optimizers import ADAM, COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.circuit.library import RawFeatureVector
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import MinMaxScaler
# https://github.com/qiskit-community/qiskit-community-tutorials/blob/master/machine_learning/vqc_feature_map_comparison.ipynb
wine = load_wine()
X = wine.data
y = wine.target
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
df_filtered = df[df['target'].isin([0, 1])]
X = df_filtered.drop('target', axis=1)
y = df_filtered['target']
df = df_filtered
print(" ================== WINE DATASET LABELS ================== ")
print(y)
print(" ================== WINE DATASET FEATURES ================== ")
print(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" ================== WINE TRAIN DATASET FEATURE DESCRIPTION ================== ")
print(x_train.describe())
print(" ================== WINE TRAIN DATASET LABEL DESCRIPTION ================== ")
print(y_train.describe())
print(" ================== WINE TEST DATASET FEATURE DESCRIPTION ================== ")
print(x_test.describe())
print(" ================== WINE TEST DATASET LABEL DESCRIPTION ================== ")
print(y_test.describe())
print(" ================== WINE TRAIN DATASET FEATURE INFO ================== ")
print(x_train.info())
print(" ================== WINE TRAIN DATASET LABEL INFO ================== ")
print(y_train.info())
print(" ================== WINE TEST DATASET FEATURE INFO ================== ")
print(x_test.info())
print(" ================== WINE TEST DATASET LABEL INFO ================== ")
print(y_test.info())
print(" ================== WINE DATASET DESCRIPTION ================== ")
dataset_description = wine.DESCR
print(dataset_description)
df = pd.DataFrame(X, columns=wine.feature_names)
df['label'] = y
np.bincount(df["label"])
df.label.value_counts(normalize=True).round(3)
dataset_info = df.info()
msno.bar(df)
print(" ================== WINE DATASET HEATMAP ================== ")
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(df.iloc[:,:-1].corr(), annot=True, cmap="Reds")
print(" ================== WINE DATASET PAIRPLOT ================== ")
sns.pairplot(df, hue='label',  palette="tab10",  corner=True)
print(" ================== VALUE COUNTS PLOT ================== ")
categorical_columns = [col for col in df.columns if df[col].nunique() <= 20]
n_cols = 3
n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()
for i, col in enumerate(categorical_columns):
    df[col].value_counts().sort_index().plot(kind="bar", color="skyblue", ax=axes[i])
    axes[i].set_title(f"Value Counts - {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()
print(" ================== MINMAX SCALER ================== ")
minmax = MinMaxScaler()
X_sc = minmax.fit_transform(X)
print(" ================== PCA ================== ")
pc_range = np.arange(1,X_sc.shape[1] + 1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sc)
print(" ================== TRAIN TEST SPLIT ================== ")
x_train, x_test, y_train, y_test = split(X_pca, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)
print(" ================== MINMAX SCALER ================== ")
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

input_shape = (x_train.shape[1],)
num_classes = len(np.unique(y_train))
n_features = 2
y_train = np.array(y_train)
raw_feature_map = RawFeatureVector(feature_dimension=n_features)
raw_feature_map_optimizer = COBYLA(maxiter=100)
zz_feature_map = ZZFeatureMap(feature_dimension=n_features)
zz_feature_map_optimizer = ADAM(maxiter=100)
ansatz1 = TwoLocal(raw_feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3).decompose()
ansatz2 = TwoLocal(zz_feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3).decompose()

vqc1 = VQC(feature_map=raw_feature_map, ansatz=ansatz1, loss="cross_entropy", optimizer=raw_feature_map_optimizer)
vqc1.fit(x_train, y_train)
print(f'Results with RawFeatureVector: train set accuracy -> {vqc1.score(x_train, y_train)}, test set accuracy -> {vqc1.score(x_test, y_test)}.')
vqc2 = VQC(feature_map=zz_feature_map, ansatz=ansatz2, loss="cross_entropy", optimizer=zz_feature_map_optimizer)
vqc2.fit(x_train, y_train)
print(f'Results with ZZFeatureMap: train set accuracy -> {vqc2.score(x_train, y_train)}, test set accuracy -> {vqc2.score(x_test, y_test)}.')
ansatz1.draw("mpl", style="clifford")
plt.show()
ansatz2.draw("mpl", style="clifford")
plt.show()
X = x_train
y = y_train
h = .02  # step size
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = vqc1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z).reshape(xx.shape)
plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap="coolwarm")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("VQC Decision Boundary for Raw Feature Map")
plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 4))
Z = vqc2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap="coolwarm")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("VQC Decision Boundary for ZZ Feature Map")
plt.tight_layout()
plt.show()
