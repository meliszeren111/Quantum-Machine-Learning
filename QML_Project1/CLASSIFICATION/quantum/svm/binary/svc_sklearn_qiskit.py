from qiskit_machine_learning.algorithms import QSVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_wine, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

covtype = fetch_covtype()
df = pd.DataFrame(covtype.data, columns=covtype.feature_names)
df['target'] = covtype.target
df_filtered = df[df['target'].isin([1, 2])].sample(n=200, random_state=42)
X = df_filtered.iloc[:,0:2]
y = df_filtered['target']
print(" ================== TRAIN TEST SPLIT ================== ")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df = pd.DataFrame(X, columns=covtype.feature_names)
df['label'] = y
df.label.value_counts(normalize=True).round(3)
print(" ================== MINMAX SCALER ================== ")
min_max_scaler = MinMaxScaler()
scaled_X = min_max_scaler.fit_transform(X)
print(" ================== PCA ================== ")
pc_range = np.arange(1,scaled_X.shape[1] + 1)
pca = PCA(n_components=None)
pca.fit(scaled_X)
print(" ================== ZZFeatureMap ================== ")
feature_map = ZZFeatureMap(feature_dimension=2, reps=1, entanglement="linear")
sampler = Sampler()
print(" ================== QSVC MODEL ================== ")
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
svc = QSVC(quantum_kernel=kernel)
print(" ================== QSVC TRAINING ================== ")
svc.fit(x_train, y_train)
print(" ================== QSVC TESTING ================== ")
score = svc.score(x_test, y_test)
print(f"Testing accuracy: {score:.2f}")
print(" ================== QSVC PREDICTIONS ================== ")
y_pred = svc.predict(x_test)
print(" ================== QSVC TRAIN/TEST SPLIT PLOT ================== ")
plt.figure(figsize=(8,6))
plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train, cmap='coolwarm', label='Train', marker='o', alpha=0.6)
plt.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], c=y_test, cmap='coolwarm', label='Test', marker='x', alpha=0.8)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("Train/Test Split")
plt.legend()
plt.grid(True)
plt.show()
print(" ================== PCA PLOT ================== ")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled_X)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.6)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of covtype Dataset (Class 0 and 1)")
plt.colorbar(label="Class")
plt.grid(True)
plt.show()
print(" ================== ACCURACY PLOT ================== ")
accuracies = []
reps_range = range(1, 5)
print(" ================== ZZFEATURE MAP REPETITION PLOT ================== ")
for reps in reps_range:
    feature_map = ZZFeatureMap(feature_dimension=2, reps=reps, entanglement="linear")
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    svc = QSVC(quantum_kernel=kernel)
    svc.fit(x_train, y_train)
    acc = svc.score(x_test, y_test)
    accuracies.append(acc)
plt.plot(reps_range, accuracies, marker='o')
plt.xlabel("ZZFeatureMap Reps")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Feature Map Reps")
plt.grid(True)
print(" ================== SVM CONFUSION MATRIX PLOT ================== ")
cm_binary = confusion_matrix(y_test, y_pred, labels=svc.classes_)
disp_cm_binary = ConfusionMatrixDisplay(confusion_matrix=cm_binary, display_labels=svc.classes_)
disp_cm_binary.plot(cmap="Oranges")
plt.show()



