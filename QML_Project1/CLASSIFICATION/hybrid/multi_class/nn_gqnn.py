from GQNN.models import QuantumClassifier_EstimatorQNN_CPU
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = fetch_covtype(as_frame=True)
X = data.data
y = data.target
df = X.copy()
df['target'] = y
X = df.drop('target', axis=1)
y = df['target']
minmax = MinMaxScaler()
X_sc = minmax.fit_transform(X)
pc_range = np.arange(1,X_sc.shape[1] + 1)
pca_2d = PCA(n_components=2)
x_pca_2d = pca_2d.fit_transform(X_sc)
x_train, x_test, y_train, y_test = train_test_split(x_pca_2d, y, test_size=0.2, random_state=42)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
print(" ================== PCA ================== ")
model = QuantumClassifier_EstimatorQNN_CPU(num_qubits=2, maxiter=60)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
adjusted_score = 1 - score
print(f"Model accuracy (adjusted): {adjusted_score * 100:.2f}%")
y_pred = model.predict(x_test)
y_pred = np.round(y_pred).astype(int)
print(" ================== PREDICTION PLOT ================== ")
plt.figure()
for x_row, y_target, y_p in zip(X.to_numpy(), y, y_pred):
    if y_target == 1:
        plt.plot(x_row[0], x_row[1], "bo")
    else:
        plt.plot(x_row[0], x_row[1], "go")
    if y_target != y_p:
        plt.scatter(x_row[0], x_row[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
plt.title("Classification Results")
plt.show()
print(" ================== CONFUSION MATRIX PLOT ================== ")
cm_binary = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
disp_cm_binary = ConfusionMatrixDisplay(confusion_matrix=cm_binary, display_labels=np.unique(y_test))
disp_cm_binary.plot(cmap="Oranges")
plt.title("Confusion Matrix")
plt.show()

