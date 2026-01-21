from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02_neural_network_classifier_and_regressor.html
data = fetch_covtype(as_frame=True)
X = data.data
y = data.target
X = StandardScaler().fit_transform(X)
X = PCA(n_components=2).fit_transform(X)
df = pd.DataFrame(X, columns=["component1", "component2"])
df['label'] = y
df = df[df["label"]]
df["label"] = df["label"]
y = df["label"].values
X = df.drop(columns=["label"]).values
print(" ================== TRAIN TEST SPLIT ================== ")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df['label'] = y
df.label.value_counts(normalize=True).round(3)
print(" ================== MINMAX SCALER ================== ")
min_max_scaler = MinMaxScaler()
scaled_X = min_max_scaler.fit_transform(X)
print(" ================== PCA ================== ")
pc_range = np.arange(1,scaled_X.shape[1] + 1)
pca = PCA(n_components=None)
pca.fit(scaled_X)
print(" ================== QNN CIRCUIT ================== ")
num_inputs = 13
qc = QNNCircuit(ansatz=RealAmplitudes(num_inputs, reps=1))
output_shape = 13
def parity(x):
    return "{:b}".format(x).count("1") % 2
print(" ================== QNN SAMPLER ================== ")
sampler = Sampler()
sampler_qnn = SamplerQNN(
    circuit=qc,
    interpret=parity,
    output_shape=output_shape,
    sampler=sampler,
)
print(" ================== QNN CLASSIFIER ================== ")
losses = []
def callback_func(weights, loss):
    losses.append(loss)
nn_classifier = NeuralNetworkClassifier(neural_network=sampler_qnn, optimizer=COBYLA(maxiter=30), callback=callback_func)
plt.rcParams["figure.figsize"] = (12, 6)
nn_classifier.fit(x_train, y_train)
plt.rcParams["figure.figsize"] = (6, 4)
nn_classifier.score(x_test, y_test)
print(" ================== LOSS PLOT ================== ")
plt.figure(figsize=(6, 4))
plt.plot(losses, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid(True)
plt.show()
plt.close()
circuit_fig = qc.draw("mpl", style="clifford")
circuit_fig.show()
print(" ================== PREDICTION PLOT ================== ")
y_pred = nn_classifier.predict(x_test)
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

