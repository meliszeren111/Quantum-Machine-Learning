import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_covtype
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = fetch_covtype(as_frame=True)
X = data.data
y = data.target
X = StandardScaler().fit_transform(X)
X = PCA(n_components=2).fit_transform(X)
df = pd.DataFrame(X, columns=["component1", "component2"])
df['label'] = y
df = df[df["label"].isin([1, 2])]
df["label"] = df["label"].replace({1: 0, 2: 1})
y = df["label"].values
X = df.drop(columns=["label"]).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
n_qubits = 2
n_layers = 6
n_parameters = 3
rotation_angle_for_1_st_qubit = 0.1
rotation_angle_for_2_nd_qubit = 0.2
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    inputs = tf.reshape(inputs, [-1])
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))
classical_rotation_angles = ([rotation_angle_for_1_st_qubit, rotation_angle_for_2_nd_qubit])
classical_weight_data = np.random.randn(n_layers, n_qubits, n_parameters)
fig, ax = qml.draw_mpl(quantum_circuit)(inputs=classical_rotation_angles, weights=classical_weight_data)
fig.show()
weight_shapes = {"weights": (n_layers, n_qubits, n_parameters)}
quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    quantum_layer,
    tf.keras.layers.Activation('sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=3, batch_size=1, validation_data=(X_test, y_test))
pd.DataFrame(history.history)[["accuracy", "val_accuracy"]].plot(title="Accuracy")
pd.DataFrame(history.history)[["loss", "val_loss"]].plot(title="Loss")
plt.show()
