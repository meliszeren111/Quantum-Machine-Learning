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
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=10)
X = pca.fit_transform(X)
df = pd.DataFrame(X)
df['label'] = y
y = df["label"]
X = df.drop(columns=["label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
n_qubits = 10
n_layers = 10
n_parameters = 3
n_2_nd_layer_nodes = 8
n_5_th_layer_nodes = 4
n_last_layer_nodes = 7
rotation_angle_for_1_st_qubit = 0.1
rotation_angle_for_2_nd_qubit = 0.2
rotation_angle_for_3_rd_qubit = 0.1
rotation_angle_for_4_th_qubit = 0.3
rotation_angle_for_5_th_qubit = 0.5
rotation_angle_for_6_th_qubit = 0.2
rotation_angle_for_7_th_qubit = 0.1
rotation_angle_for_8_th_qubit = 0.8
rotation_angle_for_9_th_qubit = 0.7
rotation_angle_for_10_th_qubit = 0.1
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
weight_shapes = {"weights": (n_layers, n_qubits, n_parameters)}
quantum_layer1 = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
quantum_layer2 = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
quantum_layer3 = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
quantum_layer4 = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
quantum_layer5 = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qubits)
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(n_qubits,)),
    quantum_layer1,
    quantum_layer2,
    quantum_layer3,
    quantum_layer4,
    quantum_layer5,
    tf.keras.layers.Dense(n_last_layer_nodes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
input_shape = X_train.shape[1:] # input shape: (batch_size, n)
history = model.fit(X_train, y_train, epochs=3, batch_size=2)
accuracy = history.history["accuracy"]
loss = history.history["loss"]
df = pd.DataFrame({"accuracy": accuracy, "loss": loss})
df.plot(title="Train Accuracy/Loss")
plt.show()
classical_rotation_angles = ([rotation_angle_for_1_st_qubit, rotation_angle_for_2_nd_qubit, rotation_angle_for_3_rd_qubit, rotation_angle_for_4_th_qubit, rotation_angle_for_5_th_qubit, rotation_angle_for_6_th_qubit, rotation_angle_for_7_th_qubit, rotation_angle_for_8_th_qubit, rotation_angle_for_9_th_qubit, rotation_angle_for_10_th_qubit])
classical_weight_data = (n_layers, n_qubits, n_parameters)
fig, ax = qml.draw_mpl(quantum_circuit)(inputs=classical_rotation_angles, weights=classical_weight_data)
fig.show()



