from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.datasets import fetch_covtype
import pandas as pd
data = fetch_covtype(as_frame=True)
X = data.data
y = data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
df = pd.DataFrame(X, columns=["component1", "component2"])
df['label'] = y
df = df[(df["label"] == 1) | (df["label"] == 2)]
df["label"] = df["label"].replace({1: 0, 2: 1})
y = df["label"]
X = df.drop(columns=["label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
print(" ================== MINMAX SCALER ================== ")
input_shape = (X_train.shape[1],)
output_shape = 1
print(" ================== TENSORFLOW NN MODEL ================== ")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_shape, activation='sigmoid')
])
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256, epochs=10)
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
df = pd.DataFrame({"accuracy": accuracy, "loss": loss})
val_df = pd.DataFrame({"val_accuracy": val_accuracy, "val_loss": val_loss})
df.plot(title="Train Accuracy/Loss")
plt.show()
val_df.plot(title="Validation Accuracy/Loss")
plt.show()
