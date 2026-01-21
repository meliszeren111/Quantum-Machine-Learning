import pandas as pd
from sklearn.datasets import fetch_covtype
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = fetch_covtype(as_frame=True)
X = data.data
y = data.target
pca = PCA(n_components=2)
X = pca.fit_transform(X)
df = pd.DataFrame(X, columns=["component1", "component2"])
df['label'] = y
df = df[(df["label"] == 1) | (df["label"] == 2)]
y = df["label"]
X = df.drop(columns=["label"])
print(" ================== DESCRIPTION OF X ================== ")
print(X.describe())
print(" ================== DESCRIPTION OF Y ================== ")
print(y.describe())
print(" ================== VALUE COUNTS OF Y ================== ")
print(y.value_counts().sort_index())
print(" ================== INFORMATION ABOUT X DATA ================== ")
print(X.info())
print(" ================== INFORMATION ABOUT Y DATA ================== ")
print(y.info())
print(" ================== DESCRIPTION OF ENTIRE DATA ================== ")
print(data.DESCR)
plt.figure(figsize=(8,5))
sns.countplot(x=y, palette='Reds')
plt.title("Cover Type Class Distribution")
plt.xlabel("Class")
plt.ylabel("Size")
plt.show()

X.iloc[:, :10].hist(figsize=(15, 10), bins=30)
plt.suptitle("Distribution of 2 principle components")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(X.iloc[:, :10].corr(), annot=True, cmap='Greens')
plt.title("Correlation between 2 principle components")
plt.show()

