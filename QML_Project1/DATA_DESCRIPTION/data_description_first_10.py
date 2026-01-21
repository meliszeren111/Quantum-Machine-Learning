
from sklearn.datasets import fetch_covtype
import seaborn as sns
import matplotlib.pyplot as plt

data = fetch_covtype(as_frame=True)
X = data.data
y = data.target
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
sns.countplot(x=y, palette='Greens')
plt.title("Cover Type Class Distribution")
plt.xlabel("Class")
plt.ylabel("Size")
plt.show()

X.iloc[:, :10].hist(figsize=(15, 10), bins=30)
plt.suptitle("Distribution of first 10 Cover Type variables")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(X.iloc[:, :10].corr(), annot=True, cmap='Blues')
plt.title("Correlation between first 10 Cover Type Variables")
plt.show()

