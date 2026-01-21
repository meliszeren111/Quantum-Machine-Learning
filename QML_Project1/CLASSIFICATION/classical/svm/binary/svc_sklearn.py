from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score as accuracy, multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
df = df.sample(n=10000, random_state=42)
y = df["label"]
X = df.drop(columns=["label"])
print(" ================== CROSS VALIDATION ================== ")
crossval = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
parameters = [
    {"kernel": ["linear"], "C": [0.1, 1]},
    {"kernel": ["rbf"], "C": [1], "gamma": [0.1, 1]}
]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
print(" ================== SVM ================== ")
model = SVC()
print(" ================== SVM PARAMETER SEARCH ================== ")
classifier = GridSearchCV(estimator=model, param_grid=parameters, cv=crossval, verbose=2, n_jobs=-1)
classifier.fit(X_train, y_train)
print(" ================== SVM RESULTS ================== ")
scores = pd.DataFrame(classifier.cv_results_)
scores = scores.sort_values(by="rank_test_score").set_index("rank_test_score")
int_cols = ["param_C", "param_kernel", "param_degree", "param_gamma", "mean_test_score"]
print(" ================== SVM BEST ESTIMATOR ================== ")
best_model = classifier.best_estimator_
print(" ================== SVM PREDICTIONS ================== ")
predictions = best_model.predict(X_test)
round(accuracy(y_test, predictions), 3)
pred_train = classifier.best_estimator_.predict(X_train)
pred_test = classifier.best_estimator_.predict(X_test)
print(f"Best parameters are: {classifier.best_params_}, with a score of {round(classifier.best_score_,3)}")
print(f"Accuracy on training set is: {round(accuracy(y_train, pred_train), 3)}")
print(f"Accuracy on test set is : {round(accuracy(y_test, pred_test), 3)}")
print(classification_report(y_test, pred_test, labels=[0,1,2]))
print(" ================== SVM CONFUSION MATRIX ================== ")
cm_binary = confusion_matrix(y_test, predictions, labels=classifier.classes_)
disp_cm_binary = ConfusionMatrixDisplay(confusion_matrix=cm_binary, display_labels=classifier.classes_)
disp_cm_binary.plot(cmap="Oranges")
cm_multi = multilabel_confusion_matrix(y_test, predictions, labels=classifier.classes_)
for idx, (label, matrix) in enumerate(zip(classifier.classes_, cm_multi)):
    plt.figure(figsize=(4, 3))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for class '{label}'")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
plt.show()

