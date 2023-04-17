import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from joblib import dump

iris_data = load_iris()

X = iris_data['data']
y = iris_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)

clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(confusion_matrix(y_test, y_pred))
print(iris_data['target_names'][y_pred])

dump(clf, 'iris_model.joblib')
tree.plot_tree(clf)
plt.show()
