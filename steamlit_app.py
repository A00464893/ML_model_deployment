import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
import os

# Load the Iris dataset
data = load_iris()
if os.path.exists('iris_model.joblib'):
    clf = load('iris_model.joblib')
else:
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='class')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)
    # Train a decision tree classifier on the data
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    dump(clf, 'iris_model.joblib')


# Define the Streamlit app
def app():
    # Set the app title
    st.title("Iris Classifier")

    # Create a form for the user to input data
    sepal_length = st.slider("Sepal length", 0.0, 10.0, 5.0)
    sepal_width = st.slider("Sepal width", 0.0, 10.0, 5.0)
    petal_length = st.slider("Petal length", 0.0, 10.0, 5.0)
    petal_width = st.slider("Petal width", 0.0, 10.0, 5.0)

    # Make a prediction based on the user's input
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Show the prediction to the user
    st.write(f"Predicted class: {data.target_names[prediction[0]]}")

# Run the app
if __name__ == "__main__":
    app()
