import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import numpy as np

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
import os

keras = tf.keras

use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

df = pd.read_excel('hotel_review.xlsx')
def sentiment_analysis_of_review(df):
    if os.path.exists('model.pkl'):
        model = keras.models.load_model('model.pkl')
        return model
    else:
        df["review"] = df["Negative_Review"].replace('No Negative', '') + ' ' + df["Positive_Review"].replace('No Positive',' ')
        df["review_type"] = df["Reviewer_Score"].apply(
            lambda x: "bad" if x < 6 else "good"
        )
        df = df[["review", "review_type"]]

        RANDOM_SEED = 42

        type_one_hot = OneHotEncoder(sparse=False).fit_transform(
            df.review_type.to_numpy().reshape(-1, 1)
        )

        X_train, X_test, y_train, y_test = train_test_split(
            df.review,
            type_one_hot,
            test_size=.1,
            random_state=RANDOM_SEED
        )

        X_train = np.array([tf.reshape(use(r), [-1]).numpy() for r in X_train])
        X_test = np.array([tf.reshape(use(r), [-1]).numpy() for r in X_test])


        model = keras.Sequential()

        model.add(
        keras.layers.Dense(
            units=256,
            input_shape=(X_train.shape[1], ),
            activation='relu'
        )
        )
        model.add(
        keras.layers.Dropout(rate=0.5)
        )
        model.add(
        keras.layers.Dense(
            units=128,
            activation='relu'
        )
        )
        model.add(
        keras.layers.Dropout(rate=0.5)
        )
        model.add(keras.layers.Dense(2, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(0.001),
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=512,
            validation_split=0.1,
            verbose=1,
            shuffle=True
        )

        model.evaluate(X_test, y_test)

        y_pred = model.predict(X_test)

        y_pred = ["Bad" if np.argmax(i) == 0 else "Good" for i in y_pred]

        model.save('model.pkl')

        return model
# Define the Streamlit app
def app():
    st.title("Hotel Review Sentiment Analysis")

    model = sentiment_analysis_of_review(df)
    # Create a text input for the user to input the text to be classified
    text_input = st.text_input("Enter the Review to be Classified:")
    if text_imput.strip() == '':
        st.write("")
    else :
        # Preprocess the input text
        review = use([text_input])
        review = np.array([tf.reshape(review, [-1]).numpy()])

        y_pred = model.predict(review)
        y_pred = ["Bad" if np.argmax(i) == 0 else "Good" for i in y_pred][0]

        # Display the prediction
        st.write("The sentiment of the review given is:", y_pred)


if __name__ == '__main__':
    app()
