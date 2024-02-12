import streamlit as st
from load_model import get_model, emotion_decoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model and tokenizer


@st.cache_resource
def load_saved_model():
    return get_model()


model, tokenizer = load_saved_model()


def predict_emotion(text: str) -> str:
    """
    Predict the emotion from the text
    :param text: input text
    :return: predicted emotion
    """
    max_length = 35  # used when training the model

    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    sequence_padded = pad_sequences(
        sequence, maxlen=max_length, padding='post')
    # Predict the emotion
    prediction = model.predict(sequence_padded)
    prediction_class = np.argmax(prediction)

    return emotion_decoder(prediction_class), max(prediction[0])


# Stramlit app


st.title("Emotika: Emotion Analysis App")
# add link to github
if st.button('View on GitHub'):
    st.markdown(
        '[GitHub Repository](https://github.com/abdulmunimjemal/emotika)')

st.write('Welcome to the Emotion Analysis App! Enter some text below to analyze the emotion.')

user_input = st.text_area("Input Text", "Type Here")
if st.button('Analyze Emotion'):
    if user_input:
        # Predict emotion
        emotion, confidence = predict_emotion(user_input)
        st.markdown(f'Predicted Emotion: **{emotion.capitalize()}**')
        st.write(f'Confidence: {round(confidence*100, 2)}%')

# Description
st.markdown("""
### Description
This is a simple web application for emotion analysis. It loads a trained model and tokenizer, accepts input text from the user, runs it through the model, and predicts the corresponding emotion.

It classifies input text in to six categories of emotions: Joy, Sadness, Anger, Fear, Love, and Surprise.
""")

# Footer
st.markdown('---')
st.write(
    'Made with ❤️ in Ethiopia by [Abdulmunim J Jemal](https://www.linkedin.com/in/abdulmunim-jemal/)')
