import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle


def get_model() -> list:
    """
    Load the model and tokenizer
    :return: model and tokenizer
    """
    # Load the model
    model = load_model('models/NN_model.h5')
    tokenizer = pickle.load(open('models/NN_tokenizer.pickle', 'rb'))
    return model, tokenizer


def emotion_decoder(emotion: int) -> str:
    """
    Decode the emotion from the integer value
    :param emotion: integer value of the emotion
    :return: string value of the emotion
    """
    emotions_map = {
        0: 'joy',
        1: 'sadness',
        2: 'anger',
        3: 'fear',
        4: 'love',
        5: 'surprise',
    }
    return emotions_map[emotion]
