import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle


def get_model(model="NN_model.h5", tokenizer="NN_tokenizer.pickle") -> list:
    """
    Load the model and tokenizer
    :return: model and tokenizer
    """
    # Load the model
    model = load_model(f'models/{model}')
    tokenizer = pickle.load(open(f'models/{tokenizer}', 'rb'))
    return model, tokenizer


def emotion_decoder(emotion: int, include_emojis: bool = True) -> str:
    """
    Decode the emotion from the integer value
    :param emotion: integer value of the emotion
    :return: string value of the emotion
    """
    emotions_map = {
        0: 'Joy',
        1: 'Sadness',
        2: 'Anger',
        3: 'Fear',
        4: 'Love',
        5: 'Surprise',
    }

    emotions_map_emojis = {
        0: 'Joyful 😊',
        1: 'Sad 😢',
        2: 'Angry 😠',
        3: 'Fear 😨',
        4: 'Love it 😍',
        5: 'Wow! Surprise 😲',
    }
    return emotions_map[emotion] if not include_emojis else emotions_map_emojis[emotion]
