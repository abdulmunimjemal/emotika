import pandas as pd
import re
import emoji
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split


def load_dataset(file_path: str):
    """
    Load the raw dataset from CSV
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print("Unable To read the file, Check the file Path")
        return pd.DataFrame()


def clean_text(text: str, remove_mentions: bool = True, replace_emoji=True):
    """
    Clean the text data by removing special chaarcters, URLs, and other noises.
    """
    if replace_emoji:
        text = emoji.demojize(text)

    if remove_mentions:
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    text = re.sub(r'http\S+', '', text)  # remove urls
    text = re.sub(r'[^\w\s]', '', text)  # remove special chars
    # Remove repetitions ( whaaat => what )
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    text = re.sub(pattern, r"\1", text)
    return text


def preprocess_text(text: str):
    """
    Preprocess the text data by tokenizing, case conversion, and stopword removal
    """
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens


def apply_stemming(tokens: list):
    """
    Apply stemming to reduce words to their base forms.
    """
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
