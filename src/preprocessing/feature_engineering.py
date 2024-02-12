from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import FastText


def bow_representation(data: str):
    """
    Generate Bag of Words representation of the data
    """
    vectorizer = CountVectorizer()
    # fit the vectorizer on the data
    bow_matrix = vectorizer.fit_transform(data)
    return bow_matric


def tfidf_representation(data: str):
    """
    Generate TF-IDF representation of the text data
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix


def word_embeddings(data: data, embedding_size=100, window=5, min_count=5, epochs=5):
    """
    Generate word embeddings representation of the text data
    """
    model = FastText(sentences=data, vector_size=embedding_size,
                     window=window, min_count=min_count, epochs=epochs)
    word_vectors = model.wv
    return word_vectors
