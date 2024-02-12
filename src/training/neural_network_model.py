import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensroflow.keras.layers import Dense, Embedding, LTSM, Dropout


def build_model(input_dim, output_dim):
    """
    Build and compile the neural network model.
    """
    model = Sequential()
    # Add embedding layer
    model.add(Embedding(input_dim, output_dim))
    # Add LTSM layer
    model.add(LTSM(128))
    # Add dropout layer
    model.add(Dropout(0.5))
    # Add output layer
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    """
    Train the neural network model.
    """
    history = model.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=batch_size, epochs=epochs, verbose=1)
    return history
