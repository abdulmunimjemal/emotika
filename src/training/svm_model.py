from sklearn.svm import SVC


def train_model(X_train, y_train, kernel='rbf'):
    """
    Train the Random Forest Algorithm
    """
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model
