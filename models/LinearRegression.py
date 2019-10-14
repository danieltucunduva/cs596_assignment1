from sklearn.linear_model import SGDClassifier


class LinearRegression:
    def __init__(self):
        """
        Initialises LinearRegression classifier with initializing
        alpha(learning rate), number of epochs
        """
        self.alpha = 0.01
        self.epochs = 1000
        self.linear_model = SGDClassifier(learning_rate='adaptive', eta0=self.alpha,
                                                max_iter=self.epochs, loss='squared_loss', n_jobs=-1)

    def train(self, X_train, y_train):
        """
        Train Linear regression classifier
        """
        self.linear_model.fit(X=X_train, y=y_train)
    
    def predict(self, X_test):
        """
        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        pred = self.linear_model.predict(X=X_test)
        return pred
