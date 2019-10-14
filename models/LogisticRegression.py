from sklearn.linear_model import SGDClassifier


class LogisticRegression:
    def __init__(self):
        """
        Initialises LogisticRegression classifier
        """
        self.alpha = 0.01
        self.epochs = 500
        self.reg_const = 0.01
        self.logistic_regression = SGDClassifier(learning_rate='adaptive', eta0=self.alpha, alpha=self.reg_const,
                                                 max_iter=self.epochs, loss='log', n_jobs=-1)

    def train(self, X_train, y_train):
        """
        Train Logistic regression classifier
        """
        self.logistic_regression.fit(X=X_train, y=y_train)

    def predict(self, X_test):
        """
        Predict labels for data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        pred = self.logistic_regression.predict(X_test)
        return pred
