from sklearn.linear_model import Perceptron as SklearnPerceptron


class Perceptron:
    def __init__(self):
        """
        Initialises Perceptron classifier with initializing 
        weights, alpha(learning rate) and number of epochs.
        """
        self.w = None
        self.alpha = 0.01
        self.epochs = 300
        self.perceptron = SklearnPerceptron(alpha=self.alpha, max_iter=self.epochs, n_jobs=-1)

    def train(self, X_train, y_train):
        """
        Train the Perceptron classifier. Use the perceptron update rule
        as introduced in Lecture 3.

        Inputs:
        - X_train: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y_train: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.perceptron.fit(X=X_train, y=y_train)

    def predict(self, X_test):
        """
        Predict labels for test data using the trained weights.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        pred = self.perceptron.predict(X_test)
        return pred
