from sklearn.linear_model import SGDClassifier


class Softmax:
    def __init__(self):
        """
        Initialises Softmax classifier with initializing 
        weights, alpha(learning rate), number of epochs
        and regularization constant.
        """
        self.w = None
        self.alpha = 0.01
        self.epochs = 1000
        self.reg_const = 0.01
        self.softmax = self.svm = SGDClassifier(learning_rate='adaptive', eta0=self.alpha, alpha=self.reg_const,
                                                max_iter=self.epochs, penalty='l2',
                                                loss='log', n_jobs=-1)

    def train(self, X_train, y_train):
        """
        Train Softmax classifier using stochastic gradient descent.

        Inputs:
        - X_train: A numpy array of shape (N, D) containing training data;
        N examples with D dimensions
        - y_train: A numpy array of shape (N,) containing training labels;
        
        Hint : Operate with Minibatches of the data for SGD
        """
        self.softmax.fit(X=X_train, y=y_train)

    def predict(self, X_test):
        """
        Use the trained weights of softmax classifier to predict labels for
        data points.

        Inputs:
        - X_test: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        pred = self.softmax.predict(X=X_test)
        return pred
