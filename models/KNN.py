from sklearn.neighbors import KNeighborsClassifier


class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.classifier = None
        self.k = k

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.classifier = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1)
        self.classifier.fit(X, y)

    def predict(self, X_test):
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        return self.classifier.predict(X_test)
