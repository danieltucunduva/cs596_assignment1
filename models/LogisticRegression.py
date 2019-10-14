import torch
from torch.autograd import Variable
from sklearn.linear_model import SGDClassifier

# Logistic Regression Class (using Pytorch function)
# epochs (float, optional) – epochs  (default: 500)
# lr (float, optional) – learning rate (default: 0.001)
# momentum (float, optional) – momentum factor (default: 0.9)
# weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)

class LogisticRegressionM(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionM, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class LogisticRegression():
    #    def __init__(self,epochs = 6, lr=0.001,momentum=0.9, weight_decay=0):
    def __init__(self, epochs=6, lr=0.001, momentum=0.9, decay=0):
        # Hyperparameters
        self.momentum = momentum
        self.lr = lr
        self.epochs = epochs
        self.decay = decay
        self.logistic = LogisticRegressionM(3072, 10)

    def train(self, train_loader):
        criterion = torch.nn.CrossEntropyLoss()  # computes softmax and then the cross entropy
        #      optimizer = torch.optim.SGD(self.logistic.parameters(), lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
        optimizer = torch.optim.Adagrad(self.logistic.parameters(), lr=self.lr, lr_decay=self.decay)
        for epoch in range(int(self.epochs)):
            for i, (images, labels) in enumerate(train_loader):
                images = Variable(images.view(-1, 3 * 32 * 32))
                labels = Variable(labels)

                optimizer.zero_grad()
                outputs = self.logistic(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, test_loader):
        # calculate Accuracy
        pred = []
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images.view(-1, 3 * 32 * 32))
            outputs = self.logistic(images)
            _, predicted = torch.max(outputs.data, 1)
            pred.append(predicted)
            total += labels.size(0)
            # for gpu, bring the predicted and labels back to cpu fro python operations to work
            correct += (predicted == labels).sum()
        accuracy = 100 * correct / total
        return {"pred": pred, "accuracy": accuracy}


class LogisticRegressionSKL:
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


