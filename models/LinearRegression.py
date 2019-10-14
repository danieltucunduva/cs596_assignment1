import numpy as np
import torch
from torch.autograd import Variable
from sklearn.linear_model import SGDClassifier

# Pytorch Linear Model Class
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(3072, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    # Linear Regression Class (using Pytorch function)


# epochs (float, optional) – epochs  (default: 500)
# lr (float, optional) – learning rate (default: 0.001)
# momentum (float, optional) – momentum factor (default: 0.9)
# weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
# x_scale (float, optional) – Since the maximum number of R,G,B is 255, I devide every X data by 255 to make the value of X at the same scale with Y (default: 255)

class LinearRegression():
    #    def __init__(self,epochs = 500, lr=0.001,momentum=0.9, weight_decay=0, x_scale=255):
    def __init__(self, epochs=500, lr=0.001, momentum=0.9, decay=0, x_scale=255):
        # Hyperparameters
        self.momentum = momentum
        self.lr = lr
        self.epochs = epochs
        self.decay = decay
        self.x_scale = x_scale

        # our model
        self.linear = LinearRegressionModel()

    def train(self, X_train, y_train):
        # prepare the data for training.
        X = X_train.copy()
        X = np.array(X, np.float32) / self.x_scale
        x_data = Variable(torch.Tensor(X))
        y_data = Variable(torch.Tensor(y_train))

        criterion = torch.nn.MSELoss(size_average=True)
        # We also tried other loss function such as L1Loss
        # criterion = torch.nn.L1Loss()

        #      optimizer = torch.optim.SGD(self.linear.parameters(), lr=self.lr,momentum=self.momentum,weight_decay=self.decay)
        optimizer = torch.optim.Adagrad(self.linear.parameters(), lr=self.lr, lr_decay=self.decay)
        for epoch in range(self.epochs):
            # Forward pass: Compute predicted y by passing
            # x to the model
            pred_y = self.linear(x_data)

            # Compute and print loss
            loss = criterion(pred_y, y_data)

            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X_test):
        # prepare the data for training.
        XV = X_test.copy()
        XV = np.array(XV, np.float32) / self.x_scale
        new_var = Variable(torch.Tensor(XV))
        pred = self.linear(new_var)
        return pred


class LinearRegressionSKL:
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
