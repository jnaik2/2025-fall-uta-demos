"""
This file implements a Logistic Regression classifier

Brown cs1420, Spring 2025
"""

import random

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Calculates element-wise softmax of the input array

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Softmax output of the given array x
    """
    e = np.exp(x - np.max(x))
    return (e + 1e-6) / (np.sum(e) + 1e-6)


class LogisticRegression:
    """
    Multiclass logistic regression model that learns weights using
    stochastic gradient descent (SGD).
    """

    def __init__(
        self, n_features: int, n_classes: int, batch_size: int, conv_threshold: float
    ) -> None:
        """Constructor for a LogisticRegression classifier instance

        Parameters
        ----------
        n_features : int
            The number of features in the classification problem
        n_classes : int
            The number of classes in the classification problem
        batch_size : int
            Batch size to use in SGD
        conv_threshold : float
            Convergence threshold; once reached, discontinues the optimization loop

        Attributes
        ----------
        alpha : int
            The learning rate used in SGD
        weights : np.ndarray
            Model weights
        """
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros(
            (n_classes, n_features + 1)
        )  # NOTE: An extra row added for the bias
        self.alpha = 0.03
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X: np.ndarray, Y: np.ndarray) -> int:
        """This implements the main training loop for the model, optimized
        using stochastic gradient descent.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array containing the datasets. Each row corresponds to one example, and
            each column corresponds to one feature. Padded by 1 column for the bias term.
        Y : np.ndarray
            A 1D Numpy array containing the labels corresponding to each example.

        Returns
        -------
        int
            Number of epochs taken to converge
        """
        # TODO: Add your solution code here.
        n = X.shape[0]
        all_idxs = list(range(n))
        done = False
        loss_vals = []
        epoch = 0

        while not done:
            epoch += 1
            random.shuffle(all_idxs)

            for start in range(0, n, self.batch_size):
                batch = all_idxs[start:start + self.batch_size]
                n_prime = len(batch)
                x_batch = X[batch]
                y_batch = Y[batch]

                grad = np.zeros_like(self.weights)

            for i in range(len(batch)):
                x_i = x_batch[i]
                y_i = y_batch[i]
                scores = np.dot(self.weights, x_i)
                probs = softmax(scores) 

            for c in range(self.n_classes):
                expected = 1 if y_i == c else 0
                grad[c] += (probs[c] - expected) * x_i

            self.weights -= self.alpha * grad / n_prime

            current_loss = self.loss(X, Y)
            loss_vals.append(current_loss)

            if len(loss_vals) > 1:
                if abs(loss_vals[-1] - loss_vals[-2]) < self.conv_threshold:
                    done = True

        return epoch

    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Calculates average log loss on the predictions made by the model
        on dataset X against the corresponding labels Y.

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array representing a dataset. Each row corresponds to one example,
            and each column corresponds to one feature. Padded by 1 column for the bias.
        Y : np.ndarray
            1D Numpy array containing the corresponding labels to each example in dataset X.

        Returns
        -------
        float
            Average loss of the model on the dataset
        """
        # TODO: Add your solution code here.
        total_loss = 0.0
        for i in range(X.shape[0]):
            for j in range(self.n_classes):
                if Y[i] == j:
                    x = X[i]
                    y = Y[i]
                    probs = softmax(np.dot(self.weights, x) )
                    total_loss += -np.log(probs[y]) 
                else: 
                    total_loss += 0.0
        return total_loss / X.shape[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute predictions based on the learned parameters and examples X

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array representing a dataset. Each row corresponds to one example,
            and each column corresponds to one feature. Padded by 1 column for the bias.

        Returns
        -------
        np.ndarray
            1D Numpy array of predictions corresponding to each example in X
        """
        # TODO: Add your solution code here.
        prob = softmax(np.dot(self.weights, X.T))
        pred = np.argmax(prob, axis=0)
        return pred

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Outputs the accuracy of the trained model on a given test
        dataset X and labels Y.

        Parameters
        ----------
        X : np.ndarray
            A 2D Numpy array representing a dataset. Each row corresponds to one example,
            and each column corresponds to one feature. Padded by 1 column for the bias.
        Y : np.ndarray
            1D Numpy array containing the corresponding labels to each example in dataset X.

        Returns
        -------
        float
            Accuracy percentage (between 0 and 1) on the given test set.
        """
        # TODO: Add your solution code here.
        pred = self.predict(X)
        return np.mean(pred == Y)