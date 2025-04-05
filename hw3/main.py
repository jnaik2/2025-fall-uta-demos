"""
This file contains the main program to read data, run the classifier,
and print results.

You do not need to make edits to this file, but you may feel free to
do so. This code should not run by default in your final submission.

Brown cs1420, Spring 2025
"""

from pathlib import Path

import numpy as np
from models import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_FILE_NAME = 'normalized_data.csv'
# DATA_FILE_NAME = "unnormalized_data.csv"
# DATA_FILE_NAME = 'normalized_data_nosens.csv'

CENSUS_FILE_PATH = Path().resolve() / "data" / DATA_FILE_NAME

NUM_CLASSES = 3
BATCH_SIZE = 1  # TODO: tune this parameter
CONV_THRESHOLD = 1  # TODO: tune this parameter


def import_census(file_path: Path) -> tuple[np.ndarray, ...]:
    """Help function to import the census dataset. Uses sklearn's test/train split.

    Parameters
    ----------
    file_path : Path
        File path to census data set

    Returns
    -------
    tuple[np.ndarray, ...]
        Tuple containing training data inputs, training data
        labels, testing data inputs, and testing data labels in
        that order
    """
    data = np.genfromtxt(file_path, delimiter=",", skip_header=False)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0
    )
    return X_train, Y_train, X_test, Y_test


def test_logreg() -> float:
    """Runs the model training and test loop on the census dataset.

    Returns
    -------
    float
        Returns model accuracy
    """
    X_train, Y_train, X_test, Y_test = import_census(CENSUS_FILE_PATH)
    num_features = X_train.shape[1]

    # Add a bias
    X_train_b = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_test_b = np.append(X_test, np.ones((len(X_test), 1)), axis=1)

    ### Logistic Regression ###
    model = LogisticRegression(num_features, NUM_CLASSES, BATCH_SIZE, CONV_THRESHOLD)
    num_epochs = model.train(X_train_b, Y_train)
    acc = model.accuracy(X_test_b, Y_test) * 100
    print(f"Test Accuracy: {acc}%")
    print(f"Number of Epochs: {num_epochs}")

    return acc


def main() -> None:
    """
    Main driving function.
    """
    # Set random seeds. DO NOT CHANGE THIS IN YOUR FINAL SUBMISSION.
    # NOTE: random.seed(0)
    np.random.seed(0)

    test_logreg()

    # Calculate average accuracies over five different random seeds.
    # accuracies = []
    # for i in range(5):
    #     random.seed(i)
    #     np.random.seed(i)
    #     accuracies.append(test_logreg())
    #
    # print("Average accuracy for " + DATA_FILE_NAME +":")
    # print(np.mean(accuracies))


if __name__ == "__main__":
    main()