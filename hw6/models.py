import copy
import math
import random
from typing import Callable, Optional

import numpy as np


def node_score_error(prob: float) -> float:
    """
    Calculates the node score using the train error of a subset of a dataset.
    With 2 classes, note that C(p) = min{p, 1-p}.

    Parameters
    ----------
    prob : float
        Float representing probability value.

    Returns
    -------
    float
        Float value representing node score.
    """
    # TODO: Add your solution code here.
    return min(prob, 1 - prob)


def node_score_entropy(prob: float) -> float:
    """
    Calculates the node score using the entropy of a subset of the data.
    With 2 classes, note that C(p) = -p * log(p) - (1-p) * log(1-p)
    For the purposes of this calculation, we adopt the convention 0*log0 = 0.

    Parameters
    ----------
    prob : float
        Float representing probability value.

    Returns
    -------
    float
        Float value representing node score.
    """
    # TODO: Add your solution code here.
    return (
        -prob * np.log(prob) - (1 - prob) * np.log(1 - prob)
        if prob > 0 and prob < 1
        else 0.0
    )


def node_score_gini(prob: float) -> float:
    """
    Calculate the node score using the Gini index of a subset of the data.
    With 2 classes, note that C(p) = 2 * p * (1-p)

    Parameters
    ----------
    prob : float
        Float representing probability value.

    Returns
    -------
    float
        Float value representing node score.
    """
    # TODO: Add your solution code here.
    return 2 * prob * (1 - prob)


class Node:
    """
    Class to define a node in the decision tree.
    """

    def __init__(
        self,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        depth: int = 0,
        index_split_on: int = 0,
        isleaf: bool = False,
        label: int = 1,
    ) -> None:
        """Constructor for a node.

        Attributes
        ----------
        left : Optional["Node"]
            Optional left child node, by default None
        right : Optional["Node"]
            Optional right child node, by default None
        depth : int, optional
            Indication of current node depth in tree, by default 0
        index_split_on : int, optional
            Indication of index to split on, by default 0
        isleaf : bool, optional
            Should be true if the current node is a leaf, by default False
        label : int, optional
            Label value for current node, by default 1
        info : dict
            Dictionary to track gain values for visualization
        """
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {}

    def _set_info(self, gain: float, num_samples: int) -> None:
        """
        Helper function to add to info attribute.
        You do not need to modify this.
        """

        self.info["gain"] = gain
        self.info["num_samples"] = num_samples


class DecisionTree:
    """Class defining a complete decision tree model"""

    def __init__(
        self,
        data: np.ndarray,
        validation_data: np.ndarray = None,
        gain_function: Callable = node_score_entropy,
        max_depth: int = 40,
    ) -> None:
        """Constructor

        Parameters
        ----------
        data : np.ndarray
            Training dataset
        validation_data : np.ndarray, optional
            Validation dataset, by default None
        gain_function : Callable, optional
            Function pointer to use as gain function, by default node_score_entropy
        max_depth : int, optional
            Max node depth allowed in this tree, by default 40
        """
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)

    def predict(self, features: np.ndarray) -> int:
        """
        Calculate prediction label given a row of features.
        You do not need to modify this.

        Parameters
        ----------
        features : np.ndarray
           Numpy array of a row of features.

        Returns
        -------
        int
            Predicted label value
        """
        return self._predict_recurs(self.root, features)

    def accuracy(self, data: np.ndarray) -> float:
        """
        Calculates the model accuracy on the given data. You do not need to modify this.

        Parameters
        ----------
        data : np.ndarray
           Provided dataset

        Returns
        -------
        float
            Accuracy value
        """
        return 1 - self.loss(data)

    def loss(self, data: np.ndarray) -> float:
        """
        Calculates loss on the given data. You do not need to modify this.

        Parameters
        ----------
        data : np.ndarray
           Provided dataset

        Returns
        -------
        float
            Loss value
        """
        labels = data[:, 0]
        errors = 0.0
        for i in range(data.shape[0]):
            errors += self.predict(data[i]) != labels[i]
        return errors / data.shape[0]

    def _predict_recurs(self, node: Node, row: np.ndarray) -> int:
        """
        Helper function to predict the label given a row of features by
        traversing the tree to its leaves to obtain the label. You do
        not need to modify this.

        Parameters
        ----------
        node: Node
            Current node
        row : np.ndarray
            Numpy array containing the row of features.

        Returns
        -------
        int
            Current node label or result of recursive call
        """
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)

    def _prune_recurs(self, node: Node, validation_data: np.ndarray) -> None:
        """
        Prunes the tree bottom up recursively. DO NOT prune if the node is a leaf
        or if the node is non-leaf and has at least one non-leaf child. On the other hand,
        DO prune if doing so could reduce loss on the validation data.

        NOTE: This might be slightly different from the pruning described in lecture.
        Here we won't consider pruning a node's parent if we don't prune the node
        itself (i.e. we will only prune nodes that have two leaves as children.)

        HINT: Think about what variables/class attributes need to be set when pruning
        a node!

        Parameters
        ----------
        node : Node
           Current node
        validation_data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m+1), the first column
           is 1 or 0 corresponding to label

        Returns
        -------
        None
        """
        # TODO: Add your solution code here.
        if node.isleaf == False:
            if not node.isleaf and node.left.isleaf and node.right.isleaf:
                old_loss = self.loss(validation_data)
                left = node.left
                right = node.right

                node.isleaf = True
                node.left = None
                node.right = None
                new_loss = self.loss(validation_data)

                if new_loss >= old_loss:
                    node.isleaf = False
                    node.left = left
                    node.right = right

    def _is_terminal(
        self, node: Node, data: np.ndarray, indices: list[int]
    ) -> tuple[bool, int]:
        """
        Determines whether or not the node should stop splitting. Stop the recursion
        in the following cases -
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node exceeds the maximum depth.

        Parameters
        ----------
        node : Node
           Current node
        data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m+1), the first column is 1 or 0 corresponding to label
        indices : list
            List of indices to split on

        Returns
        -------
        tuple[bool, int]
            A tuple consisting of a boolean in the first index and an integer label in the second.
            - For the boolean, True indicates that the passed Node should be a leaf; return False otherwise.
            - For the integer label, indicate the leaf label by majority or the label the passed Node would
              have if we were to terminate at it instead. If there is no data left (i.e., len(data) == 0),
              return a label randomly (see eg. numpy.random.choice())
        """
        # TODO: Add your solution code here.
        if len(data) == 0:
            return True, random.choice([0, 1])

        one_count = np.mean(data[:, 0])
        label = 1 if one_count > 0.5 else 0

        if label == 1:
            return True, label
        elif label == 0:
            return True, label
        elif node.depth == self.max_depth:
            return True, label
        elif len(indices) == 0:
            return True, label
        return False, label

    def _split_recurs(self, node: Node, data: np.ndarray, indices: list[int]) -> None:
        """
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        First, check if the node needs to be split (see _is_terminal()).  If so, find the
        optimal column to split on by maximizing information gain, ie, with _calc_gain().
        Store the label predicted for this node, the split column, and use _set_info().
        Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.

        Parameters
        ----------
        node : Node
           Current node
        data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m + 1), the first column is 1 or 0 corresponding to label
        indices : list
            List of indices to split on

        Returns
        -------
        None
        """
        # TODO: Add your solution code here.
        done, label = self._is_terminal(node, data, indices)
        node.label = label
        if done:
            node.isleaf = True
            return

        best_gain = 0
        best_index = None
        for idx in indices:
            gain = self._calc_gain(data, idx, self.gain_function)
            if gain > best_gain:
                best_gain = gain
                best_index = idx

        if best_gain == 0:
            node.isleaf = True
            return
        
        node.index_split_on = best_index
        indices.remove(best_index)

        left = data[data[:, best_index] == 0]
        right = data[data[:, best_index] == 1]

        node.left = Node(depth=node.depth + 1)
        node.right = Node(depth=node.depth + 1)
        self._split_recurs(node.left, left, indices)
        self._split_recurs(node.right, right, indices)

    def _calc_gain(
        self,
        data: np.ndarray,
        split_index: int,
        gain_function: Callable[[float], float],
    ) -> float:
        """
        Calculate the gain of the proposed splitting.

        Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])

        Here, C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and Gini functions.

        Parameters
        ----------
        data : np.ndarray
           Numpy array with size of n*(m_1 + m_2 + ... + m_m + 1), the first column is 1 or 0 corresponding to label
        split_index : int
            Int representing the index to split on.
        gain_function: function
            Function pointer to gain function (float to float) which calculates the node score. One of
            node_score_error, node_score_entropy, or node_score_gini at the top of this file.

        Returns
        -------
        float
            The gain value of the proposed splitting.
        """
        # TODO: Add your solution code here.
        y = np.array([row[0] for row in data])
        xi = np.array([row[split_index] for row in data])
        total_count = len(y)

        if len(y) != 0 and len(xi) != 0:
            class_one_count = np.count_nonzero(y)
            class_one_f = class_one_count/total_count

            x_one_count = np.count_nonzero(xi)
            x_one_f = x_one_count/total_count
            x_zero_count = total_count - x_one_count
            x_zero_f = x_zero_count/total_count
            c1x1_f = np.sum((xi == 1) & (y == 1))/x_one_count if x_one_count else 0
            c0x0_f = np.sum((xi == 0) & (y == 0))/x_zero_count if x_zero_count else 0

            gain = gain_function(class_one_f) - x_zero_f * gain_function(c0x0_f) - x_one_f * gain_function(c1x1_f)

        else:
            gain = 0.0
        return gain

    def print_tree(self) -> None:
        """
        Helper function for tree_visualization. Note that this is only useful
        for very shallow trees. You do not need to modify this.
        """
        print("---START PRINT TREE---")

        def print_subtree(node, indent=""):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = "split attribute = {:d}; gain = {:f}; number of samples = {:d}".format(
                    node.index_split_on, node.info["gain"], node.info["num_samples"]
                )
            left = indent + "0 -> " + print_subtree(node.left, indent + "\t\t")
            right = indent + "1 -> " + print_subtree(node.right, indent + "\t\t")
            return decision + "\n" + left + "\n" + right

        print(print_subtree(self.root))
        print("----END PRINT TREE---")

    def loss_plot_vec(self, data: np.ndarray) -> np.ndarray:
        """
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.

        Parameters
        ----------
        data : np.ndarray
           2D numpy array containing data

        Returns
        -------
        np.ndarray
            Numpy array containing the loss
        """
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info["curr_num_correct"]
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec) / len(data)

    def _loss_plot_recurs(
        self, node: Node, rows: np.ndarray, prev_num_correct: int
    ) -> None:
        """
        Visualization of the loss when the tree expands. You do not need to modify this.

        Parameters
        ----------
        node : Node
           Node
        rows: np.ndarray
            Numpy 2D array containg the rows of features
        prev_num_correct: int
            Int representing the previous number of correct entries.

        Returns
        -------
        None
        """
        labels = rows[:, 0]
        curr_num_correct = np.sum(labels == node.label) - prev_num_correct
        node.info["curr_num_correct"] = curr_num_correct

        if not node.isleaf:
            left_data = rows[rows[:, node.index_split_on] == 0]
            right_data = rows[rows[:, node.index_split_on] == 1]

            left_labels = left_data[:, 0]
            left_num_correct = np.sum(
                left_labels == node.label
            )
            right_labels = right_data[:, 0]
            right_num_correct = np.sum(right_labels == node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)