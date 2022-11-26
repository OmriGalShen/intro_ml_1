from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


class Classifier:
    def __init__(self, k: int, x_train: np.array, y_train: np.array):
        self.k = k
        self.examples: List[Example] = [Example(*x_y_train) for x_y_train in zip(x_train, y_train)]


class Example:
    def __init__(self, x: np.array, label: float):
        self.x = x
        self.label = label


def learnknn(k: int, x_train: np.array, y_train: np.array) -> Classifier:
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return Classifier(k, x_train, y_train)


def predictknn(classifier: Classifier, x_test: np.array) -> np.array:
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    # apply classification for each example
    result_apply_classification = np.apply_along_axis(classify, 1, x_test, classifier)
    # fix shape from (n,) -> (n,1)
    result_fix_shape = np.expand_dims(result_apply_classification, axis=1)
    return result_fix_shape


def classify(x: np.array, classifier: Classifier) -> np.float:
    """
    :param classifier: data structure returned from the function learnknn
    :param x: numpy array of size (d, 1) containing test example that will be classified
    :return: classification 0-9
    """
    k: int = classifier.k
    dataset: List[Example] = classifier.examples
    nearest_neighbors = sorted(dataset, key=lambda data_point: distance.euclidean(x, data_point.x))
    k_nearest_neighbors = nearest_neighbors[:k]
    k_nearest_neighbors_labels = [neighbor.label for neighbor in k_nearest_neighbors]
    majority_vote_label = max(k_nearest_neighbors_labels)
    return majority_vote_label


def sanity_test():
    k = 1
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 0, 1])
    classifier = learnknn(k, x_train, y_train)
    x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    y_testprediction = predictknn(classifier, x_test)
    assert (y_testprediction == np.array([[1], [0], [0], [1]])).all()


def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def question_2_part_a(k, sample_sizes_count):
    data = np.load('mnist_all.npz')
    train_sample_sizes = np.linspace(1, 100, num=sample_sizes_count, dtype=int)
    repeats = 10

    avg_errors, min_errors, max_errors = [], [], []
    for sample_size in train_sample_sizes:
        avg_error, min_error, max_error = calculate_errors(data, sample_size, k, repeats)
        avg_errors.append(avg_error)
        min_errors.append(min_error)
        max_errors.append(max_error)

    avg_errors, min_errors, max_errors = np.array(avg_errors), np.array(min_errors), np.array(max_errors)
    avg_distance_min = avg_errors - min_errors
    avg_distance_max = max_errors - avg_errors
    mix_max_errors = np.vstack((avg_distance_min, avg_distance_max))
    show_results(x_axis=train_sample_sizes, y_axis=avg_errors, repeats=repeats, k=k, error_bar=mix_max_errors)


def calculate_errors(data, train_sample_size: int = 100, k: int = 1, repeats: int = 1):
    repeats_errors = [calculate_error(data, k=k, train_sample_size=train_sample_size) for _ in range(repeats)]
    return np.mean(repeats_errors), min(repeats_errors), max(repeats_errors)


def calculate_error(data, k: int, train_sample_size: int) -> float:
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], train_sample_size)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(k, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # fix shape from (n,) -> (n,1)
    y_test = np.expand_dims(y_test, axis=1)

    return float(np.mean(y_test != preds))


def show_results(x_axis, y_axis, repeats: int, k: int, error_bar):
    fig, ax = plt.subplots()
    ax.set_xlabel('sample size')
    ax.set_ylabel(f'mean error {repeats} repeats')
    ax.set_title(f"Question 2A (k={k})")
    plt.errorbar(x=x_axis, y=y_axis, yerr=error_bar, marker='o', ecolor='red', capsize=3)
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    sanity_test()
    simple_test()
    question_2_part_a(k=1, sample_sizes_count=10)
