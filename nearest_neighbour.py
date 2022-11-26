import numpy as np
from scipy.spatial import distance

from utils import gensmallm


class Classifier:
    def __init__(self, k: int, x_train: np.array, y_train: np.array):
        self.k = k
        self.examples = list(zip(x_train, y_train))


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
    examples: np.array = classifier.examples
    nearest_neighbors = sorted(examples, key=lambda example: distance.euclidean(x, example[0]))
    k_nearest_neighbors = nearest_neighbors[:k]
    k_nearest_neighbors_labels = [neighbor[1] for neighbor in k_nearest_neighbors]
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


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    sanity_test()
    simple_test()
