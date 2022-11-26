import numpy as np
from matplotlib import pyplot as plt

from nearest_neighbour import learnknn, predictknn
from utils import gensmallm

LABELS = [2, 3, 5, 6]


def question_2_part_a(k, sample_sizes_steps):
    data = np.load('mnist_all.npz')
    train_sample_sizes = np.linspace(1, 100, num=sample_sizes_steps, dtype=int)
    repeats = 10

    avg_errors, min_errors, max_errors = [], [], []
    for sample_size in train_sample_sizes:
        avg_error, min_error, max_error = calculate_errors_with_repeats(data, sample_size, k, repeats)
        avg_errors.append(avg_error)
        min_errors.append(min_error)
        max_errors.append(max_error)

    avg_errors, min_errors, max_errors = np.array(avg_errors), np.array(min_errors), np.array(max_errors)
    avg_distance_min = avg_errors - min_errors
    avg_distance_max = max_errors - avg_errors
    mix_max_errors = np.vstack((avg_distance_min, avg_distance_max))
    show_results(x_axis=train_sample_sizes, y_axis=avg_errors, repeats=repeats, k=k, error_bar=mix_max_errors)


def calculate_errors_with_repeats(data, train_sample_size: int = 100, k: int = 1, repeats: int = 1):
    repeats_errors = [calculate_error(data, k=k, train_sample_size=train_sample_size) for _ in range(repeats)]
    return np.mean(repeats_errors), min(repeats_errors), max(repeats_errors)


def calculate_error(data, k: int, train_sample_size: int) -> float:
    train_data = [data[f'train{label}'] for label in LABELS]
    test_data = [data[f'test{label}'] for label in LABELS]
    test_size = sum(map(lambda test: test.shape[0], test_data))

    x_train, y_train = gensmallm(train_data, LABELS, train_sample_size)
    x_test, y_test = gensmallm(test_data, LABELS, test_size)

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
    question_2_part_a(k=1, sample_sizes_steps=10)
