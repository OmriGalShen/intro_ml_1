import random

import numpy as np
from matplotlib import pyplot as plt

from nearest_neighbour import learnknn, predictknn
from utils import gensmallm

LABELS = [2, 3, 5, 6]
CORRUPTED_FACTOR = 0.15


def run_knn_over_sample_size(k: int, sample_size_max: int, sample_sizes_steps: int, repeats: int, title: str):
    print(f"Running knn with k={k}, sample_size_max={sample_size_max} "
          f"sample_sizes_steps={sample_sizes_steps}, repeats={repeats}"
          f"...")
    train_sample_sizes = np.linspace(1, sample_size_max, num=sample_sizes_steps, dtype=int)

    avg_errors, min_errors, max_errors = [], [], []
    for sample_size in train_sample_sizes:
        print(f"sample_size: {sample_size}")
        avg_error, min_error, max_error = calculate_errors_with_repeats(sample_size, k, repeats, is_corrupted=False)
        avg_errors.append(avg_error)
        min_errors.append(min_error)
        max_errors.append(max_error)

    error_bar = calculate_error_bar(avg_errors, min_errors, max_errors)
    show_results(x_axis=train_sample_sizes, y_axis=np.array(avg_errors), repeats=repeats, error_bar=error_bar,
                 title=title, x_label='sample size')
    print("done!")


def run_knn_over_k(max_k: int, sample_size, repeats: int, title: str, is_corrupted):
    print(f"Running knn with max_k={max_k}, "
          f"sample_size={sample_size}, repeats={repeats}"
          f"...")

    k_values = np.linspace(1, max_k, num=max_k, dtype=int)
    avg_errors, min_errors, max_errors = [], [], []
    for k in k_values:
        print(f"k: {k}")
        avg_error, min_error, max_error = calculate_errors_with_repeats(sample_size, k, repeats, is_corrupted)
        avg_errors.append(avg_error)
        min_errors.append(min_error)
        max_errors.append(max_error)

    error_bar = calculate_error_bar(avg_errors, min_errors, max_errors)
    show_results(x_axis=k_values, y_axis=np.array(avg_errors), repeats=repeats, error_bar=error_bar,
                 title=title, x_label='k')
    print("done!")


def calculate_error_bar(avg_errors, min_errors, max_errors):
    avg_errors, min_errors, max_errors = np.array(avg_errors), np.array(min_errors), np.array(max_errors)
    avg_distance_min = avg_errors - min_errors
    avg_distance_max = max_errors - avg_errors
    return np.vstack((avg_distance_min, avg_distance_max))


def calculate_errors_with_repeats(train_sample_size: int, k: int, repeats: int, is_corrupted):
    repeats_errors = [calculate_error(k, train_sample_size, is_corrupted) for _ in range(repeats)]
    return np.mean(repeats_errors), min(repeats_errors), max(repeats_errors)


def calculate_error(k: int, train_sample_size: int, is_corrupted) -> float:
    train_data = [data[f'train{label}'] for label in LABELS]
    test_data = [data[f'test{label}'] for label in LABELS]
    test_size = sum(map(lambda test: test.shape[0], test_data))

    x_train, y_train = gensmallm(train_data, LABELS, train_sample_size)
    x_test, y_test = gensmallm(test_data, LABELS, test_size)

    if is_corrupted:
        corrupt(y_train)
        corrupt(y_test)

    classifier = learnknn(k, x_train, y_train)
    preds = predictknn(classifier, x_test)
    # fix shape from (n,) -> (n,1)
    y_test = np.expand_dims(y_test, axis=1)
    return float(np.mean(y_test != preds))


def corrupt(y):
    train_corrupted_indices = random.sample(range(len(y)), int(CORRUPTED_FACTOR * len(y)))
    for train_corrupted_index in train_corrupted_indices:
        label_to_change = y[train_corrupted_index]
        y[train_corrupted_index] = random.choice([label for label in LABELS if label != label_to_change])


def show_results(x_axis, y_axis, repeats: int, error_bar, title: str, x_label: str):
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(f'mean error {repeats} repeats')
    ax.set_title(f"{title}")
    plt.errorbar(x=x_axis, y=y_axis, yerr=error_bar, marker='o', ecolor='red', capsize=3)
    plt.show()


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    data = np.load('mnist_all.npz')
    # question 2a
    run_knn_over_sample_size(title="Question 2A (k=1)",
                             k=1,
                             sample_size_max=100,
                             sample_sizes_steps=10,
                             repeats=10)
    # question 2e
    run_knn_over_k(title="Question 2e (sample=200)",
                   max_k=11,
                   sample_size=200,
                   repeats=10,
                   is_corrupted=False)

    # question 2f
    run_knn_over_k(title="Question 2f (sample=200)",
                   max_k=11,
                   sample_size=200,
                   repeats=10,
                   is_corrupted=True)
