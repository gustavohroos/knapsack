import numpy as np
from itertools import product
import time
import os
import argparse
import pandas as pd


def calculate_value(weights, values, max_weight, combination):
    weight = np.sum(combination * weights)
    value = np.sum(combination * values)

    if weight > max_weight:
        return 0, 0

    return weight, value


def read_file(filename):
    weights = []
    values = []

    if filename.split('_')[1] < '25':
        instance_size = 'low'
    else:
        instance_size = 'large'

    try:
        with open(f'instances_01_KP/{instance_size}/{filename}') as f:
            header = f.readline()
            max_weight = int(header.split()[1])
            file = f.readlines()
            if instance_size == 'large':
                file = file[:-1]

        for line in file:
            w, v = line.split()
            weights.append(int(w))
            values.append(int(v))

        with open(f"instances_01_KP/{instance_size}-optimum/{filename}") as f:
            optimal_value = int(f.readline())

    except:
        print('File not found')
        exit()

    return np.array(weights), np.array(values), max_weight, optimal_value


def bruteforce(W, weights, values):
    max_value = 0
    final_weight = 0
    best_items = np.zeros(len(weights), dtype=int)
    start = time.time()

    for combination in product([0, 1], repeat=len(weights)):
        weight, value = calculate_value(
            weights, values, W, np.array(combination))

        if value > max_value:
            final_weight = weight
            max_value = value
            best_items = np.array(combination)

    return time.time() - start, max_value, final_weight, best_items


def dynamic(W, weights, values):
    time_start = time.time()
    N = len(values)
    dp = [[0 for _ in range(W + 1)] for _ in range(N + 1)]

    for i in range(1, N + 1):
        for j in range(1, W + 1):
            if weights[i-1] > j:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(values[i-1] + dp[i-1]
                               [j-weights[i-1]], dp[i-1][j])

    value, weight = dp[N][W], W
    return time.time() - time_start, value, weight, dp[N][W]


def greedy(W, weights, values):
    time_start = time.time()
    N = len(values)

    # sorting
    value_per_weight = values / weights
    indexes = np.argsort(value_per_weight)[::-1]
    values = values[indexes]
    weights = weights[indexes]

    remainingW = W
    value = 0
    j_ = 0
    knapsack = np.zeros(N)
    for j in range(N):
        if weights[j] > remainingW:
            knapsack[j] = 0
        else:
            knapsack[j] = 1
            remainingW = remainingW - weights[j]
            value = value + values[j]
        if values[j] > values[j_]:
            j_ = j
    if values[j_] > value:
        value = values[j_]
        for j in range(N):
            knapsack[j] = 0
            knapsack[j_] = 1

    return time.time() - time_start, value, W - remainingW, knapsack


def fptas(W, weights, values, epsilon):
    N = len(values)
    max_value = np.max(values)
    scaling_factor = (epsilon * max_value) / N
    scaled_values = np.ceil(values / scaling_factor)
    # scaled_capacity = int(np.ceil(W / scaling_factor))

    return dynamic(W, weights, scaled_values)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--file', type=str)
    args.add_argument('--algorithm', type=str)

    args = args.parse_args()

    print('Running file: ', args.file)
    print('Running algorithm: ', args.algorithm)

    weights, values, max_weight, max_value = read_file(args.file)

    if args.algorithm == 'bruteforce':
        total_time, value, final_weight, best_items = bruteforce(
            max_weight, weights, values)
    elif args.algorithm == 'dynamic':
        total_time, value, final_weight, best_items = dynamic(
            max_weight, weights, values)
    elif args.algorithm == 'greedy':
        total_time, value, final_weight, best_items = greedy(
            max_weight, weights, values)
    elif args.algorithm == 'fptas':
        total_time, value, final_weight, best_items = fptas(
            max_weight, weights, values, 0.1)
    else:
        print('Invalid algorithm')
        exit()

    print('Max weight: ', max_weight)
    print('Best value: ', max_value)
    print('Final weight: ', final_weight)
    print('Final value: ', value)
    # print('Best items: ', best_items)
    print('Total time: ', total_time)

    df = pd.DataFrame({'file': {}, 'algorithm': {},
                      'time': {}, 'value': {}})

    df = pd.concat([df, pd.DataFrame(
        {'file': args.file, 'algorithm': args.algorithm, 'time': total_time, 'value': value}, index=[0])])

    filename = 'results_low.csv'

    if os.path.exists(filename):
        df = pd.concat([pd.read_csv(filename), df], ignore_index=True)
        pd.DataFrame(df).to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False, header=True)
