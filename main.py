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

    if int(filename.split('_')[1]) > 3:
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
            v, w = line.split()
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

    selected_items = np.zeros(N)
    i, j = N, W
    while i > 0 and j > 0:
        if dp[i][j] != dp[i - 1][j]:
            selected_items[i - 1] = 1
            j -= weights[i - 1]
        i -= 1

    return time.time() - time_start, value, weight, selected_items


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
    knapsack = np.zeros(N)
    first = False
    didnt_fit = 0
    for i in range(N):
        if weights[i] <= remainingW:
            knapsack[i] = 1
            remainingW = remainingW - weights[i]
            value = value + values[i]
        else:
            if not first and weights[i] <= W:
                didnt_fit = i
                first = True
    if values[didnt_fit] > value and weights[didnt_fit] <= W:
        value = values[didnt_fit]
        for i in range(N):
            knapsack[i] = 0
            knapsack[didnt_fit] = 1

    return time.time() - time_start, value, W - remainingW, knapsack


def base_profit(value, weight, s):
    if weight == 0:
        return 0
    return value if s >= weight else 0


def min_dynamic(W, weights, values):  # O = (n² * Vmax)
    N = len(values)
    max_value = np.max(values)
    max_weight = np.max(weights)

    table = [[0 for _ in range(N * max_value + 1)] for _ in range(N)]

    val = 1
    while val <= values[0]:
        table[0][val] = weights[0]
        val += 1

    val = values[0] + 1
    while val <= N * max_value:
        table[0][val] = np.iinfo(np.int32).max - max_weight
        val += 1

    for i in range(1, N):
        for j in range(1, N * max_value + 1):
            new_target = max(0, j - values[i])
            if table[i - 1][j] <= table[i - 1][new_target] + weights[i]:
                table[i][j] = table[i - 1][j]
            else:
                table[i][j] = table[i - 1][new_target] + weights[i]

    result = -1
    for i in range(N * max_value + 1):
        if table[N - 1][i] > W:
            result = i - 1
            break

    return result


def fptas(W, weights, values, epsilon):  # O = (n² * Vmax)
    time_start = time.time()

    N = len(values)
    max_value = np.max(values)

    scaling_factor = np.ceil((epsilon * max_value) / N)
    scaled_values = np.uint32(np.floor_divide(values, scaling_factor))

    result = min_dynamic(W, weights, scaled_values)

    return time.time() - time_start, result, W, np.zeros(N)


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
            max_weight, weights, values, 0.5)
    else:
        print('Invalid algorithm')
        exit()

    print('Max weight: ', max_weight)
    print('Final weight: ', final_weight)
    print('Best value: ', max_value)
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
