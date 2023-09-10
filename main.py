import numpy as np
from itertools import product
import time
import os

def load_data():

    problems = {i+1: {} for i in range(len(os.listdir('data')))}

    for i in range(len(os.listdir('data'))):
        with open(f'data/p0{i+1}/p0{i+1}_c.txt', 'r') as f:
            problems[i+1]['max_weight'] = int(f.read())

        with open(f'data/p0{i+1}/p0{i+1}_w.txt', 'r') as f:
            weights = np.array([int(x) for x in f.readlines()])
            problems[i+1]['weights'] = weights.astype(int)

        with open(f'data/p0{i+1}/p0{i+1}_p.txt', 'r') as f:
            values = np.array([int(x) for x in f.readlines()])
            problems[i+1]['values'] = values.astype(int)
        
        with open(f'data/p0{i+1}/p0{i+1}_s.txt', 'r') as f:
            solution = np.array([int(x) for x in f.readlines()])
            problems[i+1]['solution'] = solution.astype(int)

    return problems

def bruteforce(weights, values, max_weight):

    def calculate_value(weights, values, max_weight, combination):
        weight = np.sum(combination * weights)
        value = np.sum(combination * values)

        if weight > max_weight:
            return 0, 0
        else:
            return value, weight
        
    max_value = 0
    final_weight = 0
    best_items = np.zeros(len(weights), dtype=int)
    start = time.time()

    for combination in product([0, 1], repeat=len(weights)):
        value, weight = calculate_value(weights, values, max_weight, np.array(combination))

        if value > max_value:
            final_weight = weight
            max_value = value
            best_items = np.array(combination)
    
    return time.time() - start, max_value, final_weight, best_items

def read_file(path):
    weights = []
    values = []
    
    with open(path) as f:
        header = f.readline()
        max_weight = int(header.split()[1])
        file = f.readlines()
        solution = file[-1]
        file = file[:-1]

    solution = np.array([int(x) for x in solution.split()])

    for line in file:
        v, w = line.split()
        weights.append(int(w))
        values.append(int(v))

    def calculate_value(weights, values, max_weight, combination):
        weight = np.sum(combination * weights)
        value = np.sum(combination * values)

        if weight > max_weight:
            return 0, 0

        return weight, value
        
    value = calculate_value(weights, values, max_weight, solution)

    print(value)

    return np.array(weights), np.array(values), max_weight


def dinamica(N, W, values, weights):

    time_start = time.time()

    dp = [[0 for _ in range(W + 1)] for _ in range(N + 1)]
 
    for i in range(1, N + 1):
        for j in range(1, W + 1):
            if weights[i-1] > j:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(values[i-1] + dp[i-1]
                               [j-weights[i-1]], dp[i-1][j])
 
    return dp[N][W], time.time() - time_start

def greedy(N, W, values, weights):
    time_start = time.time()

    # sorting
    value_per_weight = np.array(values) / np.array(weights)
    indexes = np.argsort(value_per_weight)[::-1]
    values = values[indexes]
    weights = weights[indexes]

    W_ = W
    value = 0
    j_ = 0
    knapsack = np.zeros(N)
    for j in range(N):
        if weights[j] > W_:
            knapsack[j] = 0
        else:
            knapsack[j] = 1
            W_ = W_ - weights[j]
            value = value + values[j]
        if values[j] > values[j_]:
            j_ = j
    if values[j_] > value:
        value = values[j_]
        for j in range(N):
            knapsack[j] = 0
            knapsack[j_] = 1

    return value, time.time() - time_start

if __name__ == '__main__':

    problems = load_data()


    path_1 = "instances_01_KP/large_scale/knapPI_1_1000_1000_1"
    path_2 = "instances_01_KP/large_scale/knapPI_2_1000_1000_1"
    path_3 = "instances_01_KP/large_scale/knapPI_3_1000_1000_1"

    weights_1, values_1, max_weight_1 = read_file(path_1)
    weights_2, values_2, max_weight_2 = read_file(path_2)
    weights_3, values_3, max_weight_3 = read_file(path_3)

    """ print("Dinamica")
    print(dinamica(len(weights_1), max_weight_1, values_1, weights_1), len(weights_1))
    print(dinamica(len(weights_2), max_weight_2, values_2, weights_2), len(weights_2))
    print(dinamica(len(weights_3), max_weight_3, values_3, weights_3), len(weights_3)) """

    print("Greedy")
    print(greedy(len(weights_1), max_weight_1, values_1, weights_1), len(weights_1))
    print(greedy(len(weights_2), max_weight_2, values_2, weights_2), len(weights_2))
    print(greedy(len(weights_3), max_weight_3, values_3, weights_3), len(weights_3))
    # total_time, max_value, final_weight, best_items = bruteforce(weights, values, max_weight)

    # print('Max weight: ', max_weight)
    # print('Best value: ', max_value)
    # print('Final weight: ', final_weight)
    # print('Best items: ', best_items)
    # print('Total time: ', total_time)


    # for i in problems.keys():

    #     total_time, max_value, final_weight, best_items = bruteforce(problems[i]['weights'], problems[i]['values'], problems[i]['max_weight'])

    #     print('Problem: ', i)
    #     print('Max weight: ', problems[i]['max_weight'])
    #     print('Best value: ', max_value)
    #     print('Final weight: ', final_weight)
    #     print('Best items: ', best_items)
    #     print('Total time: ', total_time)

    #     print('Equals solution? ', (problems[i]['solution'] == best_items).all())



