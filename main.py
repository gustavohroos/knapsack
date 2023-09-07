import numpy as np
from itertools import product
import time
import os

all_weights = []
max_weights = []
all_values = []
all_solutions = []

for i in range(len(os.listdir('data'))):

    with open(f'data/p0{i+1}/p0{i+1}_c.txt', 'r') as f:
        max_weights.append(int(f.read()))

    with open(f'data/p0{i+1}/p0{i+1}_w.txt', 'r') as f:
        weights = np.array([int(x) for x in f.readlines()])
        all_weights.append(weights.astype(int))

    with open(f'data/p0{i+1}/p0{i+1}_p.txt', 'r') as f:
        values = np.array([int(x) for x in f.readlines()])
        all_values.append(values.astype(int))
    
    with open(f'data/p0{i+1}/p0{i+1}_s.txt', 'r') as f:
        solution = np.array([int(x) for x in f.readlines()])
        all_solutions.append(solution.astype(int))

    print('Problem: ', i+1)
    print('Max weight: ', max_weights[i])
    print('Weights: ', all_weights[i])
    print('Values: ', all_values[i])
    print('Solution: ', all_solutions[i])

def bruteforce(weights, values, max_weight):
    n = len(weights)
    max_value = 0
    final_weight = 0
    best_items = np.zeros(n, dtype=int)
    start = time.time()

    for combination in product([0, 1], repeat=n):
        weight = np.sum(np.array(combination) * weights)
        value = np.sum(np.array(combination) * values)

        if weight <= max_weight and value > max_value:
            final_weight = weight
            max_value = value
            best_items = np.array(combination)
    
    return time.time() - start, max_value, final_weight, best_items

for i in range(len(os.listdir('data'))):
    weights = all_weights[i]
    values = all_values[i]
    solution = all_solutions[i]
    MAX_WEIGHT = max_weights[i]

    total_time, max_value, final_weight, best_items = bruteforce(weights, values, MAX_WEIGHT)

    print('Problem: ', i+1)
    print('Max weight: ', MAX_WEIGHT)
    print('Best value: ', max_value)
    print('Final weight: ', final_weight)
    print('Best items: ', best_items)
    print('Total time: ', total_time)

    print('Equals solution? ', (solution == best_items).all())