import numpy as np
from itertools import product
import time

with open('data/p01/p01_c.txt') as f:
    MAX_WEIGHT = int(f.readline())

with open('data/p01/p01_w.txt') as f:
    weights = np.array([int(x) for x in f.readlines()])

with open('data/p01/p01_p.txt') as f:
    values = np.array([int(x) for x in f.readlines()])

with open('data/p01/p01_s.txt') as f:
    solution = np.array([int(x) for x in f.readlines()])

print('Max weight: ', MAX_WEIGHT)
print('Weights: ', weights)
print('Values: ', values)
print('Solution: ', solution)


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

total_time, max_value, final_weight, best_items = bruteforce(weights, values, MAX_WEIGHT)

print('Max value: ', max_value)
print('Final weight: ', final_weight)
print('Best items: ', best_items)
print('Total time: ', total_time)

print('Equals solution? ', (solution == best_items).all())