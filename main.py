import numpy as np

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

    best_items = np.zeros(n, dtype=int)  # Initialize an array to store the selected items

    for i in range(2**n):
        weight = 0
        value = 0
        items = np.zeros(n, dtype=int)  # Initialize an array to represent selected items in binary

        for j in range(n):
            if (i >> j) & 1:
                weight += weights[j]
                value += values[j]
                items[j] = 1  # Mark item as selected

        if weight <= max_weight and value > max_value:
            final_weight = weight
            max_value = value
            best_items = items.copy()  # Update the best_items array
    
    return max_value, final_weight, best_items

max_value, final_weight, best_items = bruteforce(weights, values, MAX_WEIGHT)

print('Max value: ', max_value)
print('Final weight: ', final_weight)
print('Best items: ', best_items)

print('Equals solution? ', (solution == best_items).all())