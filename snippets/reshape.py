import numpy as np

arr = np.array([
    [
        [1.0, 1.1],
        [1.2, 1.3]
    ],
    [
        [2.0, 2.1],
        [2.2, 2.3]
    ],
    [
        [3.0, 3.1],
        [3.3, 3.3]
    ],
])

rows = len(arr)

size = arr.size

print('Original: \n', arr)

print('Reshape: \n', arr.reshape(rows, size // rows))

