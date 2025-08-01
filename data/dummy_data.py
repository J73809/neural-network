import numpy as np

def generate_data(samples=500):
    data = []
    for _ in range(samples):
        x = np.random.rand(7, 1)  # input vector (7x1)

        # Clear pattern: sum of first 3 elements decides the class (5 classes)
        s = np.sum(x[:3])
        
        # Assign class based on sum ranges (non-overlapping)
        if s < 1.0:
            label = 0
        elif s < 1.5:
            label = 1
        elif s < 2.0:
            label = 2
        elif s < 2.5:
            label = 3
        else:
            label = 4

        y = np.zeros((5, 1))
        y[label, 0] = 1  # one-hot

        data.append((x, y))

    return data