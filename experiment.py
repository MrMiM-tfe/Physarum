import numpy as np

a = np.zeros((10, 10, 3))
r = np.random.randint(0, 10, (5, 5, 2))

# a[:, 0] = 1

print(r)
b = a[r[:, 0], r[:, 1]]

for i in range(5):
    for j in range(5):
        print(a[i, j], end="")
    print("")