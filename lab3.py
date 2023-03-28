from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def gauss_elimination_with_pivoting(A, B):
    n = len(A)
    # Гауссово исключение с частичным поворотом
    for i in range(n):
        # Частичный поворот
        max_index = i
        for j in range(i + 1, n):
            if abs(A[j][i]) > abs(A[max_index][i]):
                max_index = j
        A[i], A[max_index] = A[max_index], A[i]
        B[i], B[max_index] = B[max_index], B[i]

        # исключение
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i + 1, n):
                A[j][k] = A[j][k] - factor * A[i][k]
            B[j] = B[j] - factor * B[i]

    # Обратная замена
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (B[i] - s) / A[i][i]

    return x


def solve_func(coefficients, x):
    sum = 0

    for i in range(len(coefficients)):
        sum += coefficients[i] * x ** i

    return sum


def generate_graph_data(coefficients, left_bound, right_bound, step):
    a: list[Any] = []
    b: list[Any] = []

    for x in np.arange(left_bound, right_bound, step):
        a.append(x)
        b.append(solve_func(coefficients, x))

    return a, b


x = [8, 10, 12, 14, 16]
f = [4, 9, 5, 1, 16]

n = len(x)

X = [[0] * n for _ in range(n)]
Y = f

for i in range(n):
    for j in range(n):
        X[i][j] = x[i] ** j

result = gauss_elimination_with_pivoting(X, Y)

a, b = generate_graph_data(result, 0, 10, 0.1)

plt.plot(a, b)
plt.xlabel('Похуй')
plt.ylabel('Насрать вообще')
plt.show()
