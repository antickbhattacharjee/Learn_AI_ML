# Assignment 01: Python Vectorization and OOP

## Problem 1: Vectorization
You are given two large matrices $A$ and $B$, both of size $1000 \times 1000$. 

**Task:**
1. Write a pure Python nested loop function to compute the Matrix Addition $C = A + B$.
2. Write a NumPy function to perform the same operation.
3. Profile both functions using the `time` module and calculate the speedup multiplier.

## Problem 2: OOP Custom Scaler
In ML, we frequently scale data using a StandardScaler: $z = \frac{(x - \mu)}{\sigma}$.

**Task:**
Create a class called `StandardScaler`.
It should have:
1. `__init__(self)`: Initializes state.
2. `fit(self, data)`: Calculates the mean $\mu$ and standard deviation $\sigma$ of the incoming list/array.
3. `transform(self, data)`: Returns a new array where every element has been scaled using the computed $\mu$ and $\sigma$.
4. `fit_transform(self, data)`: Calls fit, then transform.

Test your code on `data = [10, 20, 30, 40, 50]`.
