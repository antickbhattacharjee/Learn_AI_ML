"""
Solution for Assignment 01: Python Vectorization and OOP
"""

import time
import random
import math
try:
    import numpy as np
except ImportError:
    pass

# --- Problem 1: Vectorization ---

def problem_1_solution():
    print("--- Problem 1: Vectorization ---")
    size = 1000
    
    # 1. Pure Python Initialization
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    
    # Python Addition
    start_time = time.time()
    C = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            C[i][j] = A[i][j] + B[i][j]
    py_time = time.time() - start_time
    print(f"Python Nested Loop Time: {py_time:.4f} sec")
    
    # 2. NumPy Addition
    if 'np' in globals():
        A_np = np.array(A)
        B_np = np.array(B)
        
        start_time = time.time()
        C_np = A_np + B_np
        np_time = time.time() - start_time
        print(f"NumPy Vectorized Time: {np_time:.6f} sec")
        
        # 3. Speedup
        print(f"Speedup: {py_time / np_time:.2f}x faster")
    else:
        print("NumPy not installed. Cannot run vectorized comparison.")


# --- Problem 2: OOP Custom Scaler ---

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, data):
        """Calculates mean and standard deviation."""
        n = len(data)
        if n == 0:
            return
            
        self.mean_ = sum(data) / n
        variance = sum((x - self.mean_)**2 for x in data) / n
        self.std_ = math.sqrt(variance)
        
    def transform(self, data):
        """Scales the data using calculated mean and std."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
            
        if self.std_ == 0:
            # Handle zero division if all elements are the same
            return [0.0 for _ in data]
            
        return [(x - self.mean_) / self.std_ for x in data]
        
    def fit_transform(self, data):
        """Convenience method to fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


def problem_2_solution():
    print("\n--- Problem 2: OOP Custom Scaler ---")
    data = [10, 20, 30, 40, 50]
    print(f"Original Data: {data}")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    print(f"Calculated Mean: {scaler.mean_}")
    print(f"Calculated Std: {scaler.std_:.2f}")
    
    # Format to 4 decimal places for readability
    scaled_data_formatted = [round(x, 4) for x in scaled_data]
    print(f"Scaled Data: {scaled_data_formatted}")
    
    # Verification: Mean of scaled data should be ~0, std should be ~1
    scaled_mean = sum(scaled_data) / len(scaled_data)
    scaled_variance = sum((x - scaled_mean)**2 for x in scaled_data) / len(scaled_data)
    print(f"Verification -> Scaled Mean: {scaled_mean:.4f}, Scaled Std: {math.sqrt(scaled_variance):.4f}")

if __name__ == "__main__":
    problem_1_solution()
    problem_2_solution()
