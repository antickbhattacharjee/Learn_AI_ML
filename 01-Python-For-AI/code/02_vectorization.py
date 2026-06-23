"""
Module 01: Python for AI
Topic: The Power of Vectorization

This script demonstrates the immense performance difference between
using standard Python loops and utilizing NumPy's vectorized operations
which wrap highly optimized C and Fortran code.
"""

import time
import random

try:
    import numpy as np
except ImportError:
    print("NumPy is required. Please install it using 'pip install numpy'")
    exit(1)

def python_loop_dot_product(a, b):
    """Calculates dot product using native Python."""
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def numpy_vectorized_dot_product(a, b):
    """Calculates dot product using NumPy."""
    # NumPy arrays support direct vector operations
    return np.dot(a, b)

if __name__ == "__main__":
    N = 10_000_000
    print(f"Creating vectors of size {N}...")
    
    # Generate data
    list_a = [random.random() for _ in range(N)]
    list_b = [random.random() for _ in range(N)]
    
    # Convert to NumPy contiguous arrays
    arr_a = np.array(list_a)
    arr_b = np.array(list_b)
    
    print("\n--- Running Python For-Loop ---")
    start = time.time()
    res1 = python_loop_dot_product(list_a, list_b)
    loop_time = time.time() - start
    print(f"Result: {res1}")
    print(f"Time Taken: {loop_time:.4f} seconds")
    
    print("\n--- Running NumPy Vectorization ---")
    start = time.time()
    res2 = numpy_vectorized_dot_product(arr_a, arr_b)
    vec_time = time.time() - start
    print(f"Result: {res2}")
    print(f"Time Taken: {vec_time:.6f} seconds")
    
    speedup = loop_time / vec_time if vec_time > 0 else float('inf')
    print(f"\nVectorization is roughly {speedup:.2f}x faster.")
    
    print("\nExplanation:")
    print("Python loops through memory that is scattered, creating a C-struct for every")
    print("single float. NumPy stores a single block of raw C-floats and executes the")
    print("addition in a highly parallelized C loop using SIMD registers.")
