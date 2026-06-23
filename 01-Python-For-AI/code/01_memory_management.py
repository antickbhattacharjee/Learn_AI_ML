"""
Module 01: Python for AI
Topic: Memory Management and Performance Profiling

This script demonstrates the memory overhead of Python objects and how
generators and iterators can save memory when processing large datasets.
"""

import sys
import time

def memory_overhead_demo():
    print("--- Memory Overhead of Python Objects ---")
    
    # An integer is not just 4 or 8 bytes in Python.
    x = 10
    print(f"Size of an integer (10): {sys.getsizeof(x)} bytes")
    
    # An empty list
    empty_list = []
    print(f"Size of an empty list: {sys.getsizeof(empty_list)} bytes")
    
    # List with 1000 integers
    # Notice that a list stores pointers, not the actual values continuously
    list_1k = [i for i in range(1000)]
    size_of_list_pointers = sys.getsizeof(list_1k)
    size_of_elements = sum(sys.getsizeof(i) for i in list_1k)
    total_size = size_of_list_pointers + size_of_elements
    print(f"Total size of list with 1000 elements: {total_size} bytes")


def generator_vs_list_demo():
    print("\n--- Generators vs Lists for Big Data ---")
    
    N = 10_000_000
    
    # WARNING: Do not run the list generation on low-RAM systems with very high N.
    print(f"Processing {N} elements...")
    
    # Using a Generator Expression (Lazy Evaluation)
    gen = (i ** 2 for i in range(N))
    print(f"Memory used by generator: {sys.getsizeof(gen)} bytes")
    
    # Let's iterate through the generator (we won't print to save time)
    start_time = time.time()
    for _ in gen:
        pass
    print(f"Generator Iteration Time: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    memory_overhead_demo()
    generator_vs_list_demo()
    
    print("\nTakeaway: In AI, we deal with billions of parameters. Using native Python lists")
    print("is catastrophic for memory. This is why we use C-backed arrays like NumPy.")
