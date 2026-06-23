"""
Module 02: Mathematics for ML
Topic: Matrix Operations from Scratch

This script manually implements vector dot products and matrix multiplication
to demonstrate the underlying summation formulas.
"""

def dot_product(vec_a, vec_b):
    """
    Computes the dot product of two vectors: a . b = Σ (a_i * b_i)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length.")
    
    result = 0.0
    for i in range(len(vec_a)):
        result += vec_a[i] * vec_b[i]
    return result


def matrix_multiply(A, B):
    """
    Computes the matrix multiplication C = A * B.
    If A is (m x n) and B is (n x p), C will be (m x p).
    C_{i,j} = Σ_{k=1}^{n} A_{i,k} B_{k,j}
    """
    m = len(A)
    n = len(A[0])
    
    # Check if B has n rows
    if len(B) != n:
        raise ValueError("Inner dimensions must match (A cols == B rows).")
    
    p = len(B[0])
    
    # Initialize output matrix C with zeros (shape m x p)
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    
    # Perform multiplication
    for i in range(m):
        for j in range(p):
            # The value of C[i][j] is the dot product of the i-th row of A and the j-th col of B
            dot_val = 0.0
            for k in range(n):
                dot_val += A[i][k] * B[k][j]
            C[i][j] = dot_val
            
    return C

def print_matrix(name, mat):
    print(f"{name}:")
    for row in mat:
        # Format numbers to 2 decimal places
        print("  [" + ", ".join([f"{val:6.2f}" for val in row]) + "]")
    print()


if __name__ == "__main__":
    print("--- 1. Vector Dot Product ---")
    v1 = [1, 2, 3]
    v2 = [4, -5, 6]
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"Dot Product (v1 . v2) = {dot_product(v1, v2)}")
    
    print("\n--- 2. Matrix Multiplication ---")
    # A is 2x3
    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    
    # B is 3x2
    B = [
        [7, 8],
        [9, 1],
        [2, 3]
    ]
    
    print_matrix("Matrix A (2x3)", A)
    print_matrix("Matrix B (3x2)", B)
    
    C = matrix_multiply(A, B)
    print_matrix("Result C = A * B (2x2)", C)
    
    print("Verification:")
    print("C[0][0] = (1*7) + (2*9) + (3*2) = 7 + 18 + 6 = 31")
