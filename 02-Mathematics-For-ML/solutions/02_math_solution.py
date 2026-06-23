"""
Solution for Assignment 02: Mathematics for ML
"""

import math

# --- Problem 1: Matrix Multiplication ---

def matrix_multiply(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def problem_1_solution():
    print("--- Problem 1: Matrix Multiplication ---")
    X = [[1, 2], 
         [3, 4]]
    W = [[5, 6], 
         [7, 8]]
         
    Z = matrix_multiply(X, W)
    print("Z = X * W:")
    for row in Z:
        print(row)
        
    print("\nManual Verification:")
    print("Z[0][0] = (1*5) + (2*7) = 5 + 14 = 19")
    print("Z[0][1] = (1*6) + (2*8) = 6 + 16 = 22")
    print("Z[1][0] = (3*5) + (4*7) = 15 + 28 = 43")
    print("Z[1][1] = (3*6) + (4*8) = 18 + 32 = 50")


# --- Problem 2: Distance Metrics ---

def l2_distance(vec_a, vec_b):
    return math.sqrt(sum((vec_a[i] - vec_b[i])**2 for i in range(len(vec_a))))

def cosine_similarity(vec_a, vec_b):
    dot = sum(vec_a[i] * vec_b[i] for i in range(len(vec_a)))
    mag_a = math.sqrt(sum(v**2 for v in vec_a))
    mag_b = math.sqrt(sum(v**2 for v in vec_b))
    return dot / (mag_a * mag_b)

def problem_2_solution():
    print("\n--- Problem 2: Distance Metrics ---")
    doc_a = [5, 4, 0, 0]
    doc_b = [4, 5, 0, 1]
    doc_c = [0, 0, 5, 4]
    
    # 1. L2 Distance
    print(f"L2 Distance (A, B): {l2_distance(doc_a, doc_b):.4f}")
    print(f"L2 Distance (A, C): {l2_distance(doc_a, doc_c):.4f}")
    
    # 2. Cosine Similarity
    print(f"Cosine Sim (A, B): {cosine_similarity(doc_a, doc_b):.4f}")
    print(f"Cosine Sim (A, C): {cosine_similarity(doc_a, doc_c):.4f}")
    
    # 3. Explanation
    print("\nWhy Cosine Similarity for Text?")
    print("If doc_A and doc_B have exactly the same proportions of words, but doc_B")
    print("is twice as long (e.g. A=[1,1], B=[2,2]), their Euclidean distance will be")
    print("large, incorrectly implying they are different. Their Cosine Similarity")
    print("will be 1.0, correctly implying they discuss the exact same topics.")

if __name__ == "__main__":
    problem_1_solution()
    problem_2_solution()
