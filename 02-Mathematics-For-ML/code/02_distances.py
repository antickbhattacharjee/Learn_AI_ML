"""
Module 02: Mathematics for ML
Topic: Distance Metrics and Norms

This script manually implements the L1 Norm, L2 Norm, and Cosine Similarity,
which are foundational for algorithms like K-Nearest Neighbors and K-Means.
"""

import math

def l1_distance(vec_a, vec_b):
    """
    Manhattan Distance (L1 Norm): Σ |a_i - b_i|
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length.")
        
    distance = 0.0
    for i in range(len(vec_a)):
        distance += abs(vec_a[i] - vec_b[i])
    return distance


def l2_distance(vec_a, vec_b):
    """
    Euclidean Distance (L2 Norm): sqrt( Σ (a_i - b_i)^2 )
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length.")
        
    sum_sq = 0.0
    for i in range(len(vec_a)):
        sum_sq += (vec_a[i] - vec_b[i])**2
    return math.sqrt(sum_sq)


def vector_magnitude(vec):
    """
    Computes the L2 norm of a single vector (distance from origin).
    ||v||_2 = sqrt( Σ v_i^2 )
    """
    return math.sqrt(sum(v**2 for v in vec))


def cosine_similarity(vec_a, vec_b):
    """
    Cosine Similarity: (a . b) / (||a|| * ||b||)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length.")
        
    # 1. Compute Dot Product
    dot_prod = sum(vec_a[i] * vec_b[i] for i in range(len(vec_a)))
    
    # 2. Compute Magnitudes
    mag_a = vector_magnitude(vec_a)
    mag_b = vector_magnitude(vec_b)
    
    if mag_a == 0 or mag_b == 0:
        return 0.0 # Prevent division by zero
        
    # 3. Calculate Similarity
    return dot_prod / (mag_a * mag_b)


if __name__ == "__main__":
    print("--- Distance Metrics ---")
    
    # Imagine two users rating 3 movies on a scale of 1-5
    user_1 = [5, 4, 1]
    user_2 = [4, 5, 1]
    user_3 = [1, 1, 5]
    
    print(f"User 1 ratings: {user_1}")
    print(f"User 2 ratings: {user_2}")
    print(f"User 3 ratings: {user_3}\n")
    
    print(f"L1 Distance (User 1, User 2): {l1_distance(user_1, user_2)}")
    print(f"L1 Distance (User 1, User 3): {l1_distance(user_1, user_3)}\n")
    
    print(f"L2 Distance (User 1, User 2): {l2_distance(user_1, user_2):.4f}")
    print(f"L2 Distance (User 1, User 3): {l2_distance(user_1, user_3):.4f}\n")
    
    print("--- Cosine Similarity ---")
    print("Range is [-1, 1]. 1 means perfectly aligned, 0 is orthogonal, -1 is opposite.")
    print(f"Similarity (User 1, User 2): {cosine_similarity(user_1, user_2):.4f}")
    print(f"Similarity (User 1, User 3): {cosine_similarity(user_1, user_3):.4f}")
    
    print("\nTakeaway: Distance metrics tell us how similar data points are.")
    print("User 1 and User 2 have similar tastes, resulting in low L2 distance")
    print("and high Cosine Similarity.")
