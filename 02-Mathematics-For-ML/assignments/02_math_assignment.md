# Assignment 02: Mathematics for ML

## Problem 1: Manual Matrix Multiplication
Given the following matrices:
$$ X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} $$
$$ W = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} $$

1. Calculate $Z = XW$ manually on paper. Write down the summation for each cell $Z_{i,j}$.
2. Write a Python script to compute $Z = XW$ using the pure Python `matrix_multiply` function we built in this module. Do the answers match?

## Problem 2: Distance Metrics
Given three text documents represented as word count vectors:
* `doc_A` (Sports): `[5, 4, 0, 0]`
* `doc_B` (Sports): `[4, 5, 0, 1]`
* `doc_C` (Politics): `[0, 0, 5, 4]`

1. Calculate the Euclidean Distance (L2 Norm) between `doc_A` & `doc_B`, and `doc_A` & `doc_C`.
2. Calculate the Cosine Similarity between `doc_A` & `doc_B`, and `doc_A` & `doc_C`.
3. Explain why Cosine Similarity is often preferred over Euclidean distance for text data.
