# Module 02: Mathematics for Machine Learning

To understand how Machine Learning models learn, you must understand the language they speak: **Mathematics**. 

This module covers the absolute foundations: Sets, Functions, Vectors, and Matrices. We will build up to measuring distance in high-dimensional spaces, a critical concept for clustering and classification algorithms.

## Table of Contents
1. [Sets and Relations](#1-sets-and-relations)
2. [Functions in ML](#2-functions-in-ml)
3. [Vectors and Vector Spaces](#3-vectors-and-vector-spaces)
4. [Matrix Operations](#4-matrix-operations)
5. [Distance Metrics (Norms)](#5-distance-metrics-norms)

---

## 1. Sets and Relations

A **Set** is a well-defined collection of distinct objects. In Machine Learning, our dataset is a set of samples.

### Mathematical Definition
If $X$ is a dataset of images, we denote $x_i \in X$ to mean that a specific image $x_i$ belongs to the set $X$.

**Common Sets in ML:**
* $\mathbb{R}$: The set of all real numbers (e.g., house prices, temperatures).
* $\mathbb{Z}$: The set of all integers.
* $\mathbb{B} = \{0, 1\}$: Boolean set, used for binary classification (e.g., Spam vs Not Spam).

---

## 2. Functions in ML

A **Function** $f$ maps elements from a domain $X$ to a codomain $Y$. 
$$ f: X \rightarrow Y $$

In Machine Learning, a model is simply a function that we are trying to approximate.
* **Domain ($X$):** The input features (e.g., pixel values of an image).
* **Codomain ($Y$):** The predicted label (e.g., "Cat" or "Dog").

### Linear Functions
The most fundamental ML function is the linear equation:
$$ f(x) = wx + b $$
Where $w$ is the **weight** and $b$ is the **bias**. The goal of training a model is to find the optimal values for $w$ and $b$ that minimize the error of the function across the set $X$.

---

## 3. Vectors and Vector Spaces

A **Vector** is an ordered array of numbers. In ML, a single data point is almost always represented as a vector.

For example, a house can be represented by a vector $x$:
$$ x = \begin{bmatrix} 2500 \\ 3 \\ 15 \end{bmatrix} $$
*(Where 2500 is square feet, 3 is bedrooms, and 15 is age in years).*

This vector exists in a 3-dimensional **Vector Space**, denoted as $\mathbb{R}^3$.

### Dot Product
The dot product of two vectors $a$ and $b$ of length $n$ is a scalar representing how "aligned" they are:
$$ a \cdot b = \sum_{i=1}^{n} a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n $$

If the dot product is 0, the vectors are **orthogonal** (perpendicular) to each other, meaning they share no correlation in that vector space.

---

## 4. Matrix Operations

A **Matrix** is a 2D grid of numbers. If a vector represents a single house, a matrix represents the entire dataset of houses.

Let $X$ be a matrix of shape $(m, n)$, where $m$ is the number of samples and $n$ is the number of features.
$$ X = \begin{bmatrix} x_{1,1} & x_{1,2} & \dots & x_{1,n} \\ x_{2,1} & x_{2,2} & \dots & x_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m,1} & x_{m,2} & \dots & x_{m,n} \end{bmatrix} $$

### Matrix Multiplication
When we pass a dataset $X$ through a linear layer of a Neural Network, we perform a matrix multiplication with a weight matrix $W$:
$$ Z = XW $$

If $X$ has shape $(m, n)$ and $W$ has shape $(n, p)$, the resulting matrix $Z$ will have shape $(m, p)$. 
*Rule:* The inner dimensions ($n$) must match.

$$ C_{i,j} = \sum_{k=1}^{n} A_{i,k} B_{k,j} $$

*See `code/01_matrix_math.py` for the manual implementation of this summation.*

---

## 5. Distance Metrics (Norms)

To classify data (e.g., K-Nearest Neighbors), we need a mathematical way to measure "how close" two vectors are. We use **Norms**.

### L1 Norm (Manhattan Distance)
The sum of absolute differences.
$$ ||x||_1 = \sum_{i=1}^{n} |x_i| $$
*Distance between $a$ and $b$:* $\sum |a_i - b_i|$

### L2 Norm (Euclidean Distance)
The shortest straight-line distance between two points.
$$ ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2} $$
*Distance between $a$ and $b$:* $\sqrt{\sum (a_i - b_i)^2}$

### Cosine Similarity
Measures the angle between two vectors, ignoring their magnitude. Heavily used in NLP to compare document embeddings.
$$ \cos(\theta) = \frac{a \cdot b}{||a||_2 ||b||_2} $$

---

## Next Steps
1. Navigate to the `code/` directory and run `01_matrix_math.py` and `02_distances.py`.
2. Attempt the math assignment in `assignments/02_math_assignment.md`.
