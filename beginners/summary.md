# Summary misunderstand concepts from projects after implementing from scratch

## 1. Sales Prediction

**Keyword: Linear Regression**

- Input: Multiple Variables
### Normal Equation
- a closed-form solution that directly calculates the optimal $\theta$ values without the need for an iterative optimization algorithim like gradient descent.

$$
\theta = (X^T X)^{-1} X^T y
$$
- where:
  - $X$ is the matrix of input features.
  - $y$ is the vector of output values.
  - $X^T$ is the transpose of $x$.
  - $\theta$ is the vector of coefficients.
- Order of operations:
  1. **Transpose** $X$ to align the dimensions for matrix multiplication.
  2. **Matrix Multiplication** $X^T X$ results in a square matrix that is used to calculate its inverse.
  3. **Matrix Multiplication** $$X^T y$ results in a vector that is used for the final multiplication.
  4. **Final Multiplication** between $X^T X$ and $X^T y$ to obtain the coefficients $\theta$ that minimize the loss fuction.
- This process is efficient for small datasets and offers a direct, one-step solution to finding the optimal coefficients. However, it can be computationally expensive and may not work for datasets with a large number of features or when the matrix $X^T X$ is singular.