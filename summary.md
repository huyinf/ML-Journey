# Summary misunderstand concepts from projects after implementing from scratch

## 1. Sales Prediction

**Keyword: Multiple Linear Regression**

### Normal Equation
- a closed-form solution that directly calculates the optimal $\theta$ values without requiring an iterative optimization algorithm like gradient descent.

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
  2. **Matrix Multiplication** $X^T X$ results in a square matrix used to calculate its inverse.
  3. **Matrix Multiplication** $$X^T y$ results in a vector used for the final multiplication.
  4. **Final Multiplication** between $X^T X$ and $X^T y$ to obtain the coefficients $\theta$ that minimize the loss fuction.
- This process is efficient for small datasets and offers a direct, one-step solution to finding the optimal coefficients. However, it can be computationally expensive and may not work for datasets with a large number of features or when the matrix $X^T X$ is singular.

**Mathematics Explanation**
- Given a function $f: \mathbb{R}^{2\times 2} \mapsto \mathbb{R}$ and a matrix
    
$$
A = \begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}\\
\end{bmatrix}
$$

- The gradient of $f(A)$ respect to matrix $A$ is formulated by:    

$$
\nabla_A f(A) = 
\begin{bmatrix}
\displaystyle\frac{\partial f}{\partial A_{11}} & \displaystyle\frac{\partial f}{\partial A_{12}}\\
\displaystyle\frac{\partial f}{\partial A_{11}} & \displaystyle\frac{\partial f}{\partial A_{12}}\\
\end{bmatrix}
$$

- **Cost function**

$$
J = \frac{1}{2}\sum_{i=1}^m\left(h_{\theta}(x^{(i)})-y^{(i)}\right)^2
$$
- We have $X, \theta, y$:

$$
X = \begin{bmatrix}
x^{(1)^T}\\
\vdots\\
x^{(m)^T}
\end{bmatrix},\quad \theta =
\begin{bmatrix}
\theta_0\\
\theta_1\\
\vdots\\
\theta_n
\end{bmatrix},\quad y = 
\begin{bmatrix}
y^{(1)}\\
\vdots\\
y^{(m)}\\
\end{bmatrix}
$$

$$
X\theta = \begin{bmatrix}
x^{(1)^T}\\
\vdots\\
x^{(m)^T}
\end{bmatrix} \begin{bmatrix}
\theta_0\\
\theta_1\\
\vdots\\
\theta_n
\end{bmatrix}
= \begin{bmatrix}
X^{(1)^T}\theta\\
\vdots\\
X^{(m)^T}\theta
\end{bmatrix} = \begin{bmatrix}
h_{\theta}(x^{(1)})\\
\vdots\\
h_{\theta}(x^{(m)})\\
\end{bmatrix}\\
X\theta-y = \begin{bmatrix}
h_{\theta}^{(1)}-y^{(1)}\\
\vdots\\
h_{\theta}^{(m)}-y^{(m)}
\end{bmatrix}
$$

$$
\sum_{i=1}^m\left(h_{\theta}(x^{(i)})-y^{(i)}\right)^2 = (X\theta-y)^T (X\theta-y)\\
\Rightarrow J = \frac{1}{2}(X\theta-y)^T (X\theta-y)
$$

- To minimize cost function $J$, take the gradient:

$$
\begin{aligned}
\nabla_{\theta}J(\theta) &= \nabla_{\theta}\frac{1}{2}(X\theta-y)^T(X\theta-y)\\
&= \frac{1}{2}\nabla_{\theta}(\theta^TX^T-y^T)(X\theta-y)\\
&= \frac{1}{2}\nabla_{\theta}[\theta^TX^TX\theta-\theta^TX^Ty-y^TX\theta+y^Ty]\\
&=\frac{1}{2}[X^TX\theta+X^TX\theta-X^Ty-X^Ty]\\
&= X^TX\theta-X^Ty\\
&=\vec{0}\quad(\text{optmial value})\\
\end{aligned}
$$

$$
\begin{aligned}
&\Rightarrow X^TX\theta=X^Ty \quad(\text{Normal Equation})\\
&\Rightarrow \theta = (X^TX)^{-1}X^Ty.
\end{aligned}
$$

## 2. Disease Prediction

**Keywords: Cross-Validation, Confustion Matrix, SVM, Navie Bayes, Random Forest, Classifier**

- Cleaning data:
  - Drop empty column
  - Encode the target column i.e. "prognosis" from string type to numerical type by using a **label encoder**.
- Model building:
  - Use Support Vector Classifier, Naive Bayes Classifier, and Random Forest Classifier.
  - Use **confusion matrix** to determine the quality of the models.
- Inference:
  - Use the prediction of combined results from all models to get a more robust and accurate prediction.

- Visualization

![Confusion Matrix](disease_predictionmgs/cf_matrix.png)

## 3. Handwritten Digit Recognition

**Keywords: **