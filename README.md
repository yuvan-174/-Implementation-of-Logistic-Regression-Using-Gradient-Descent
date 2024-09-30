# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```py
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: YUVAN SUNDAR S
RegisterNumber:21223040250
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5  # To avoid log(0)
    cost = (-1/m) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1-y).T, np.log(1-h + epsilon)))
    return cost

# Gradient descent function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# Add intercept term (bias)
def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)

# Main logistic regression function
def logistic_regression(X, y, alpha, iterations):
    X = add_intercept(X)
    theta = np.zeros(X.shape[1])
    theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
    return theta, cost_history

# Prediction function
def predict(X, theta):
    X = add_intercept(X)
    probabilities = sigmoid(np.dot(X, theta))
    return probabilities >= 0.5

# Example usage 
if __name__ == "__main__":
    # Input data (2 features and binary output)
    X = np.array([[0.5, 1.5], [1.0, 1.8], [2.5, 2.3], [3.5, 2.9], [4.5, 3.8], [5.0, 4.5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Parameters
    alpha = 0.1
    iterations = 1000

    # Run logistic regression
    theta, cost_history = logistic_regression(X, y, alpha, iterations)

    # Print the learned parameters and final cost
    print(f"Learned Parameters: {theta}")
    print(f"Final Cost: {cost_history[-1]}")

    # Plot the cost function over iterations
    plt.plot(range(len(cost_history)), cost_history, 'r')
    plt.title("Cost Function using Gradient Descent")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()

    # Prediction
    predictions = predict(X, theta)
    print("Predictions:", predictions)

*/
```

## Output:
![Screenshot 2024-09-30 153308](https://github.com/user-attachments/assets/2f9c3b57-fc32-475d-bec4-187de9f5b304)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

