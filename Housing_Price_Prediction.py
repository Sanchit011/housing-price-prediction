# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    """
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    #no. of features n
    n = X_norm.shape[1]

    for i in range(n):
        mu[i] = np.mean(X_norm[:, i])
        sigma[i] = np.std(X_norm[:, i])
        X_norm[:, i] = (X_norm[:, i] - mu[i])/sigma[i]

    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    """
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    
    predictions = np.dot(X, theta)
    error = predictions - y
    squared_error = error*error
    J = (np.sum(squared_error))/(2*m)

    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    """
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    
    for i in range(num_iters):
        predictions = np.dot(X, theta)
        error = predictions - y
        error = error[None, :]
        summation = np.dot(error, X)
        theta = theta - ((alpha/m)*summation)[0]    
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history


# Load data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size

# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# Choose some alpha value - try changing this
alpha = 0.01
num_iters = 500
# init theta and run gradient descent
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

print("\n ----------------------------------------------------------------")
# Display the gradient descent's result
print('theta computed from gradient descent: {:s}'.format(str(theta)))

# Estimating the price of a 1650 sq-ft, 3 bedroom house 
#normalizing the given input
size = (1650 - mu[0])/sigma[0]
br = (3 - mu[1])/sigma[1]

price = np.dot([1, size, br], theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${:.0f}'.format(price))
print("------------------------------------------------------------------ \n")

prediction = np.dot(X, theta)

deviation = abs((prediction - y)/y)
avg_dev = np.sum(deviation)/47
print("Average deviation is")
print(avg_dev)


# Plot the convergence graph
plt.plot(np.arange(len(J_history)), J_history, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()






