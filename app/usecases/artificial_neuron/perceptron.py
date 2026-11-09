import numpy as np
from typing import Union
class Perceptron(object):
  """
  Perceptron classifier
  
  Parameters
  ----------
  eta : float
    Learning rate (between 0.0 and 1.0)
  n_iter : int
    Passes over the training dataset
  random_state : int
    Random number generator seed for random weight initialization
  
  Attributes
  ----------
  weights : 1d-array
    Weights after fitting
  errors : list
    Number of misclassifications in every epoch
  """
  
  def __init__(self, eta:float=0.01, n_iter:int=50, random_state:int=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state
  
  def fit(self, x:np.ndarray, y:np.ndarray):
    """
    Fit training data

    The fit method is used to train the Perceptron model using the training data provided.

    Parameters
    ----------
    x: {array-like}, shape = [n_samples, n_features]
    Training vectors, where n_samples is the number of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
    Target values.

    The fit method works as follows:

    1. Initialize the weights and the error list. The weights are initialized to small random numbers drawn from a normal distribution with standard deviation 0.01. The error list is initialized to an empty list.

    2. For each iteration in the specified number of iterations (self.n_iter), do the following:

        a. Initialize the error count to 0.

        b. For each input vector (xi) and corresponding target value (target) in the training data, do the following:

            i. Calculate the prediction for xi using the predict method.

            ii. Calculate the error as the difference between the target value and the prediction.

            iii. Calculate the update as the learning rate (self.eta) times the error.

            iv. Update the weights by adding the update times the input vector to the weights (excluding the bias weight).

            v. Update the bias weight by adding the update to the bias weight.

            vi. If the update is not close to 0 (using a relative tolerance of 1e-09 and an absolute tolerance of 1e-09), increment the error count.

        c. Append the error count to the error list.

    3. Return the trained model (self).

    The fit method uses the perceptron learning rule to update the weights. The perceptron learning rule is a simple rule that adjusts the weights according to the error of the prediction. If the prediction is correct, the weights are not adjusted. If the prediction is incorrect, the weights are adjusted in the direction that would make the prediction correct.

    The fit method also keeps track of the number of errors in each iteration. This can be used to check if the model is learning from the training data. If the number of errors decreases over time, it means that the model is learning.

    The fit method returns the trained model. After training, the model can be used to make predictions on new data using the predict method.
    """
    
    rgen = np.random.default_rng(self.random_state)
    self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1]) # It initializes the weights to small random numbers drawn from a normal distribution with standard deviation 0.01
    
    self.errors = []
    
    for _ in range(self.n_iter):
      errors = 0
      for xi, target in zip(x,y):
        prediction = self.predict(xi)
        error = target - prediction
        update = self.eta * error
        self.weights[1:] += update * xi
        self.weights[0] += update
        errors += int(not np.isclose(update, 0.0, rtol=1e-09, atol=1e-09))
      self.errors.append(errors)
    
    return self
  
  
  def net_input(self, x:np.ndarray) -> Union[np.float64, np.ndarray]:
    """
    Calculate net input

    This function calculates the dot product of the input array `x` and the weights of the perceptron (excluding the first weight, which is the bias). 
    The dot product is then added to the bias to calculate the net input to the activation function.

    The dot product is calculated using the numpy function `np.dot`, which multiplies the corresponding elements of `x` and `self.weights[1:]` and sums the results. 
    For example, if `x = [x1, x2, x3]` and `self.weights[1:] = [w1, w2, w3]`, then `np.dot(x, self.weights[1:])` is `x1*w1 + x2*w2 + x3*w3`.

    The net input is used in the activation function to determine the output of the perceptron. 
    It represents the weighted sum of the inputs, where the weights are the coefficients that determine the relative importance of each input.

    The output of this function is a float if `x` and `self.weights[1:]` are 1-D arrays, or a numpy array if `x` and `self.weights[1:]` are 2-D arrays.
    
    Parameters
    ----------
    x: {array-like}, shape = [n_samples, n_features]
    Training vectors, where n_samples is the number of samples and n_features is teh number of features
    
    Returns
    ----------
    net_input: numpy.float64 or np.ndarray
    """
    dot_product = np.dot(x, self.weights[1:]) # it calculates the dot product of the input and the weights
    bias = self.weights[0]
    return dot_product + bias
  
  def predict(self, x:np.ndarray)->np.ndarray:
    """
    Return class label after unit step
    
    This method is used to predict the class label of an input vector 'x'.
    
    It first calculates the net input by taking the dot product of the input vector 'x' and the weights of the perceptron.
    The net input is then passed through a unit step function, which is implemented using numpy's 'where' function.
    
    The 'where' function checks if the net input is greater than or equal to 0. If it is, it returns 1, otherwise it returns -1.
    This means that the perceptron is using a binary classification scheme where class labels are either 1 or -1.
    
    The output of this method is a numpy array containing the predicted class labels for each input vector in 'x'.
    
    Parameters
    ----------
    x: {array-like}, shape = [n_samples, n_features]
    Training vectors, where n_samples is the number of samples and n_features is teh number of features
    
    Returns
    ----------
    class_labels: numpy.ndarray
    """
    return np.where(self.net_input(x) >= 0.0,1,-1)