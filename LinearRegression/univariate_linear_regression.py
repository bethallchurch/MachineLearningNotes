from matplotlib import pyplot as plt
import numpy as np

test_training_set = np.array([
  [1, 2],
  [2, 3],
  [2, 4],
  [3, 1],
  [3, 2],
  [4, 3],
  [5, 4],
  [5, 5],
  [6, 4],
  [7, 4],
  [7, 5],
  [8, 6],
  [9, 6]
])

class UnivariateLinearRegression(object):

  def __init__(self, initial_theta_0, initial_theta_1, training_set, learning_rate=0.01, iterations=500):
    """
    Initialise values.
    """
    self.theta_0 = initial_theta_0
    self.theta_1 = initial_theta_1
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.X = training_set[:, 0]
    self.Y = training_set[:, 1]

  def fit(self, graph=False):
    # Start walking down the cost function.
    for _ in range(self.iterations):
      self.gradient_descent()
    # Print the result.
    print(self.theta_0, self.theta_1)
    if graph:
      # Show the final fit on a graph.
      self.plot_hypothesis()

  def hypothesis(self, x):
    """
    For a given x, predict y.
    """
    # Formula for a linear function.
    return self.theta_0 + (self.theta_1 * x)

  def sum_squared_error(self):
    """
    Implementation of sum of squared errors.
    """
    m = len(self.X) # Number of samples in the training set.
    # Run through all input variables and predict y values for them.
    predicted_Y = self.hypothesis(self.X)
    # Calculate the differences between the predicted value and the actual value.
    error = sum((self.Y - predicted_Y) ** 2)
    # Find the mean error.
    mean_error = error / m
    # Divide by 2 for convenience when calculating gradient descent.
    return mean_error / 2

  def theta_0_derivative(self):
    """
    The partial derivative of the cost function with respect to theta 0.
    """
    m = len(self.X) # Number of samples in the training set.
    # The predicted output variables.
    predicted_Y = self.hypothesis(self.X)
    # The difference.
    difference = sum(predicted_Y - self.Y)
    # The average difference.
    average_difference = difference / m
    return average_difference

  def theta_1_derivative(self):
    """
    The partial derivative of the cost function with respect to theta 1.
    """
    m = len(self.X) # Number of samples in the training set.
    # The predicted output variables.
    predicted_Y = self.hypothesis(self.X)
    # The difference.
    difference = sum((predicted_Y - self.Y) * self.X)
    # The average difference.
    average_difference = difference / m
    return average_difference

  def gradient_descent(self):
    """
    One iteration of the gradient descent algorithm.
    """
    # Calculate the new value for theta_0.
    new_theta_0 = self.theta_0 - self.learning_rate * self.theta_0_derivative() * self.sum_squared_error()
    # Calculate the new value for theta_1.
    new_theta_1 = self.theta_1 - self.learning_rate * self.theta_1_derivative() * self.sum_squared_error()
    # Update theta_0 and theta_1 simultaneously.
    self.theta_0 = new_theta_0
    self.theta_1 = new_theta_1

  def plot_hypothesis(self):
    """
    Show the training set and hypothesis function on the same graph.
    """
    # All the predicted y values, as given by the hypothesis function.
    predicted_Y = self.hypothesis(self.X)
    # Plot the training set.
    plt.plot(self.X, self.Y, 'rx')
    # Plot the hypothesis.
    plt.plot(self.X, predicted_Y)
    # Make sure the axes start and stop in the right places.
    plt.axis([0, max(self.X) + 1, 0, max(self.Y) + 1])
    # Show the graph.
    plt.show()