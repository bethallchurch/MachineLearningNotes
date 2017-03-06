from matplotlib import pyplot as plt

test_training_set = [
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
]

class UnivariateLinearRegression(object):

  def __init__(self, initial_theta_0, initial_theta_1, training_set, learning_rate=0.01, iterations=500):
    """
    Initialise values.
    """
    self.theta_0 = initial_theta_0
    self.theta_1 = initial_theta_1
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.training_set = training_set

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
    Explicit implementation of sum of squared errors function.
    """
    m = len(self.training_set) # Number of samples in the training set.
    total_error = 0
    # For each sample...
    for sample in self.training_set:
      # find the input variable,
      x = sample[0]
      # the actual output variable,
      y = sample[1]
      # and the predicted output variable.
      predicted_y = self.hypothesis(x)
      # Calculate the difference between the predicted value and the actual value.
      # Square it to make it an absolute value.
      difference = (predicted_y - y) ** 2
      # Add it to the sum of the total errors.
      total_error += difference
    # Find the mean error.
    mean_error = total_error / m
    # Divide by 2 for convenience when calculating gradient descent.
    return mean_error / 2

  def theta_0_derivative(self):
    """
    The derivative of ... with respect to theta 0.
    """
    m = len(self.training_set) # Number of samples in the training set.
    total = 0
    for sample in self.training_set:
      # find the input variable,
      x = sample[0]
      # the actual output variable,
      y = sample[1]
      # and the predicted output variable.
      predicted_y = self.hypothesis(x)
      # Find the difference.
      difference = predicted_y - y
      # Add to total.
      total += difference
      # Divide by m.
      return total / m

  def theta_1_derivative(self):
    """
    The derivative of ... with respect to theta 1.
    """
    m = len(self.training_set) # Number of samples in the training set.
    total = 0
    for sample in self.training_set:
      # find the input variable,
      x = sample[0]
      # the actual output variable,
      y = sample[1]
      # and the predicted output variable.
      predicted_y = self.hypothesis(x)
      # Find the difference.
      difference = predicted_y - y
      # Add to total.
      total += (difference * x)
      # Divide by m.
      return total / m


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
    # All the x values in the training set.
    x_values = [sample[0] for sample in self.training_set]
    # All the y values in the training set.
    y_values = [sample[1] for sample in self.training_set]
    # All the predicted y values, as given by the hypothesis function.
    predicted_y_values = [self.hypothesis(x) for x in x_values]
    # Plot the training set.
    plt.plot(x_values, y_values, 'rx')
    # Plot the hypothesis.
    plt.plot(x_values, predicted_y_values)
    # Make sure the axes start and stop in the right places.
    plt.axis([0, max(x_values) + 1, 0, max(y_values) + 1])
    # Show the graph.
    plt.show()