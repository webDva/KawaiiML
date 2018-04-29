def gradientdescent(X, Y, weight = 0, bias = 0, epochs = 1000, gamma = 0.00001):
    """Finds the gradient of the mean squared error cost function for a set of training samples.
    
    X: a list of the x-components of a two-dimensional training set
    Y: a list of the y-components of a two-dimensional training set
    weight: the initial weight hyperparameter of the hypothesis
    bias: the initial bias hyperparameter of the hypothesis
    epochs: number of iterations to train for
    gamma: the learning rate

    returns: a tuple of the new weight and bias 
    """

    # initialize the h0 and h1 hyperparameters to the supplied weight and bias
    current_h0 = weight
    current_h1 = bias

    # partial derivatives of the two hyperparameters
    df0 = lambda x, y, m, b: -2 * x * (y - (m * x + b))
    df1 = lambda x, y, m, b: -2 * (y - (m * x + b))

    # compute the gradient descent
    for epoch in range(epochs):
        previous_h0 = current_h0
        previous_h1 = current_h1
        for i in range(len(X)):
            current_h0 += -gamma * df0(X[i], Y[i], previous_h0, previous_h1)
            current_h1 += -gamma * df1(X[i], Y[i], previous_h0, previous_h1)

    return (current_h0, current_h1)