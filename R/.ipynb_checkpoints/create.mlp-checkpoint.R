#' Initialize a Neural Network.
#' @description Create a new neural network based on a formula and data frame.
#'
#' This function creates a new neural network with randomized weights to be trained with the `train()` function.
#' @param formula An object of class `formula` describing the model to be fit.
#' @param data A `data.frame` object containing the training and validation data.
#' @param hidden A vector indicating how many layers and hidden notes the network will have. Passing c(4,5,2) creates a network with 3 hidden layers with 4, 5, and 2 nodes, respectively.
#' @param type One of "regression", "logistic", or "classification".
#' @param activation The name of a function to be used for activations in the hidden layers. "leaky.relu", "relu", "tanh", and "sigmoid" are included in the package.
#' @param valid.split The fraction of the data to be used for validation.
#' @param randomize If TRUE, the data frame will be randomly sampled before train/validation split.
#'     If not, the final rows are the validation data.
#' @param learning.rate The learning rate of the network.
#' @param momentum.beta Beta parameter for gradient descent with momentum.
#' @param rmpsprop.beta Beta parameter for RMSprop gradient descent.
#' @param batch.size Minibatch size. Choosing 1 is equivalent to stochastic gradient descent.
#' @return An object of class `mlp`
#' @examples
#'  create.mlp(Species ~ ., data=iris, hidden=c(5,5,5), type="classification")

create.mlp <- function(formula, data=NULL, hidden=c(), type=NULL,
        activation="leaky.relu", valid.split = 0.25, randomize=TRUE,
        learning.rate = 0.001, momentum.beta=0.9, rmsprop.beta=0.999, batch.size=64)
{
    # The network will contain *everything* about the network
    network = list()

    # Does the user want us to randomize the order?
    if (randomize) data = data[sample(nrow(data)),]
    
    network$formula = formula
    dataset = split.dataset(network$formula, data)
    X = dataset$X
    Y = dataset$Y
    
    # Determine the type of network, if not specified
    # This is done by inspecting the type of the raw Y data
    if (is.null(type))
    {
        if (is.factor(dataset$raw.Y))
            type = "classification"
        else if (is.logical(dataset$raw.Y) || (max(dataset$raw.Y) <= 1.0 && max(dataset$raw.Y) >= 0))
            type = "logistic"
        else
            type = "regression"
    }
    
    # Do a train/valid split
    nTrain = floor(ncol(X)*(1-valid.split))
    
    network$valid$raw.X = X[,(nTrain+1):ncol(X)]
    network$valid$Y = matrix(dataset$Y[,(nTrain+1):ncol(X)], nrow=nrow(dataset$Y))
    #network$valid$raw.Y = dataset$raw.Y[(nTrain+1):ncol(X)]
    X = X[,1:nTrain]
    Y = matrix(Y[,1:nTrain], nrow=nrow(dataset$Y))
    
    # Number of features in the input
    network$nx = nrow(X)
    
    # Normalize the X data
    network$train$raw.X = X
    network$train$sigma = apply(X, 1, sd)
    network$train$xbar  = apply(X, 1, mean)
    network$train$X = (X - network$train$xbar)/network$train$sigma
        
    # If output is a factor, then "raw.Y" still has labels, "Y" is one-hot encoded.
    network$valid
    network$train$raw.Y = dataset$raw.Y[1:nTrain]
    network$train$Y = Y

    # Number of training samples
    network$train$m = ncol(X)
    
    # Normalize the X and Y for the validation set as well
    network$valid$X =(network$valid$raw.X - network$train$xbar)/network$train$sigma
    network$valid$raw.Y = dataset$raw.Y[(nTrain+1):length(dataset$raw.Y)]
    #network$valid$Y = network$valid$raw.Y
    
    # Learning rate, minibatch size, other parameters for learning
    network$train$learning.rate = learning.rate
    network$train$batch.size = batch.size
    
    # The averaged loss, tracked over training
    network$train$J = c()
    network$valid$J = c()
    network$train$epochs = 0
    
    # Activation for hidden layers
    network$activation = match.fun(activation)
    
    # Determine loss and output activation function based on the output activation
    network$type = type
    if (type == "logistic")
    {
        network$output.activation = sigmoid
        network$loss = logit.loss # Logistic Loss for binary classification
        network$Nout = 1 # Only need one output node
    }
    else if (type == "regression")
    {
        network$output.activation = linear
        network$loss = mse.loss   # Mean Squared Error for regression
        network$Nout = 1 # Only need one output node
        
        # Normalize the output as well
        network$train$sigma.y = sd(Y)
        network$train$xbar.y  = mean(Y)
        network$train$Y = (Y - network$train$xbar.y)/network$train$sigma.y
        network$valid$Y = (network$valid$Y - network$train$xbar.y)/network$train$sigma.y
    }
    else # assume out is "classification"
    {
        network$output.activation = softmax
        network$loss = cce.loss   # Categorical Cross-Entropy loss for multinomial classification
        network$Nout = nrow(Y)
        
        network$levels = dataset$levels

        network$train$accuracy = c()
        network$valid$accuracy = c()
    }

    # Add an output layer with a single node to the list of hidden layers
    layers = append(hidden, network$Nout)

    # This variable is a list of "sizes", indicating what the dimension of each layer is
    # The first size needs to be the number of features in X
    sizes = append(layers, network$nx, 0)

    network$nodes = layers
    network$layers = length(layers)

    
    # Empty lists which will hold the matrices of weights and biases
    weights = list()
    biases  = list()

    # Empty lists which will hold the momentum values
    vweights = list()
    vbiases  = list()
    
    # Empty lists which will hold the RMSprop S values
    sweights = list()
    sbiases  = list()
    
    # Loop over hidden layers to initialize the W and B (weight and bias) matrices
    for (i in 1:length(layers))
    {
        # The W matrix is n(i) x n(i-1)
        # The B matrix is n(i) x 1
        rows = sizes[i+1]
        cols = sizes[i]
    
        # Initialize W with random weights
        # This is necessary for symmetry breaking during backpropagation
        W = matrix(rnorm(rows*cols,0,0.1), nrow=rows, ncol=cols)
    
        # Initialize B with all 0s
        B = numeric(rows)
    
        # Initialize the momentum for W and B
        vdW = matrix(numeric(rows*cols), nrow=rows, ncol=cols)
        vdB = numeric(rows)
        # Initialize the RMSprop S values for W and B
        sdW = matrix(numeric(rows*cols), nrow=rows, ncol=cols)
        sdB = numeric(rows)
        
        # Add them to the lists
        weights[[i]]  = W
        biases[[i]]   = B
        
        vweights[[i]] = vdW
        vbiases[[i]]  = vdB
        
        sweights[[i]] = sdW
        sbiases[[i]]  = sdB
    }

    network$W = weights
    network$B = biases
 
    network$train$momentum = list()
    network$train$momentum$beta = momentum.beta
    network$train$momentum$vdW = vweights
    network$train$momentum$vdB = vbiases
    network$train$rmsprop = list()
    network$train$rmsprop$beta = rmsprop.beta
    network$train$rmsprop$epsilon = 1e-6
    network$train$rmsprop$sdW = sweights
    network$train$rmsprop$sdB = sbiases

    class(network) = "mlp"

    return(network)
}
