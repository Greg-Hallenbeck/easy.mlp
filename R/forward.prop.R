#' Performs forward propagation through the neural network
#' Most users should use predict.mlp() as a wrapper
#' @param network An object of class `mlp` containing the neural network.
#' @param X The design matrix (if set to NULL), or a pre-processed matrix of input data.
#' @param return.cache if TRUE, instead of the final activation, the network returns all intermediate activations.
#'     Used for backwards propagation.
#' @return The activations of the final layer of the network. If return.cache is TRUE, then all intermediate layer activations are instead returned in a list.
forward.prop <- function(network, X=NULL, return.cache=FALSE)
{
    # The neural network is calculated as a series of layers
    # with Z as the linear output of each layer and A as the activation on that Z
    # In other words, if (i) is the current layer, and W and B are the weights and biases:
    # Z(i) = W(i)*A(i-1) + B(i)
    # A(i) = activation(Z(i))
    
    # These contain the values of Z and A for each layer, and are used
    # for backpropagation through the network
    Zvals = list()
    Avals = list()
    
    # If nothing is specified, use the training data
    if(is.null(X))
        X = network$train$X
    
    # Initialize the activations to the input
    # i.e. A(0) = X
    A = X
    
    # Loop over the layers
    # But not the last layer, which can have a different activation
    for (i in 1:(network$layers-1))
    {
        # Fail-safe for networks with only one layer (i.e. only an output layer)
        if (network$layers == 1)
            break
                
        # Calculate Z = WX + B
        Z = network$W[[i]] %*% A + network$B[[i]]
        
        # Pass the output Z through an activation function
        # Which then is the input to the next layer
        A = network$activation(Z)
        
        Zvals[[i]] = Z
        Avals[[i]] = A
    }
    
    # Calculate the final layer's output
    Z = network$W[[network$layers]] %*% A + network$B[[network$layers]]
    A = network$output.activation(Z)
    # Yhat, e.g. the prediction, is the output layer's activation
    Y = A
    
    Zvals[[network$layers]] = Z
    Avals[[network$layers]] = A
    
    # Returns ALL outputs of activation layers, not just the final one.
    # This is used for backpropagation.
    if (return.cache) return(list(Z=Zvals, A=Avals, W=network$W, Y=Y))
    
    # Otherwise, simply return the output of the last layer, e.g. Yhat
    return(Y)
}
