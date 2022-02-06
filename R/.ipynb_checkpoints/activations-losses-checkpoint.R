#' A rectified linear unit or RELU = max(0, x) function, and its derivative.
#' @param X A matrix.
#' @param deriv Set this to TRUE if you want the derivative instead of the function.
#' @return A matrix representing the RELU(X), or its derivative.
relu <- function(X, deriv=FALSE)
{
    if (!deriv) return( pmax(X, 0) )
    
    return( ifelse(X > 0, 1, 0) )
}

#' A "leaky" RELU function, and derivative. The slope in the negative region is 0.1.
#' @param X A matrix.
#' @param deriv Set this to TRUE if you want the derivative instead of the function.
#' @return A matrix representing the leaky RELU(X), or its derivative.
leaky.relu <- function(X, deriv=FALSE)
{
    if (!deriv) return(ifelse(X > 0, X, -0.1*X))
    
    return(ifelse(X > 0, 1, 0.1))
}

#' A hyperbolic tangent function, and its derivative.
#' @param X A matrix.
#' @param deriv Set this to TRUE if you want the derivative instead of the function.
#' @return A matrix representing the tanh(X), or its derivative.
tanh <- function(X, deriv=FALSE)
{
    p = exp(X)
    m = exp(-X)
    
    if (!deriv) return((p - m)/(p + m))
        
    return(1-(p-m)**2/(p+m)**2)
}

#' A linear activation function, and its derivative.
#' @param X A matrix.
#' @param deriv Set this to TRUE if you want the derivative instead of the function.
#' @return X, or its derivative (a matrix of all 1s).
linear <- function(X, deriv=FALSE)
{
    if (deriv) return(rep(1.0, length(X)))#matrix(1.0, length(X), nrow=nrow(X), ncol=ncol(X)))
    
    return(X)
}

#' A sigmoid (= logistic) function, and its derivative.
#' @param X A matrix.
#' @param deriv Set this to TRUE if you want the derivative instead of the function.
#' @return A matrix representing the sigmoid(X), or its derivative.
sigmoid <- function(X, deriv=FALSE)
{
    s = (1+exp(-X))**-1

    if (!deriv) return(s)
    
    return(s*(1-s))
}

#' A softmax activation, used for classifier networks as the output layer.
#' @param X A matrix.
#' @param deriv Set this to TRUE if you want the derivative instead of the function.
#' @return An matrix representing the probability of each class for each observation.
softmax <- function(X, deriv=FALSE)
{
    # Calculate Boltzmann Factor (e^X) for each entry
    boltz = exp(X)
    
    # Output is each Boltzmann Factor divided by the partition function (sum of the factors)
    s = boltz/matrix(rep(colSums(boltz),nrow(X)), nrow=nrow(X), byrow=TRUE)
        
    return (s)
}

#' A Mean Squared Error Loss function, used for regression networks.
#' @param A a vector of predicted Y values.
#' @param Y the true Y values.
#' @return The floating-point loss for the network.
mse.loss <- function(A, Y)
{
    loss = mean((Y-A)**2)
    return(loss)
}
    
#' A Logistic Loss function, used for binary classification networks.
#' @param A a vector of predicted Y values.
#' @param Y the true Y values.
#' @return The floating-point loss for the network.
logit.loss <- function(A, Y)
{
    loss = -1/length(Y)*sum(Y*log(A)+(1-Y)*log(1-A))
    return(loss)
}

#' Categorical Cross-Entropy Loss function, used for classification networks
#' @param A a matrix of probabilities.
#' @param Y the true Y values.
#' @return The floating-point loss for the network.
cce.loss <- function(A, Y)
{
    # Sums -Y_i log A_i over all i categories
    loss = -1/ncol(Y)*sum(Y*log(A))
    return(loss)
}