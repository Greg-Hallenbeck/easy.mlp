#' Train a Neural Network
#' @param network An object of class `mlp` to be trained.
#' @param epochs The number of epochs to train for.
#' @return An object of class `mlp` representing the input network after training.
#' @examples
#'     network <- train(network, 100)
#'     net.trained <- train(net, 100)
train <- function(network, epochs=1000)
{
    # These variables just set to make the code more readable
    # They're all defined in create.mlp()
    m             = network$train$m
    beta          = network$train$momentum$beta
    beta2         = network$train$rmsprop$beta
    epsilon       = network$train$rmsprop$epsilon
    learning.rate = network$train$learning.rate
    batch.size    = network$train$batch.size
    
    # How many mini-batches in the dataset
    num.batches = ceiling(m/batch.size)

    # Loop over the number of training epochs
    for (iter in 1:epochs)
    {
        # Increment the number of epochs
        network$train$epochs = network$train$epochs + 1
        
        # Loop over the different mini-batches
        for (j in 1:num.batches)
        {
            # starting and ending indices in the dataset
            start = (j-1)*batch.size+1
            stop  = ifelse(j*batch.size <= m, j*batch.size, m)
        
            # Select the current training set
            X = network$train$X[,start:stop]
            Y = network$train$Y[,start:stop]
     
            # Perform forward propagation and return the "cache"
            # which contains all the intermediate values
            cache = forward.prop(network, X, return.cache=TRUE)
            
            # Perform backwards propagation
            # This requires looping over the network layers from last layer to first
            for (i in network$layers:1)
            {
                # Here I'm using the notation "dQ" = dLoss/dQ
                # Formulae and notation taken from deeplearning.ai's Neural Network course
                
                # Last layer has a special calculation for dZ
                # because it uses a different activation
                if (i == network$layers)
                    # dZ(final) = Yhat - Y
                    dZ = cache$A[[i]] - Y
                else
                    # dZ(i) = matrix multiply(W, dZ) * activation'(Z(i))
                    dZ = t(cache$W[[i+1]]) %*% dZ * network$activation(cache$Z[[i]], deriv=TRUE)
            
                # First layer has a special calculation for dW
                # because A(0) = X
                if (i != 1)
                    # dW(i) = 1/m * matrix multiply(dZ, A(i-1))
                    dW = 1/m * dZ %*% t(cache$A[[i-1]])
                else
                    # dW(i) = 1/m * matrix multiply(dZ, X)
                    dW = 1/m * dZ %*% t(X)
            
                # dB is always the same, just the average of each row of dZ.
                dB = 1/m * rowSums(dZ)
        
                ## Gradient Descent with Momentum ###########
                
                # Update momentum:
                # vdW = beta * vdW + (1-beta)*dW
                network$train$momentum$vdW[[i]] = beta * network$train$momentum$vdW[[i]] + (1-beta)*dW
                # vdB = beta * vdB + (1-beta)*dB
                network$train$momentum$vdB[[i]] = c(beta * network$train$momentum$vdB[[i]] + (1-beta)*dB)
            
                #vdW = network$train$momentum$vdW[[i]]
                #vdB = network$train$momentum$vdB[[i]]
                
                # Correct the momentum for the first few epochs from a bias effect
                # vdW_C = vdW/(1-beta^epochs)
                vdWcorr = network$train$momentum$vdW[[i]]/(1-beta**network$train$epochs)
                # vdB_C = vdB/(1-beta^epochs)
                vdBcorr = network$train$momentum$vdB[[i]]/(1-beta**network$train$epochs)
            
                ## RMSprop Gradient Descent #################
                
                # Update S values
                # sdW = beta2 * sdW + (1-beta2)*dW^2
                network$train$rmsprop$sdW[[i]] = beta2 * network$train$rmsprop$sdW[[i]] + (1-beta2)*dW**2
                # sdB = beta2 * sdB + (1-beta2)*dB^2
                network$train$rmsprop$sdB[[i]] = beta2 * network$train$rmsprop$sdB[[i]] + (1-beta2)*dB**2
            
                # Correct sdW and sdB in exactly the same way as we corrected vdW and vdB
                sdWcorr = network$train$rmsprop$sdW[[i]]/(1-beta2**network$train$epochs)
                sdBcorr = network$train$rmsprop$sdB[[i]]/(1-beta2**network$train$epochs)
            
                ## Adam Gradient Descent #########################
                
                # Update W and B, using momentum (vdW, vdB, sdW, sdB) and the instead of the raw dW, dB values.
                # Adam is just the combination of momentum and RMSprop
                # W = W - alpha*vdW/sqrt(sdW+epsilon)
                # B = B - alpha*vdB/sqrt(sdB+epsilon)
                network$W[[i]] = network$W[[i]] - learning.rate * vdWcorr / (sqrt(sdWcorr)+epsilon)
                network$B[[i]] = network$B[[i]] - learning.rate * vdBcorr / (sqrt(sdBcorr)+epsilon)
                
                # As a note:
                # "normal" gradient descent would just be
                # W = W - alpha*dW
                # B = B - alpha*dB
            
            } # end loop over layers
            
        } # end loop over mini-batches

        # We've completed a whole epoch!
        
        # Calculate losses on the training and validation datasets
        # These get stored in a vector for later visualization
        yhat   = forward.prop(network)
        yhat.v = forward.prop(network, network$valid$X)
        
        network$train$J = append(network$train$J, network$loss(yhat, network$train$Y))
        network$valid$J = append(network$valid$J, network$loss(yhat.v, network$valid$Y))
        
        # For classification datasets, we also calculate the accuracy for both sets
        if (network$type == "classification")
        {
            yhat.label = predict.mlp(network, type="labels")
            acc = sum(yhat.label == network$train$raw.Y)/length(yhat.label)
            network$train$accuracy = append(network$train$accuracy, acc)
            
            yhat.label = predict.mlp(network, "valid", type="labels")
            acc.v = sum(yhat.label == network$valid$raw.Y)/length(yhat.label)
            network$valid$accuracy = append(network$valid$accuracy, acc.v)
        }
        
    }
    
    return(network)
}
