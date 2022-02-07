#' Make predictions using the neural network.
#' @param network An object of class `mlp` used to make predictions.
#' @param newdata An object of class `data.frame` with the same input columns as the data frame used to create the network.
#'    If left NULL, the training data will be used.
#'    If set to "valid", the validation data will be used.
#' @param type If set to "numeric", then predicted numeric values (regression), probabilities (logistic), or a matrix of probability for each class (classification) is returned. For a classification network, type="labels" can be used to return the class with the highest probability.
#' @return A vector (or matrix, for a classification network) of the predicted outputs.
#' @examples
#'    predict(network, type="labels")
#'    predict(network, newdata="valid", type="labels")
#'    predict(network, newdata=test.data)
#' @export
predict.mlp <- function(network, newdata="train", type="numeric")
{
    # Does the user want to use the training data?
    if (is.null(newdata) || newdata == "train")
    {
        newdata = network$train$X
    }
    # Is new data provided? Process it into a matrix and apply normalization
    else if (is.data.frame(newdata))
    {
        newdata = split.dataset(network$formula, newdata)$X
        newdata = (newdata - network$train$xbar)/network$train$sigma
    }
    # Assume user wants to use the validation data.
    else
    {
        newdata = network$valid$X
    }
    
    # Perform the forward propagation on the processed data
    Y = forward.prop(network, newdata, FALSE)
    
    # For a regression network, un-scale the output
    if (network$type == "regression")
        Y = Y * network$train$sigma.y + network$train$xbar.y

    # This is used to put labels on the output
    # Instead of returning a one-hot encoded Y vector
    if (type == "labels" | type == "class" | type == "response")
    {
        # Figure out which label index each Y prediction has, i.e. 1, 2, 3 ...
        index = apply(Y, 2, which.max)
        # Convert that into one of the labels
        Y = network$levels[index]
    }
    
    return(Y)
    
}
