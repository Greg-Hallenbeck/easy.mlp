#' Plot the Loss or Other Metrics of a Neural Network
#' @param network An object of class `mlp` to be plotted.
#' @param metric One of "loss", "accuracy", or "residuals" to be plotted vs. training epoch.
#' @param ylim Y-Axis limits for the plot.

# Plots metrics for the neural network
#
# Inputs:
#
# network - the neural network to plot
# metric  - which metric to plot, either:
#           - "loss": average loss vs. epoch (all types of network)
#           - "accuracy": accuracy vs. epoch (for classification/logistic networks)
#           - "residuals": residuals vs. predictions (for residual networks)
# ylim    - range to plot on the yaxis
# Returns:
#
# Nothing


plot.mlp <- function(network, metric="loss", ylim=NULL)
{
    if (metric == "loss")
    {
        if (is.null(ylim))
            ylim = c(min(c(network$train$J, network$valid$J))*0.9, max(c(network$train$J,network$valid$J))*1.1)
        
        plot(network$train$J, type='l', col=2,
             xlab="Epoch", ylab="Average Loss",
             ylim=ylim,
             log="y", cex.lab=1.5)
        lines(network$valid$J, col=4, lty=2)
        legend("topright", c("Training Set", "Validation Set"), col=c(2,4), bty="n", lty=c(1,2), cex=1.2)
    }
    if (metric == "accuracy")
    {
        if (is.null(ylim))
        {
            ylim= c(min(network$valid$accuracy[floor(network$train$epochs*0.75):network$train$epochs]*0.9),
                    max(network$train$accuracy)*1.05)
        }
        
        plot(network$train$accuracy, type='l', col=2,
             xlab="Epoch", ylab="Accuracy", ylim=ylim)
        lines(network$valid$accuracy, col=4, lty=2)
        abline(h=1, lty=2)
        legend("bottomright", c("Training Set", "Validation Set"), col=c(2,4), bty="n", lty=c(1,2), cex=1.2)
    }
    if (metric == "residuals")
    {
        #yhat = predict.mlp(network)
        #residuals = network$train$raw.Y - yhat
        
        v.yhat = predict.mlp(network, "valid")
        v.residuals = network$valid$raw.Y - v.yhat
        
        plot(v.yhat, v.residuals, xlab="Predicted Value", ylab="Residuals", pch=19, col=4,
             ylim=c(-1.5*max(abs(v.residuals)), 1.5*max(abs(v.residuals))))
        #points(v.yhat, v.residuals, pch=18, col=2)
        abline(0,0, lty=2)
        legend("bottomright", c("Validation Set"), col=c(4), bty="n", pch=c(19), cex=1.2)
    }
}
