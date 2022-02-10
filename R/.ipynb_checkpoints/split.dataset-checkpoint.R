#' Split a dataset into a design (X) matrix and labels (Y), and one-hot encodes factors. Does not need to be called by the user.
#' @param formula An object of class `formula` describing the model to be fit, to determine X and Y.
#' @param data A `data.frame` containing the dataset to be split.
#' @return A list containing the design matrix (X), and the Y values (one-hot encoded as necessary).
#'     If Y is a factor, then raw.Y is still factors, and levels contains the levels that the factor can take.
split.dataset <- function(formula, data)
{    
    # This function also gets called when Y is not defined in the data
    # (e.g. when running the model on new data or testing data)
    # This code checks for that and adds a dummy Y full of 1s.
    yvar = as.character(formula[[2]])
    if (! (yvar %in% names(data)) )
        data[,yvar] = 1
    
    # Make a new dataframe only including Y and X variables, in that order.
    dataset <- model.frame(formula, data=data)
    
    # Split out the X and Y variables
    raw.Y = dataset[,1] # Split out the Y variable
    Y = mltools::one_hot(data.table::as.data.table(raw.Y)) # one-hot encode the Y, if its a factor
    Y = t(as.matrix(Y)) # Convert to a row vector
    rownames(Y) = NULL # Strip the column and row names from the matrix
    colnames(Y) = NULL

    # If the dataset has factors, make sure to store what they are
    # so that when making predictions later, we can decode them
    if (is.factor(raw.Y))
        levels = factor(levels(raw.Y), levels(raw.Y))
                    
    X = dataset[,2:ncol(dataset)] # Split out the X variables
    X = mltools::one_hot(data.table::as.data.table(X)) # one-hot encode any factors
    X = t(as.matrix(X)) # Convert to a row vector (each column is one observation)
    rownames(X) = NULL # Strip the column and row names from the matrix
    colnames(X) = NULL
    
    return(list(X=X, Y=Y, raw.Y=raw.Y, levels=levels))
}

    # Pre-process and create a training X and Y datasets
    # This code has been taken from the lm() function
    # (which has been released under the GPL 2)
    
    # At its end, dataset is a table where column 1 = Y, and the remaining columns are X
    #mf <- match.call(expand.dots = FALSE)
    #m <- match(x = c("formula", "data"), table = names(mf), nomatch = 0L)
    #print(m)
    #mf <- mf[c(1L, m)]
    #mf$drop.unused.levels <- TRUE
    #mf[[1L]] <- quote(stats::model.frame)
    #dataset <- eval(expr = mf, envir = parent.frame())
    #mf <- eval(expr = mf, envir = parent.frame())
    #print(mf)
    #mt <- attr(mf, "terms") # allow model.frame to update it
    #dataset <- model.matrix(mt, mf)
