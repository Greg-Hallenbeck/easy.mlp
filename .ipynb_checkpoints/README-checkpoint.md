---
output: github_document
---

<!-- README.md is generated from README.Rmd. Please edit that file -->



# easy.mlp

<!-- badges: start -->
<!-- badges: end -->

The goal of easy.mlp is to quickly and easily build a neural network to fit tabular data.

* Supports the usage of the R `formula` class.
* Normalizes input and output data to the range of 0-1.
* One-hot encodes input factor data.
* Allows for any number of hidden layers and nodes.
* Supports regression, classification, and binary classification networks.
* Multiple pre-defined activation functions (and supports user-defined functions).
* Mini-batch gradient descent using the ADAM optimizer.
* Randomly splits data into training and validation data.
* Simple plotting of metrics and predictions for new data.

## Installation

You can install the released version of easy.mlp from github with:

``` r
library(devtools)

install_github("Greg-Hallenbeck/easy.mlp")
```

## Example

This is a basic example which shows you how to solve a common problem:


```r
library(easy.mlp)

data(iris)

# Initial network creation is stochastic.
set.seed(8675309)
net <- create.mlp(Species ~ ., data=iris, hidden=c(5,5,5), type="classification")

# This line can be run multiple times to train another 1,000 epochs
net <- train(net, 1000)

# Plot the loss and accuracy of the network
par(mfrow=c(1,2))
options(repr.plot.width=10, repr.plot.height=5.5)

plot(net, ylim=c(0.03, 2))
plot(net, metric="accuracy", ylim=c(0,1))
```

<img src="man/figures/README-example-1.png" title="plot of chunk example" alt="plot of chunk example" width="100%" />

```r

# Predict species for a new data point.
```

What is special about using `README.Rmd` instead of just `README.md`? You can include R chunks like so:


```r
summary(cars)
#>      speed           dist       
#>  Min.   : 4.0   Min.   :  2.00  
#>  1st Qu.:12.0   1st Qu.: 26.00  
#>  Median :15.0   Median : 36.00  
#>  Mean   :15.4   Mean   : 42.98  
#>  3rd Qu.:19.0   3rd Qu.: 56.00  
#>  Max.   :25.0   Max.   :120.00
```

You'll still need to render `README.Rmd` regularly, to keep `README.md` up-to-date. `devtools::build_readme()` is handy for this. You could also use GitHub Actions to re-render `README.Rmd` every time you push. An example workflow can be found here: <https://github.com/r-lib/actions/tree/master/examples>.

You can also embed plots, for example:

<img src="man/figures/README-pressure-1.png" title="plot of chunk pressure" alt="plot of chunk pressure" width="100%" />

In that case, don't forget to commit and push the resulting figure files, so they display on GitHub and CRAN.
