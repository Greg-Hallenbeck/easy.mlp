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

You can install the released version of easy.mlp from [CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("easy.mlp")
```

## Example

This is a basic example which shows you how to solve a common problem:


```r
source("R/*")
#> Error in library(easy.mlp): there is no package called 'easy.mlp'

data(iris)

network <- create.mlp(Species ~ ., data=iris, hidden=c(5,5,5))
#> Error in create.mlp(Species ~ ., data = iris, hidden = c(5, 5, 5)): could not find function "create.mlp"

network <- train(network, 100)
#> Error in train(network, 100): could not find function "train"

plot(network)
#> Error in plot(network): object 'network' not found
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
