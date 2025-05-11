"""
R_squared = 1 - SS_r/SS_t = 1 - SUM(datapoint-modelpoint)^2/SUM(datapoint-averagedatapoint)^2

SS_r - residual sum of squares 
SS_t - total sum of squares

So 
* If model is just the mean of the data, then SS_r/SS_t = 1 -> R_squared = 0 -> bad model 
* If model is worse than the mean of the data, then SS_r/SS_t > 1 -> R_squared < 0 -> awful model
* If model is better than the mean of the data, then SS_r/SS_t < 1 -> 0 < R_squared < 1 -> good model 

* Anything above 0.7 is good. 
* Anything above 0.9 is amazing. 
* 1.0 is suspicious and means you're over fit.
"""

"""
Adjusted R_squared

* We may want to add more features to a model to get a better R_Squared value.
* This will work because adding more terms to a linear model never decreases R_squares, 
it either increases it or leaves it the same. 
* To see this, note that the model is designed to minimise the SS_r, so either this 
stays the same  (because the extra term has a zero coefficient) or it decreases 
(becuase the extra term has a non-zero coefficient). 
* But we don't want to end up with a model that has loads of terms just to increase R_squared by a bit. 
* For this reason, we design Adj_R_squared

Adj_R_squared = 1 - (1-R_squared) * (n - 1)/(n - k - 1)

Here 
* n is the number of datapoints 
* k is the number of independent variables

So as we see, if k is larger, then R_squared needs to have 
grown substantially, else Adj_R_squared will have got smaller. 
So this penalises adding loads more features just to get a slightly larger R_squared.
"""