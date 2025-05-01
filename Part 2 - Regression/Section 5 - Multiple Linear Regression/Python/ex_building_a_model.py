"""
Building a model. 
There are 5 techniques. 

1. All in
    * Throw all the variables at the model at once. 

2. Backward elimination
    * Select a significance level 
    * Use all in 
    * Consider the predictor with the highest p value
    * If p is greater than significance level, then remove that predictor
    * Fit the model without that variable.
    * Look for variabel with highest p value, and remove if greater than SL
    * Keep repeating until all variables have p value less tahn SL, and then remove them.

3. Forward selection 
    * Select a significance level. 
    * Fit all possible simple linear regression models.
    * Select the one with the lowest P Value.
    * Keep this var, and construct all possible 2-variable mult linear regressions.  
    * Select the one with the lowest P Value.
    * Keep repeating this.
    * Stop when the variable we added ends with a  model with p value greater than SL. 
    Why, because the variable is not significant anymore. 
    * Keep the previous model, because we just added one that wasn't useful anymore. 

4. Bidirectional elimination  
    Combine the two. 
    * Select a signifance level to enter and a SL to stay in teh model 
    * Perform the next step of forward selection (new vairables must have P < SL_ENTER to enter)
    * Perform all steps of Backward elimination (Old variables must have P < SLSTAY to stay)
    * Repeat these two steps 

5. Score comparison
    * Select a criterion of goodness of fit. 
    * Select all possible combinations there are 2^{N-1}
    * Select the one with the best fit.
    * But even when there are 10 predictors there are then 1023 models so this is very resource intense.

2, 3, 4 are sometimes called Stepwise regression

Backward elimination is the fastest.
"""