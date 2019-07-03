# python_bidirectional_stepwise_selection
Automated Bidirectional Stepwise Selection 

This script is about the automated bidirectional stepwise selection. You can easily apply on Dataframes. This function returns not only the final features but also elimination iterations, so you can track what exactly happened at the iterations.

You can apply it on both Linear and Logistic problems and easily define stay and enter significance limits. Eliminations can be applied with Akaike information criterion (AIC), Bayesian information criterion (BIC), R-squared (Only works with linear), Adjusted R-squared (Only works with linear). Also, you don't have to worry about varchar variables, the code will handle it for you.
Enjoy the code!

Plase let me know if there is any logic erros or bugs in the code !

Required Libraries: pandas, numpy, statmodels

See more about stepwise regression : https://en.wikipedia.org/wiki/Stepwise_regression
