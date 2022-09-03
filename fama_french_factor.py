######################################################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm

######################################################################################################

def FamaFrench_analysis(returns, factors):
    """

    Returns the loadings of returns on the Fama French Factors:

    1- The Fama-French model is a three-factor model which enhance the CAPM, the three factors are:

            The market risk (i.e., as in the CAPM),
            The outperformance of small versus big companies,
            The outperformance of high book/market versus small book/market companies.
  
    2- Fama and French take the entire universe of stocks an put them into ten buckets and
       sorted such deciles in two ways.

            A first sorting was done according to the size, i.e., the market capitalization
            and then they compared the performance of the bottom 10% companies versus the top
            10% companies according to the size.
            The second sorting was done according to the book-to-market ratios (B/P ratio)
            and then they did the same, i.e., they looked at the performance of the bottom
            10% companies (Growth Stocks) versus the top 10% companies (Value Stocks).

    3- The index of returns must be a (not necessarily proper) subset of the index of factors.
    
    4- Returns is either a Series.

    5- The "regression" function runs a linear regression to decompose the
       dependent variable into the explanatory variables.

    """

######################################################################################################

    def regression(dependent_variable, explanatory_variables, alpha=True):
        if alpha:
            explanatory_variables = explanatory_variables.copy()
            explanatory_variables["Alpha"] = 1
        
        lm = sm.OLS(dependent_variable, explanatory_variables).fit()
        return lm

######################################################################################################

    if isinstance(returns, pd.Series):
        dependent_variable = returns
        explanatory_variables = factors.loc[returns.index]
        tilts = regression(dependent_variable, explanatory_variables).params
    else:
        raise TypeError("returns must be a Series")
    return tilts