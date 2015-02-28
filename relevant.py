"""
Attached to this email is a data set, ds_test_final.csv. The data set includes the following pieces of information:
• Column 1: Observed dependent variable
• Column 2: Fitted Residuals
• Column 3: Fitted Values
• The remainder of the columns: Predictors (not all predictor variables used in the model included in the model are included in the included data set).

To the best of your ability with the data from the file, please carry out the following tasks in Python (using whatever packages and tools you feel are the most appropriate to fully answer the following questions:
1. Load and summarize the included data.
2. Determine a reasonable measure of model quality, and use the included data to compute and report this statistic.
3. Do you consider it a good model? Why or why not?
4. Examine the columns of predictor variables. Note any predictors that should be transformed, dropped or interacted to improve the model.
Include ALL work as a python script. Explanations and answers to direct questions should be included as comments in the script.
"""

import pandas as pd
df = pd.DataFrame.from_csv('ds_test_final.csv')

#1. Below are statistics about the data table.
#   The table has 6 columns (4 indicator or predictor or "feature" variables) 
#   There are no null or NaN values in the table
# 
#   V6 is an integer value (but is processed here as a float) between 10005 and 7322564, inclusive. 
#      The quartiles for V6 indicate that it may be log-normally distributed (many more low values that large values in the fat tail)
#   V76 May also be an integer value between 1 and 4, inclusive. 
#      The quantiles indicate that it may be log-normally distributed (many more low values that large values in the fat tail)
#   The fitted values are in the range 888.28 +/- 232.82 (1-sigma)



df.describe()
#
#        Fitted_residuals  Fitted_Values              V6          V23  \
# count       2215.000000    2215.000000     2215.000000  2215.000000   
# mean           0.023772     888.280968    53117.984199     6.801445   
# std          569.907172     232.829326   204620.252853     4.700335   
# min        -1379.053409    -109.411628    10005.000000     0.180000   
# 25%         -472.148069     746.587998    14366.000000     3.270000   
# 50%          -42.758007     904.865463    22792.000000     5.610000   
# 75%          460.802717    1039.812797    43024.000000     9.105000   
# max         1399.255154    1690.290748  7322564.000000    44.820000   
#
#                V34          V76  
# count  2215.000000  2215.000000  
# mean     11.620537     2.640632  
# std       8.600352     0.512686  
# min       0.640000     1.000000  
# 25%       4.510000     2.000000  
# 50%       9.330000     3.000000  
# 75%      16.905000     3.000000  
# max      58.000000     4.000000  

df.isnull().sum()
# Fitted_residuals    0
# Fitted_Values       0
# V6                  0
# V23                 0
# V34                 0
# V76                 0
# dtype: int64

from matplotlib import pyplot as plt
df.hist()
plt.show(block=False)
plt.savefig('histograms.png')