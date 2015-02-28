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


Sections are delimitted for each question with "#####################"
"""

import pandas as pd
df = pd.DataFrame.from_csv('ds_test_final.csv')

########################################
## Question 1
#
#   `df.describe()` output, pasted below, provides statistics about the data table.
#   Notable data characteristics are described here.
#   The table has 6 columns (4 indicator or predictor or "feature" variables) 
#   There are no null or NaN values in the table
#   V6 
#      An integer value (but is processed here as a float) between 10005 and 7322564, inclusive. 
#      The quartiles and histograms for V6 (and V76) indicate that it may be log-normally distributed (many more low values that large values in the fat tail)
#   V76
#       Integer value between 1 and 4, inclusive. 
#       It may be log-normally distributed however the order from most frequent to less frequent is: 3,2,4,1 
#       The counts in V76 are: [(1, 7), (2, 811), (3, 1368), (4, 29)] or {3: 1368, 2: 811, 4: 29, 1: 7} 
#       So this is may be a categorical rather than ordinal value
#   Fitted Values: are in the range 888.28 +/- 232.82 (1-sigma)
#       The histogram shows mild assymetry in the distribution of fitted values.
#   Fitted Residuals: 0.0237 +/- 569.9 1-sigma
#       The histogram shows significant assymetry, so the model features should be transformed

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

from collections import Counter
v76 = Counter(df['V76'])
sorted(v76.iteritems())
# [(1, 7), (2, 811), (3, 1368), (4, 29)]


########################################
## Question 2

# A simple, common measure of model quality is the std of the error (residuals)
#  This can be normalized by the std or mean of the of the fitted values:
df.Fitted_residuals.std() / df.Fitted_Values.std()
# 2.4477465236730169
df.Fitted_residuals.std() / df.Fitted_Values.mean()
# 0.64158435507137113
#
# For both normalizations the error is an excessively large portion of the predicted value
# So the model is not very effective at predicting the predicted value
#
# Sometimes the correlation (normalized covariance) between the fitted value and the residuals can indicate an additional oportunity to produce a better fit
# but this correlation matrix doesn't seem to indicate that, because the cross correlation value is low
df[df.columns[:2]].corr()
#                   Fitted_residuals  Fitted_Values
# Fitted_residuals          1.000000      -0.000159
# Fitted_Values            -0.000159       1.000000

#df.cov()
#                   Fitted_residuals   Fitted_Values            V6  \
# Fitted_residuals     324794.184847      -21.125900 -1.263300e+03   
# Fitted_Values           -21.125900    54209.494818 -5.504700e+06   
# V6                    -1263.299900 -5504699.807436  4.186945e+10   
# V23                      -0.161758       17.274371  1.038102e+05   
# V34                      -0.276370      113.942164  1.678451e+05   
# V76                      -0.062802      -15.385900 -1.208506e+04   

#                             V23            V34           V76  
# Fitted_residuals      -0.161758      -0.276370     -0.062802  
# Fitted_Values         17.274371     113.942164    -15.385900  
# V6                103810.214996  167845.110320 -12085.059882  
# V23                   22.093148      31.868737     -0.799463  
# V34                   31.868737      73.966059     -1.886641  
# V76                   -0.799463      -1.886641      0.262847  

# df.corr()
#                   Fitted_residuals  Fitted_Values        V6       V23  \
# Fitted_residuals          1.000000      -0.000159 -0.000011 -0.000060   
# Fitted_Values            -0.000159       1.000000 -0.115544  0.015785   
# V6                       -0.000011      -0.115544  1.000000  0.107935   
# V23                      -0.000060       0.015785  0.107935  1.000000   
# V34                      -0.000056       0.056902  0.095377  0.788351   
# V76                      -0.000215      -0.128894 -0.115199 -0.331755   

#                        V34       V76  
# Fitted_residuals -0.000056 -0.000215  
# Fitted_Values     0.056902 -0.128894  
# V6                0.095377 -0.115199  
# V23               0.788351 -0.331755  
# V34               1.000000 -0.427880  
# V76              -0.427880  1.000000  
