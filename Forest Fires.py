
# coding: utf-8

# Regression Modeling in Practice Course
# by Wesleyan University
# 
# Linear Regression Model
# Mario Colosso V.

get_ipython().magic('matplotlib inline')

import pandas
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

pandas.set_option('display.mpl_style', 'default')   # Make the graphs a bit prettier
pandas.set_option('display.float_format', lambda x:'%.3f'%x)   # Show 3 decimals
plt.rcParams['figure.figsize'] = (15, 5)


# Load Forest Fires .csv file
fires = pandas.read_csv('forestfires.csv')


# 1. Lets have a brief look of Fires DataFrame

fires.head()   #Show first rows

# Get some descriptive statistic of the data
fires_attributes = fires.columns.values.tolist()
number_of_columns = len(fires_attributes)

statistics = pandas.DataFrame(index=range(0, number_of_columns - 4), columns=('name', 'min', 'max', 'mean'))

for attr in range(4, number_of_columns):
    idx = attr - 4
    statistics.loc[idx] = {'name': fires_attributes[attr], 
                           'min':  min(fires[fires_attributes[attr]]), 
                           'max':  max(fires[fires_attributes[attr]]),
                           'mean': fires[fires_attributes[attr]].mean()}

print(statistics)   #Show min, max and mean of variables

fires['temp'].plot()   #Plot temperature graph

fires[['temp', 'RH', 'wind', 'rain']].plot()   #Plot temperature, relative humidity, wind and rain graphs

fires.corr()   #Show correlation between variables


# ## 2. Linear regression

# Convert categorical variables (months and days) into numerical values
months_table = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days_table =   ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']

fires['month'] = [months_table.index(month) for month in fires['month'] ]
fires['day'] =   [days_table.index(day)     for day   in fires['day']   ]

fires.head()

# Center each explanatory variable
for idx in range(0, number_of_columns - 1):
    fires[fires_attributes[idx]] = fires[fires_attributes[idx]] - fires[fires_attributes[idx]].mean()

# Show descriptive statistics of all variables
for idx in range(0, number_of_columns):
    statistics.loc[idx] = {'name': fires_attributes[idx], 
                           'min':  min(fires[fires_attributes[idx]]), 
                           'max':  max(fires[fires_attributes[idx]]),
                           'mean': fires[fires_attributes[idx]].mean()}

print(statistics)   #Only explanatory variables were centered

# Generate models to test each variable
for idx in range(4, number_of_columns - 1):
    model = smf.ols(formula = "area ~ " + fires_attributes[idx], data = fires).fit()
    print(model.summary())
    print()

# The results of the linear regression models indicated than only temperature (Beta = 1.0726, p = 0.026)
# was significantly and positively associated with the total burned area due to forest fires. P-value of
# other models are greater than treshold value of 0.05 so results are not statistically significant
# to reject null hypothesis.

# Create a Linear Regression Model for a combination of variables
explanatory_variables = "FFMC + DMC + DC + ISI + temp + RH + wind + rain"
response_variable =     "area"

model = smf.ols(formula = response_variable + " ~ " + explanatory_variables, data = fires).fit()

print(model.summary())

# P-value of combination model (p = 0.410) is bigger than treshold value, so the combination of the Canadian
# Forest Fire Weather Index (FWI) system plus temperature, humidity, wind and rain are not significantly
# associated with the total burned area due to forest fires. _p-value_ of temperature in combination model
# (p = 0.282) is not longer statistically significant, a confounder variable?
