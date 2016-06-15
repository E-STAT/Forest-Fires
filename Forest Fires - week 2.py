
# coding: utf-8

# ```
# Regression Modeling in Practice Course
# Wesleyan University
# 
# Linear Regression Model
# Mario Colosso V.
# 
# The sample comes from Cortez and Morais study about predicting forest fires using 
# metereological data [Cortez and Morais, 2007]. The study includes data from 517
# forest fires in the Natural Park Montesinho (Trás-os -Montes, in northeastern Portugal)
# January 2000 to December 2003, including meteorological data, the type of vegetation
# involved (which determines the six components of the Canadian Forest Fire Weather Index
# (FWI) system --see below--) and the total burned area in order to generate a model capable
# of predicting the burned area of small fires, which are more frequent.
# 
# Measures
# The data contains:
# * X, Y: location of the fire (x,y axis spatial coordinate within the Montesinho park map:
#   from 1 to 9)
# * month, day: month and day of the week the fire occurred (january to december and monday
#   to sunday)
# * FWI system components:
#   - FFMC: Fine Fuel Moisture Code (numeric rating of the moisture content of litter and
#     other cured fine fuels: 18.7 to 96.2)
#   - DMC: Duff Moisture Code (numeric rating of the average moisture content of loosely
#     compacted organic layers of moderate depth: 1.1 to 291.3)
#   - DC: Drought Code (numeric rating of the average moisture content of deep, compact
#     organic layers: 7.9 to 860.6)
#   - ISI: Initial Spread Index (numeric rating of the expected rate of fire spread: 0.0
#     to 56.1)
# * Metereological variables:
#   - temp: temperature (2.2 to 33.3 °C)
#   - RH: relative humidity (15 to 100%)
#   - wind: wind speed (0.4 to 9.4 Km/h)
#   - rain: outside rain (0.0 to 6.4 mm/m^2)
# * area: the burned area of the forest as response variable (0.0 to 1090.84 Ha).
# 
# ```

get_ipython().magic('matplotlib inline')

import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import ceil

pandas.set_option('display.float_format', lambda x:'%.3f'%x)
plt.style.use('ggplot') # Make the graphs a bit prettier
plt.rcParams['figure.figsize'] = (15, 5)


# Load Forest Fires .csv file

fires = pandas.read_csv('forestfires.csv')


# 1. Lets have a brief look of Fires DataFrame

print(fires.head())   #Show first rows


# Get some descriptive statistic of the data

fires_attributes = fires.columns.values.tolist()
number_of_columns = len(fires_attributes)

statistics = pandas.DataFrame(index=range(0, number_of_columns - 2), 
                              columns=('min', 'max', 'mean', 'median', 'std'))

idx = 0
for attr in [0, 1] + list(range(4, number_of_columns)):
    statistics.loc[idx] = {'min':    min(fires[fires_attributes[attr]]), 
                           'max':    max(fires[fires_attributes[attr]]),
                           'mean':   fires[fires_attributes[attr]].mean(),
                           'median': fires[fires_attributes[attr]].median(),
                           'std':    fires[fires_attributes[attr]].std()}
    idx += 1
statistics.index = [fires_attributes[attr] 
                    for attr in [0, 1] + list(range(4, number_of_columns))]

print(statistics.T)   #Show min, max, mean, median and standard deviation


# Display a graph of quantitative variables vs area

attributes = [0, 1] + list(range(4, number_of_columns - 1))
n_cols = 3
n_rows = int(ceil(len(attributes) / n_cols))
fig = plt.figure()
idx = 1
for attr in attributes:
    plt.subplot(n_rows, n_cols, idx)
    plt.plot(fires['area'], fires[fires_attributes[attr]], 'b.')
    plt.xlabel('area')
    plt.ylabel(fires_attributes[attr])
    idx += 1

plt.show()


# There are some data values where the burned area is away from other values

print(fires[fires['area'] > 250])


# Plot some other variables

fires['temp'].plot()   #Plot temperature graph

fires[['temp', 'RH', 'wind', 'rain']].plot()   #Plot temperature, relative humidity, wind 
                                               #and rain graphs

print(fires.corr())   #Show correlation between variables


# 2. Linear regression

# Convert categorical variables (months and days) into numerical values

months_table = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days_table =   ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']

fires['month'] = [months_table.index(month) for month in fires['month'] ]
fires['day'] =   [days_table.index(day)     for day   in fires['day']   ]

fires['X'] -= 1
fires['Y'] -= 2

print(fires.head())


# ### Center each explanatory variable

for idx in list(range(4, number_of_columns - 1)):   #Exclude categorical variables
    fires[fires_attributes[idx]] = fires[fires_attributes[idx]] - \
                                   fires[fires_attributes[idx]].mean()

statistics = [fires[fires_attributes[idx]].mean() for idx in range(0, number_of_columns)]
statistics = pandas.DataFrame(statistics, 
                              index=fires_attributes,
                              columns=['mean'])

print(statistics.T)   #Only quantitative explanatory variables (FFMC thru rain) were centered


# Generate models to test each variable

statistics = list()
for idx in range(0, number_of_columns - 1):
    model = smf.ols(formula = "area ~ " + 
                    fires_attributes[idx], data = fires).fit()
    
    title = 'Model: area ~ ' + fires_attributes[idx]
    print('+' + "-" * (len(title) + 2) + '+' + '\n' + 
          '| ' + title + ' |' + '\n' + 
          '+' + "-" * (len(title) + 2) + '+')
    print()
    print(model.summary())
    print()
    statistics.append([model.f_pvalue, model.rsquared])


# Models summary:

statistics = pandas.DataFrame(statistics, 
                              index=fires_attributes[: number_of_columns - 1], 
                              columns=['p-value', 'R-squared'])
print(statistics.T)

print(statistics[statistics['p-value'] < 0.05])


# 'temp' is the only statistically significant variable (p-value = 0.026) but it only explains the 1% of
# forest fires. Let's show its linear model summary:

print((smf.ols(formula = "area ~ temp", data = fires).fit()).summary())


# The results of the linear regression models indicated than only temperature (Beta = 1.0726, p = 0.026,
# R-squared = 0.010) was significantly and positively associated with the total burned area due to forest
# fires. 'p-value' of other models are greater than treshold value of 0.05 so results are not statistically
# significant to reject null hypothesis.


# Create a Linear Regression Model for a combination of all variables

explanatory_variables = "X + Y + month + day + FFMC + DMC + DC + ISI + temp + RH + " +  \
                        "wind + rain"
response_variable =     "area"

model = smf.ols(formula = response_variable + " ~ " + explanatory_variables, 
                data = fires).fit()

print(model.summary())


# p-value of combination model (p = 0.410) is bigger than treshold value, so the combination of the Canadian
# Forest Fire Weather Index (FWI) system plus temperature, humidity, wind and rain are not significantly associated
# with the total burned area due to forest fires. _p-value_ of temperature in combination model (p = 0.282) is not
# longer statistically significant, a confounder variable?
# 
# Also, there is a warning in previous model summary: "The condition number is large, 1.76e+03. This might indicate
# that there are strong multicollinearity or other numerical problems."   We will review this issue next week.
