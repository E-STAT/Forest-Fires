
# coding: utf-8

# Regression Modeling in Practice Course
# Wesleyan University
# 
# Linear Regression Model
# Mario Colosso V.
# 
# The sample comes from Cortez and Morais study about predicting forest fires using 
# metereological data [Cortez and Morais, 2007]. The study includes data from 517
# forest fires in the Natural Park Montesinho (Trás-os-Montes, in northeastern Portugal)
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


# Import required libraries and set global options

# In[1]:

get_ipython().magic('matplotlib inline')

import pandas
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.tools.plotting import scatter_matrix
from math import ceil

pandas.set_option('display.float_format', lambda x:'%.3f'%x)
#pandas.set_option('display.mpl_style', 'default')   # --deprecated
plt.style.use('ggplot')   # Make the graphs a bit prettier
plt.rcParams['figure.figsize'] = (15, 5)


# Load Forest Fires .csv file
fires = pandas.read_csv('forestfires.csv')


# 1. DATA EXPLORATION

fires.head()   #Show first rows

# Get some descriptive statistic of the data

fires_attributes = fires.columns.values.tolist()
number_of_columns = len(fires_attributes)

fires.describe()   #Original data

# Display a graph of quantitative variables vs area

attributes = [0, 1] + list(range(4, number_of_columns - 1))
n_cols = 3
n_rows = int(ceil(len(attributes) / n_cols))
fig = plt.figure()
idx = 1
for attr in attributes:
    plt.subplot(n_rows, n_cols, idx)
    plt.plot(fires['area'], fires[fires_attributes[attr]], 'b.')
#    seaborn.regplot(x = fires['area'], y = fires[fires_attributes[attr]], 
#                    scatter = True, color = 'b', data = fires)
    plt.xlabel('area')
    plt.ylabel(fires_attributes[attr])
    idx += 1

plt.show()

# There are some data values where the burned area is away from other values:

print(fires[fires['area'] > 250])

# Plot some other variables

scatter_matrix(fires, figsize = (15,10))
plt.show()

# High bias are appreciated in FFMC, DC, ISI, wind and area variables


fires[['temp', 'RH', 'wind', 'rain']].plot()   #Plot temperature, relative humidity, wind 
                                               #and rain graphs

print(fires.corr())   #Show correlation between variables

def plot_corr(df, size=10):
    '''Function plots a graphical correlation matrix for each pair of columns
       in the dataframe, including the names of the attributes
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    
    Code taken from:
    http://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
    '''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap = 'YlGnBu')
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

#plt.matshow(fires.corr())
plot_corr(fires, size = 6)


# There is a medium-high correlation (0.682) between DC (Drought Code: numeric rating of the average
# moisture content of deep, compact organic layers) and DMC (Duff Moisture Code: numeric rating of the
# average moisture content of loosely compacted organic layers of moderate depth) and medium correlation
# (0.532) between ISI (Initial Spread Index: numeric rating of the expected rate of fire spread) and
# FFMC (Fine Fuel Moisture Code: numeric rating of the moisture content of litter and other cured fine
# fuels). Also, there is a inverse medium  correlation (-0.527) between temperature (temp) and relative
# humidity (RH). Other relationships are noted between temperature (temp) and FWI system components
# (FFMC, DCM, DC and ISI)


# 2. LINEAR REGRESSION

# Convert categorical variables (months and days) into numerical values

months_table = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days_table =   ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']

fires['month'] = [months_table.index(month) for month in fires['month'] ]
fires['day'] =   [days_table.index(day)     for day   in fires['day']   ]

fires['X'] -= 1
fires['Y'] -= 2

print(fires.head())

# Center each explanatory variable

for idx in list(range(4, number_of_columns - 1)):   #Exclude categorical variables
    fires[fires_attributes[idx]] = fires[fires_attributes[idx]] - \
                                   fires[fires_attributes[idx]].mean()


fires.describe()   #Only quantitative explanatory variables (FFMC thru rain) were centered

# Generate models to test each variable

def print_title(title):
    print('+' + "-" * (len(title) + 2) + '+' + '\n' + 
          '| ' + title + ' |' + '\n' + 
          '+' + "-" * (len(title) + 2) + '+')

statistics = list()
for idx in range(0, number_of_columns - 1):
    model = smf.ols(formula = "area ~ " + 
                    fires_attributes[idx], data = fires).fit()
    
    print_title('Model: area ~ ' + fires_attributes[idx])
    print()
    print(model.summary())
    print()
    statistics.append([model.f_pvalue, model.rsquared])

# Summary:

statistics = pandas.DataFrame(statistics, 
                              index=fires_attributes[: number_of_columns - 1], 
                              columns=['p-value', 'R-squared'])
print(statistics.T)

statistics[statistics['p-value'] < 0.05]

# Temperature ('temp') is the only statistically significant variable (p-value = 0.026) but it
# only explains the 1% of forest fires. Let's show its linear model summary:

print((smf.ols(formula = "area ~ temp", data = fires).fit()).summary())

# The results of the linear regression models indicated than only temperature (Beta = 1.0726,
# p = 0.026, $R^2$ = 0.010) was significantly and positively associated with the total burned
# area due to forest fires. _'p-value'_ of other models are greater than treshold value of 0.05
# so results are not statistically significant to reject null hypothesis.

# Create a Linear Regression Model for a combination of all variables

explanatory_variables = "X + Y + month + day + FFMC + DMC + DC + ISI + temp + RH + " +                         "wind + rain"
response_variable =     "area"

model = smf.ols(formula = response_variable + " ~ " + explanatory_variables, 
                data = fires).fit()

print(model.summary())

# p-value of combination model (p = 0.410) is bigger than treshold value, so the combination of
# the Canadian Forest Fire Weather Index (FWI) system plus temperature, humidity, wind and rain
# are not significantly associated with the total burned area due to forest fires. p-value of
# temperature in combination model (p = 0.282) is not longer statistically significant, a
# confounder variable?


# 3. MULTIPLE REGRESSON MODEL

# Sort explanatory variables by p-value

statistics = statistics.sort_values(by='p-value')

# Define an useful function to plot QQ and Residual plots

def print_qqplot_and_residuals_plot(model):
    # qq-plot
    ax1 = plt.subplot(1, 3, 1)
    qq_plot = sm.qqplot(model.resid, line = 'r', ax = ax1)
    
    # Residuals plot
    ax2 = plt.subplot(1, 3, 2)
    stdres = pandas.DataFrame(model.resid_pearson)
    residuals_plot = plt.plot(stdres, 'o', ls = 'None')
    plt.axhline(y = 0, color = 'r')
    plt.ylabel('Standarized Residual')
    plt.xlabel('Observation Number')
    
    plt.show()

# Generate linear models adding one explanatory variable a time

explanatory_variables = None
response_variable =     "area"

saved_models = list()

for variable in list(statistics[: number_of_columns - 1].index.values):
    if explanatory_variables == None:
        explanatory_variables = variable
    else:
        explanatory_variables += " + " + variable
    model = smf.ols(formula = response_variable + " ~ " + explanatory_variables, 
                    data = fires).fit()
    saved_models.append(model)
    
    print_title('Model: ' + response_variable + " ~ " + explanatory_variables)
    print()
    print(model.summary())
    print_qqplot_and_residuals_plot(model)
    print()


# From above results can be seen that after adding a second variable to the model (relative humidity --RH--),
# temperature --temp-- is not longer statistically significant, so its a confounder variable: "temperature
# increases due to forest fire, or forest fire is helped by high temperatures?"

# In all generated models we can see the summary of the multiple regression model, the quantile-quantile plot
# (qq-plot), left side, which "plots the quantiles of the residuals that we would theoretically see if the
# residuals followed a normal distribution, against the quantiles for the residuals estimated from our
# regression model", and, right side, the Standarized Residuals plot, which are, "simply, the residual values
# transformed to have a mean of zero and a standard deviation of one"
# 
# The qq-plots of our regression models shows a straight line with high deviations at the lower and higher
# quantiles, indicating the residuals do not follow a normal distribution.
# 
# The Standarized Residuals plot shows there are a group of observations which standarized residuals are
# greather than 2 standard deviations, and more than 3 standard deviations implying the presence of extreme
# outliers.

stdres = pandas.DataFrame(saved_models[11].resid_pearson)
stdres[(stdres[0] < -2.5) | (stdres[0] > 2.5)].T


# In all generated models, more than 1% of our observations has standardized residuals with an absolute value
# greater than 2.5, then there is evidence that the level of error within our model is unacceptable. That is,
# the model is a fairly poor fit to the observed data.

pct_outliers = list()
for idx in range(len(saved_models)):
    stdres = pandas.DataFrame(saved_models[idx].resid_pearson)
    pct_outliers.append(round(len(stdres[(stdres[0] < -2.5) | (stdres[0] > 2.5)]) / len(fires), 3))

print(pct_outliers)


# The second warning of full model indicates "The condition number is large, 1.76e+03. This might indicate that
# there are strong multicollinearity or other numerical problems", as it could be expected.
# 
# During data exploration we found that there is a medium to medium-high correlation between the average moisture
# content of deep, compact organic layers and the average moisture content of loosely compacted organic layers of
# moderate depth (DC-DMC: 0.682) and between the expected rate of fire spread and the moisture content of litter
# and other cured fine fuels (ISI- FFMC: 0.532). Also, there is a inverse medium correlation (-0.527) between
# temperature (temp) and relative humidity (RH). Other relationships are noted between temperature (temp) and
# FWI system components (FFMC, DCM, DC and ISI)


# Leverage plot

sm.graphics.influence_plot(model, size=8) #Leverage plot for full model
plt.show()


# Leverage plot permits identify observations that have an unusually large influence on the estimation of the
# predicted value of the response variable, burned area, or that are outliers, or both. The graph of full model
# shows one observation with a very high influence (observation 499, with near 90%), one with medium influence
# (observation 379, with near 45%) and one with medium-low influence (observation 22, with near 32%). The rest
# of the observations have influence under 20%
# 
# The graph of full model also show us a group of ouliers. Note this extreme outliers are the same observations
# we found during data exploration: 238, 415, 479, plus a cloud of minor outliers (residuals outside range -2 to
# 2 standard deviations), but with low influence (< 5%) on the estimation of the regression model.
# 
# No observations in this data are both high leverage and outliers.

print(fires.iloc[[22, 379, 499]])
