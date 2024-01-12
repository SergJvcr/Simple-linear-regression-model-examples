# We can use simple linear regression to explore
# the relationship between two continuous variables;
# The objective is to predict the value of an output variable 
# (or response) based on the value of an input (or predictor) variable

# Steps: 
# -creating and fitting a model
# -checking model assumptions
# -analyzing model performance
# -interpreting model coefficients
# -communicating results to stakeholders

# Working with data and plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creating and fitting the model
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the dataset
data = pd.read_csv('google_data_analitics\\marketing_and_sales_data_evaluate_lr.csv')
print(data.head(5))

# Data exploration
# Explore the data size
print(f'There are {data.shape[0]} rows and {data.shape[1]} columns.')

# Explore the independent variables
print(data[['TV', 'Radio', 'Social_Media']].describe()) # generate descriptive statistics about TV, Radio, and Social_Media

# Explore the dependent variable
# Before fitting the model, ensure the Sales for each promotion
# (i.e., row) is present. If the Sales in a row is missing, 
# that row isn't of much value to the simple linear regression model;
# Display the percentage of missing values in the Sales column in the DataFrame data
missing_sales = data['Sales'].isna().mean()
missing_sales = round(missing_sales*100, 2) # convert the missing_sales from a decimal to a percentage and round it

print(f'The percentage of missing values in the Sales column in the DataFrame data: {missing_sales}%')
# The preceding output shows that 0.13% of rows are missing the Sales value

# Visualize the sales distribution
fig = sns.histplot(data['Sales'], color='orange').set_title('The Distribution of Sales')
plt.show()
# Generally, Sales are equally distributed between 25 and 350 million

# Model building
sns.pairplot(data)
plt.show()
# TV clearly has the strongest linear relationship with Sales. 
# You could draw a straight line through the scatterplot of TV and Sales that 
# confidently estimates Sales using TV. Radio and Sales appear to have a linear relationship, 
# but there is larger variance than between TV and Sales.

# Build and fit the model

# Define the subset from our dataset
ols_data = data[['TV', 'Sales']]
# Define the OLS formula
ols_formula = 'Sales ~ TV'
# Create an OLS model
OLS = ols(formula = ols_formula, data = ols_data)
# Fit the model
model = OLS.fit()
# Save the results summary
model_results = model.summary()
# Display the model results
print(model_results)

# Check model assumptions
#To justify using simple linear regression, 
# check that the four linear regression assumptions are not violated. 
# These assumptions are:
# -Linearity
# -Independent Observations
# -Normality
# -Homoscedasticity

# Model assumption: Linearity
sns.scatterplot(x = data['TV'], y = data['Sales']).set_title('The scatterplot TV vs Sales')
plt.show()
# There is a clear linear relationship between TV and Sales, meeting the linearity assumption

# Model assumption: Independence
# The independent observation assumption states that each observation in the dataset is independent. 
# As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# Model assumption: Normality
# The normality assumption states that the errors are normally distributed
# To check this assumption we can build two plots: Histogram of the residuals and Q-Q plot of the residuals
residuals = model.resid
# Create a 1x2 plot figures
fig, axes = plt.subplots(1,2, figsize=(10, 5))
# Create a histogram with the residuals
sns.histplot(residuals, ax=axes[0], color='green')
# Set the x label of the residual plot
axes[0].set_xlabel('Residual Value')
# Set the title of the residual plot
axes[0].set_title('The residual plot')
# Create a Q-Q plot of the residuals
sm.qqplot(residuals, line='s', ax=axes[1], color='grey')
# Set the title of the Q-Q plot
axes[1].set_title('The Q-Q plot  of residuals')
# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance
plt.tight_layout()
plt.show()
# The histogram of the residuals are approximately normally distributed, 
# which supports that the normality assumption is met for this model.
# The residuals in the Q-Q plot form a straight line, further supporting that the normality assumption is met

# Model assumption: Homoscedasticity
# The homoscedasticity (constant variance) assumption is 
# that the residuals have a constant variance for all values of X
fig = sns.scatterplot(x=model.fittedvalues, y=residuals, color='lightblue')
# Set the x-axis label
fig.set_xlabel('The fitted values from the model')
# Set the y-axis label
fig.set_ylabel('Residuals')
# Set the title
fig.set_title('Fitted values vs Residuals')
# Add a line at y = 0 to visualize the variance of residuals above and below 0.
fig.axhline(0)
plt.show()
# The variance of the residuals is consistant across all  𝑋. 
# Thus, the assumption of homoscedasticity is met.

# Results and evaluation
# Using TV as X results in a simple linear regression model with  𝑅2=0.999. 
# In other words, TV explains  99.9% of the variation in Sales.
# The R-squared value will depend on the variable selected for X.

# When TV is used as the independent variable X, 
# the coefficient for the Intercept is -0.1263 and the coefficient for TV is 3.5614.

# When TV is used as the independent variable X, the linear equation is:
# 𝑌=Intercept+Slope∗𝑋
# Sales (in millions)=Intercept+Slope∗TV (in millions)
# Sales (in millions)=−0.1263+3.5614∗TV (in millions)

# According to the model, when TV is used as the independent variable X, 
# an increase of one million dollars for the TV promotional budget results 
# in an estimated 3.5614 million dollars more in sales.

# Measure the uncertainty of the coefficient estimates
# When TV is used as the independent variable, it has a p-value of  0.000 
# and a  95% confidence interval of  [3.558,3.565]. This means there is 
# a 95% chance the interval  [3.558,3.565] contains the true parameter value of the slope. 
# These results indicate little uncertainty in the estimation of the slope of X. 
# Therefore, the business can be confident in the impact TV has on Sales


# What findings would you share with others?
# Sales is relatively equally distributed between $25 and $350 million across all promotions.
# In the data visualization, TV has the strongest linear relationship with sales. 
# Radio and sales appear to have a moderate linear relationship, but there is larger variance 
# than between TV and sales. Social media and sales appear to have a weak linear relationship.
# When TV is used as the independent variable, the R-squared for the simple linear regression model 
# estimating sales is 0.999. When TV is used as the independent variable, the coefficient 
# for the intercept is -0.1263 and the coefficient for TV is 3.5614.
# When TV is used as the independent variable, the slope of the coefficient for TV has 
# a p-value of 0.000 and a 95% confidence interval of [3.558, 3.565].

# How would you frame your findings to stakeholders?
# Among TV, social media, and radio, TV had the strongest positive linear relationship with sales.
# The linear regression model estimates that 99.9% of the variation in sales is explained 
# by the TV promotional budget. In other words, nearly all of the variation in sales can be 
# explained by the TV promotional budget alone, making TV an excellent predictor of sales.
# According to the model, when TV is used as the independent variable X, 
# an increase of one million dollars for the TV promotional budget would result in 
# an estimated 3.5614 million more dollars in sales.
# The interval (3.558 million, 3.565 million) has a 95% probability of containing the true estimate
# of the increase in sales for a one million dollar increase in the TV promotional budget. 
# Therefore, the estimate provided in the previous bullet is very confident.