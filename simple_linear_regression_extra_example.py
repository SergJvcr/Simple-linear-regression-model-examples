# Import relevant Python libraries and modules.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

# Import a data
data = pd.read_csv("google_data_analitics\\marketing_sales_data.csv")

# Data exploration
print(data.head(10)) # display the first 10 rows of the data
print(f'The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.') # display number of rows, number of columns
print(data.describe(include='all'))
print(data.info())

# Cleaning the data from null-rows
missing_values = data.isna().sum() # to get the number of rows that contain missing values
print(missing_values)
data = data.dropna(axis=0) # to delete/ to drop the rows with missing values
missing_values_update = data.isna().sum() #checking how its were delete
print(missing_values_update)
print(f'After cleaning the dataset contains {data.shape[0]} rows and {data.shape[1]} columns.')

# Create a plot of pairwise relationships in the data
# This will help us visualize the relationships and check model assumptions
sns.pairplot(data)
plt.show()
# In the scatter plot of Sales over Radio, the points appear to cluster 
# around a line that indicates a positive association between the two variables. 
# Since the points cluster around a line, it seems the assumption of linearity is met

# Model building
# Select relevant columns and save them in a separate variable to prepare for regression
ols_data = data[['Radio', 'Sales']] # a subset from the main dataset
# Display first 10 rows of the new DataFrame to check it
print(ols_data.head(10))
# Create a variable with the linear regression formula
ols_formula = 'Sales ~ Radio'
# Implement the ordinary least squares (OLS) approach for linear regression
OLS = ols(formula=ols_formula, data=ols_data)
# Create a linear regression model for the data and fit the model to the data
model = OLS.fit()

# Results and evaluation
model_results = model.summary() # Get summary of results
print(model_results)
# The y-intercept is 41.5326
# The slope is 8.1733
# The linear equation: sales = 41.5326 + 8.1733 * radio promotion budget
# The interpretation: If a company has a budget of 1 million dollars more 
# for promoting their products/services on the radio, the company's sales 
# would increase by 8.1733 million dollars on average.

# Checking the model assumptions (this will us help confirm our findings):
# 1. The linearity assumption
# Plot the OLS data with the best fit regression line
sns.regplot(x='Radio', y='Sales', data=ols_data, color='gray', marker='x', line_kws=dict(color='r'))
plt.show()
# The preceding regression plot illustrates an approximately linear relationship 
# between the two variables along with the best fit line. This confirms the assumption of linearity
# 2. The normality assumption
residuals = model.resid # get the residuals from the model
# Visualize the distribution of the residuals (here are two ways - two graphs)
# Create a Q-Q plot.
fig, axes = plt.subplots(1,2, figsize=(10, 5))
# Create a histogram with the residuals
sns.histplot(residuals, ax=axes[0], color='green')
# Set the x label of the residual plot
axes[0].set_xlabel('Residual Value')
# Set the title of the residual plot
axes[0].set_title('Histogram of Residuals')
# Create a Q-Q plot of the residuals
sm.qqplot(residuals, line='s', ax=axes[1], color='lightgreen')
# Set the title of the Q-Q plot
axes[1].set_title('The Q-Q plot of residuals')
# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance
plt.tight_layout()
plt.show()
# In the preceding Q-Q plot, the points closely follow a straight
# diagonal line trending upward. This confirms that the normality assumption is met
# 3, 4. The independent observation assumption and the homoscedasticity assumption
fitted_values = model.predict(ols_data["Radio"]) # get fitted values
# Create a scatterplot of residuals against fitted values.
fig = sns.scatterplot(x=fitted_values, y=residuals, color='orange') # or we can do this: x=model.fittedvalues
# Set the x-axis label
fig.set_xlabel('Fitted values from the model')
# Set the y-axis label
fig.set_ylabel('Residuals')
# Set the title
fig.set_title('Fitted values vs Residuals')
# Add a line at y = 0 to visualize the variance of residuals above and below 0
fig.axhline(0)
plt.show()
# In the preceding scatterplot, the data points have a cloud-like resemblance and 
# do not follow an explicit pattern. So it appears that the independent observation 
# assumption has not been violated. Given that the residuals appear to be randomly spaced, 
# the homoscedasticity assumption seems to be met.

# Conclusion
# In the simple linear regression model, the y-intercept is 41.5326 and the slope is 8.1733. 
# One interpretation: If a company has a budget of 1 million dollars more for promoting 
# their products/services on the radio, the company's sales would increase 
# by 8.1733 million dollars on average. 
# OR WE CAN SAY: Companies with 1 million dollars more in their radio promotion budget 
# accrue 8.1733 million dollars more in sales on average.

# The results are statistically significant with a p-value of 0.000, which is a very small value 
# (and smaller than the common significance level of 0.05). This indicates that there is a very low 
# probability of observing data as extreme or more extreme than this dataset when the null hypothesis is true. 
# In this context, the null hypothesis (H0) is that there is no relationship between 
# radio promotion budget and sales i.e. the slope is zero, and the alternative hypothesis (Ha) is 
# that there is a relationship between radio promotion budget and sales i.e. the slope is not zero. 
# So, you could reject the null hypothesis and state that there is a relationship between 
# radio promotion budget and sales for companies in this data.

# The slope of the line of best fit that resulted from the regression model is approximate and subject 
# to uncertainty (not the exact value). The 95% confidence interval for the slope is from 7.791 to 8.555. 
# This indicates that there is a 95% probability that the interval [7.791, 8.555] contains the true value for the slope.

# Based on the dataset at hand and the regression analysis conducted here, there is a notable relationship 
# between radio promotion budget and sales for companies in this data, with a p-value of 0.000 and 
# standard error of 0.194. For companies represented by this data, 
# a 1 million dollar increase in radio promotion budget could be associated with a 8.1733 million dollar increase in sales. 
# It would be worth continuing to promote products/services on the radio. 
# Also, it is recommended to consider further examining the relationship between the two variables 
# (radio promotion budget and sales) in different contexts. For example, it would help to gather more data 
# to understand whether this relationship is different in certain industries or when promoting certain types of products/services.





