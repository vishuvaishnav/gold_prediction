import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the dataset
gold = pd.read_csv('gld_price_data.csv')

# Display the first few rows of the dataset
print(gold.head())

# Display the last few rows of the dataset
print(gold.tail())

# Get information about the dataset
print(gold.info())

# Get descriptive statistics of the dataset
print(gold.describe())

# Drop the 'Date' column as it is not needed for prediction
gold = gold.drop(columns=['Date'])

# Calculate the correlation matrix
correlation = gold.corr()

# Plot the correlation heatmap
plt.figure(figsize=(6,6))
sns.heatmap(correlation, cbar=True, square=True, annot=True, annot_kws={"size":8})
plt.show()

# Print the correlation of each feature with the gold price
print(correlation['GLD'])

# Plot the distribution of the gold price
sns.displot(gold['GLD'], color='orange')
plt.show()

# Splitting the data into features and target variable
X = gold.drop(['GLD'], axis=1)
y = gold['GLD']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
reg = RandomForestRegressor()
reg.fit(X_train, y_train)

# Make predictions on the test set
pred = reg.predict(X_test)

# Calculate the R^2 score
score = r2_score(y_test, pred)
print(f"R^2 Score: {score}")
