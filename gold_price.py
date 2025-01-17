import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the Data

data = pd.read_csv('gld_price_data.csv')

print(data.head())

data.describe()

#Data Preprocessing

print(data.isnull().sum())

data = data.fillna(method='ffill')

X = data.drop(['Date', 'GLD'], axis=1)
y = data['GLD']

#Train the Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mean = mean_squared_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mean}')
print(f'R-squared: {r2s}')

#Visualize the Results

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices')
plt.legend()
plt.title('Actual vs Predicted Gold Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# Random Forest Regressor Model

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()
reg.fit(X_train,y_train)
pred = reg.predict(X_test)
r2 = r2_score(y_test,pred)
print(f'R-squared: {r2}')
