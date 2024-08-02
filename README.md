# Gold Price Prediction

This repository contains a project for predicting gold prices using a Random Forest Regressor. The dataset used is `gld_price_data.csv`, and the analysis includes data preprocessing, exploratory data analysis, and model training.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Predicting gold prices can be a valuable tool for investors and analysts. This project aims to build a model that predicts gold prices based on historical data. We use the Random Forest Regressor from the scikit-learn library to train the model.

## Dataset

The dataset used in this project is `gld_price_data.csv`, which contains the following columns:

- `Date`: The date of the recorded gold price.
- `SPX`: S&P500 index.
- `USO`: United States Oil Fund price.
- `SLV`: iShares Silver Trust price.
- `EUR/USD`: Euro to US Dollar exchange rate.
- `GLD`: Gold price.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/yourusername/gold-price-prediction.git
cd gold-price-prediction
```

2. Make sure the dataset `gld_price_data.csv` is in the project directory.

3. Run the analysis script:

```sh
python gold_price_prediction.py
```

### Code Explanation

```python
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
```

## Results

After training the model, we evaluate its performance using the R² score. The higher the R² score, the better the model's performance.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
