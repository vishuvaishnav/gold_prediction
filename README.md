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

## Results

After training the model, we evaluate its performance using the R² score. The higher the R² score, the better the model's performance.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
