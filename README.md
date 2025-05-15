# House Price Prediction Using Linear Regression (From Scratch)

This repository contains a simple implementation of **Linear Regression** from scratch in Python, applied to the problem of predicting house prices based on various features.

---

## Project Overview

The goal of this project is to build a Linear Regression model without using any machine learning libraries (like scikit-learn) and apply it on a real-world dataset for house price prediction. The model is trained using **gradient descent** and evaluated using common regression metrics.

---

## Features

- Custom implementation of Linear Regression
- Feature scaling (standardization) applied to input features
- Training using gradient descent optimization
- Visualization of loss reduction over training epochs
- Visualization of predictions vs actual house prices
- Evaluation using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared score

---

## Dataset

The dataset used (`house_data.csv`) contains various house-related features such as number of bedrooms, bathrooms, square footage, year built, etc., along with the house prices.

**Note:** The dataset should be placed in the root directory or update the path accordingly in the notebook/script.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Venkatesh2007/house-price-prediction.git
   cd house-price-prediction

2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. Run the script or notebook:

   ```bash
   python house_price_prediction.py
   ```

   or open the Jupyter notebook and run all cells.

---

## Code Structure

* `LinearRegression` class: Custom implementation of Linear Regression with methods:

  * `fit(X, y)`: Train the model using gradient descent
  * `predict(X)`: Predict house prices for given features
  * `plot_loss()`: Plot training loss over epochs
  * `plot_predictions(X, y, preds)`: Plot predicted vs actual prices

* `standardize(X)`: Function to standardize features for better convergence

* Data loading, preprocessing, training, prediction, and evaluation steps included

---

## Results

Example output metrics from the model:

* Mean Absolute Error (MAE): *e.g., 210759.14640801313*
* Mean Squared Error (MSE): *e.g., 986608166705.6411*
* Root Mean Squared Error (RMSE): *e.g., 993281.5143279579*
* R-squared Score: *e.g., 0.03259135428037074*

Loss and prediction plots are generated to visualize training performance and accuracy.

---

## Future Improvements

* Add polynomial or interaction features to capture non-linear relationships
* Implement regularization (Ridge, Lasso) to prevent overfitting
* Explore other models like Decision Trees, Random Forests, or Gradient Boosting
* Perform hyperparameter tuning (learning rate, number of iterations)
* Better feature engineering and selection

---

## License

This project is licensed under the MIT License.

---

## Author

Badam Venkatesh
Email: [badamvenkatesh2007@gmail.com](mailto:badamvenkatesh2007@gmail.com)
GitHub: [Venkatesh2007](https://github.com/Venkatesh2007)

---
