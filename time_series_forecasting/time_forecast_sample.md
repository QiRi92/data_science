# Time Series Forecast Study with Python: Monthly Sales of Bread

Time series forecasting is a process, and the only way to get good forecasts is to practice this process.

In this tutorial, you will discover how to forecast the monthly sales of Bread with Python.

Working through this tutorial will provide you with a framework for the steps and the tools for working through your own time series forecasting problems.

After completing this tutorial, you will know:

- How to confirm your Python environment and carefully define a time series forecasting problem.
- How to create a test harness for evaluating models, develop a baseline forecast, and better understand your problem with the tools of time series analysis.
- How to develop an autoregressive integrated moving average model, save it to file, and later load it to make predictions for new time steps.

## Overview

In this tutorial, we will work through a time series forecasting project from end-to-end, from downloading the dataset and defining the problem to training a final model and making predictions.

This project is not exhaustive, but shows how you can get good results quickly by working through a time series forecasting problem systematically.

The steps of this project that we will through are as follows.

1. [Environment.](#environnment)
2. [Problem Description.](#problem)
3. [Test Harness.](#test)
4. Persistence.
5. Data Analysis.
6. ARIMA Models.
7. Model Validation.

This will provide a template for working through a time series prediction problem that you can use on your own dataset.

### <a id="environment">1. Environment</a>

This tutorial assumes an installed and working SciPy environment and dependencies, including:

- SciPy
- NumPy
- Matplotlib
- Pandas
- scikit-learn
- statsmodels

If you need help installing Python and the SciPy environment on your workstation, consider the <a href="https://www.continuum.io/downloads">Anaconda distribution</a> that manages much of it for you.

This script will help you check your installed versions of these libraries.

```python
# check the versions of key python libraries
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
```

The results on my workstation used to write this tutorial are as follows:

```
scipy: 1.13.1
numpy: 1.26.4
matplotlib: 3.9.2
pandas: 2.2.2
statsmodels: 0.14.2
sklearn: 1.5.1
```

### <a id="problem">2. Problem Description</a>

The problem is to predict the number of monthly sales of bread for the Perrin Freres label (named for a region in France).

The dataset provides the number of monthly sales of bread from January 1964 to September 1972, or just under 10 years of data.

The values are a count of millions of sales and there are 105 observations.

The dataset is credited to Makridakis and Wheelwright, 1989.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly_champagne_sales.csv">Download the dataset</a>.

Download the dataset as a CSV file and place it in your current working directory with the filename “bread.csv“.

### <a id="test">3. Test Harness</a>

We must develop a test harness to investigate the data and evaluate candidate models.

This involves two steps:

1. Defining a Validation Dataset.
2. Developing a Method for Model Evaluation.

#### 3.1 Validation Dataset

The dataset is not current. This means that we cannot easily collect updated data to validate the model.

Therefore we will pretend that it is September 1971 and withhold the last one year of data from analysis and model selection.

This final year of data will be used to validate the final model.

The code below will load the dataset as a Pandas Series and split into two, one for model development (*dataset.csv*) and the other for validation (*validation.csv*).

```python
# separate out a validation dataset
from pandas import read_csv
series = read_csv('bread.csv', header=0, index_col=0, parse_dates=True)
split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)
```

Running the example creates two files and prints the number of observations in each.

```
Dataset 93, Validation 12
```

The specific contents of these files are:

- *dataset.csv*: Observations from January 1964 to September 1971 (93 observations)
- *validation.csv*: Observations from October 1971 to September 1972 (12 observations)

The validation dataset is about 11% of the original dataset.

Note that the saved datasets do not have a header line, therefore we do not need to cater for this when working with these files later.

#### 3.2. Model Evaluation

Model evaluation will only be performed on the data in *dataset.csv* prepared in the previous section.

Model evaluation involves two elements:

1. Performance Measure.
2. Test Strategy.

##### 3.2.1 Performance Measure

The observations are a count of bread sales in millions of units.

We will evaluate the performance of predictions using the root mean squared error (RMSE). This will give more weight to predictions that are grossly wrong and will have the same units as the original data.

Any transforms to the data must be reversed before the RMSE is calculated and reported to make the performance between different methods directly comparable.

We can calculate the RMSE using the helper function from the scikit-learn library mean_squared_error() that calculates the mean squared error between a list of expected values (the test set) and the list of predictions. We can then take the square root of this value to give us an RMSE score.

For example:

```python
from sklearn.metrics import mean_squared_error
from math import sqrt
...
test = ...
predictions = ...
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
```

##### 3.2.2 Test Strategy
