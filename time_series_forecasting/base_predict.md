# How to Make Baseline Predictions for Time Series Forecasting with Python

Establishing a baseline is essential on any time series forecasting problem.

A baseline in performance gives you an idea of how well all other models will actually perform on your problem.

In this tutorial, you will discover how to develop a persistence forecast that you can use to calculate a baseline level of performance on a time series dataset with Python.

After completing this tutorial, you will know:

- The importance of calculating a baseline of performance on time series forecast problems.
- How to develop a persistence model from scratch in Python.
- How to evaluate the forecast from a persistence model and use it to establish a baseline in performance.

## Forecast Performance Baseline

A baseline in forecast performance provides a point of comparison.

It is a point of reference for all other modeling techniques on your problem. If a model achieves performance at or below the baseline, the technique should be fixed or abandoned.

The technique used to generate a forecast to calculate the baseline performance must be easy to implement and naive of problem-specific details.

Before you can establish a performance baseline on your forecast problem, you must develop a test harness. This is comprised of:

1. The **dataset** you intend to use to train and evaluate models.
2. The **resampling** technique you intend to use to estimate the performance of the technique (e.g. train/test split).
3. The **performance measure** you intend to use to evaluate forecasts (e.g. mean squared error).

Once prepared, you then need to select a naive technique that you can use to make a forecast and calculate the baseline performance.

The goal is to get a baseline performance on your time series forecast problem as quickly as possible so that you can get to work better understanding the dataset and developing more advanced models.

Three properties of a good technique for making a baseline forecast are:

- **Simple**: A method that requires little or no training or intelligence.
- **Fast**: A method that is fast to implement and computationally trivial to make a prediction.
- **Repeatable**: A method that is deterministic, meaning that it produces an expected output given the same input.

A common algorithm used in establishing a baseline performance is the persistence algorithm.

## Persistence Algorithm (the “naive” forecast)

The most common baseline method for supervised machine learning is the Zero Rule algorithm.

This algorithm predicts the majority class in the case of classification, or the average outcome in the case of regression. This could be used for time series, but does not respect the serial correlation structure in time series datasets.

The equivalent technique for use with time series dataset is the persistence algorithm.

The persistence algorithm uses the value at the previous time step (t-1) to predict the expected outcome at the next time step (t+1).

This satisfies the three above conditions for a baseline forecast.

To make this concrete, we will look at how to develop a persistence model and use it to establish a baseline performance for a simple univariate time series problem. First, let’s review the Shampoo Sales dataset.

## Shampoo Sales Dataset

This dataset describes the monthly number of shampoo sales over a 3 year period.

The units are a sales count and there are 36 observations. The original dataset is credited to Makridakis, Wheelwright, and Hyndman (1998).

Below is a sample of the first 5 rows of data, including the header row.

```
"Month","Sales"
"1-01",266.0
"1-02",145.9
"1-03",183.1
"1-04",119.3
"1-05",180.3
```

Below is a plot of the entire dataset where you can download the dataset and learn more about it.

<img width="551" alt="image" src="https://github.com/user-attachments/assets/39f7da44-881c-48f7-91be-0dc958541fec" />

The dataset shows an increasing trend, and possibly some seasonal component.

Download the dataset and place it in the current working directory with the filename “shampoo-sales.csv“.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv">Download the dataset</a>

The following snippet of code will load the Shampoo Sales dataset and plot the time series.

```python
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
series.plot()
pyplot.show()
```

Running the example plots the time series, as follows:

<img width="448" alt="image" src="https://github.com/user-attachments/assets/803f2b4c-de1e-49dc-ada7-e980f3a45f0d" />

## Persistence Algorithm

