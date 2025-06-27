## INTRODUCTION

In this mini-course, you will discover how you can get started, build accurate models and confidently complete predictive modeling time series forecasting projects using Python.

Below are 7 lessons that will get you started and productive with machine learning in Python:

- [Lesson 01: Time Series as Supervised Learning.](#lesson01)
- [Lesson 02: Load Time Series Data.](#lesson02)
- [Lesson 03: Data Visualization.](#lesson03)
- [Lesson 04: Persistence Forecast Model.](#lesson04)
- [Lesson 05: Autoregressive Forecast Model.](#lesson05)
- [Lesson 06: ARIMA Forecast Model.](#lesson06)
- [Lesson 07: Hello World End-to-End Project.](#lesson07)

## <a id="lesson01">Lesson 01: Time Series as Supervised Learning</a>

Time series problems are different to traditional prediction problems.

The addition of time adds an order to observations that both must be preserved and can provide additional information for learning algorithms.

A time series dataset may look like the following:

Time, Observation
day1, obs1
day2, obs2
day3, obs3

We can reframe this data as a supervised learning problem with inputs and outputs to be predicted. For example:

Input,	Output
?,		obs1
obs1,	obs2
obs2,	obs3
obs3,	?

You can see that the reframing means we have to discard some rows with missing data.

Once it is reframed, we can then apply all of our favorite learning algorithms like k-Nearest Neighbors and Random Forest.

For more help, see the post:

- <a href="https://github.com/QiRi92/data_science/blob/main/time_series_forecasting/supervised_learning.md">Time Series Forecasting as Supervised Learning</a>

## <a id="lesson02">Lesson 02: Load Time Series Data</a>

Before you can develop forecast models, you must load and work with your time series data.

Pandas provides tools to load data in CSV format.

In this lesson, you will download a standard time series dataset, load it in Pandas and explore it.

Download the daily female births dataset in CSV format and save it with the filename “daily-births.csv“.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv">Download the dataset</a>

You can load a time series dataset as a Pandas Series and specify the header row at line zero, as follows:

```python
from pandas import read_csv
series = read_csv('daily-births.csv', header=0, index_col=0)
```

Get used to exploring loaded time series data in Python:

- Print the first few rows using the head() function.
- Print the dimensions of the dataset using the size attribute.
- Query the dataset using a date-time string.
- Print summary statistics of the observations.

For more help, see the post:

- <a href="">How to Load and Explore Time Series Data in Python</a>

## <a id="lesson03">Lesson 03: Data Visualization</a>

Data visualization is a big part of time series forecasting.

Line plots of observations over time are popular, but there is a suite of other plots that you can use to learn more about your problem.

In this lesson, you must download a standard time series dataset and create 6 different types of plots.

Download the monthly shampoo sales dataset in CSV format and save it with the filename “shampoo-sales.csv“.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv">Download the dataset</a>

Now create the following 6 types of plots:

1. Line Plots.
2. Histograms and Density Plots.
3. Box and Whisker Plots by year or quarter.
4. Heat Maps.
5. Lag Plots or Scatter Plots.
6. Autocorrelation Plots.

Below is an example of a simple line plot to get you started:

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('shampoo-sales.csv', header=0, index_col=0)
series.plot()
pyplot.show()
```

For more help, see the post:

- <a href="">Time Series Data Visualization with Python</a>

## <a id="lesson04">Lesson 04: Persistence Forecast Model</a>

It is important to establish a baseline forecast.

The simplest forecast you can make is to use the current observation (t) to predict the observation at the next time step (t+1).

This is called the naive forecast or the persistence forecast and may be the best possible model on some time series forecast problems.

In this lesson, you will make a persistence forecast for a standard time series forecast problem.

It is important to establish a baseline forecast.

The simplest forecast you can make is to use the current observation (t) to predict the observation at the next time step (t+1).

This is called the naive forecast or the persistence forecast and may be the best possible model on some time series forecast problems.

In this lesson, you will make a persistence forecast for a standard time series forecast problem.

```python
# persistence model
def model_persistence(x):
	return x
```

Write code to load the dataset and use the persistence forecast to make a prediction for each time step in the dataset. Note, that you will not be able to make a forecast for the first time step in the dataset as there is no previous observation to use.

Store all of the predictions in a list. You can calculate a Root Mean Squared Error (RMSE) for the predictions compared to the actual observations as follows:

```python
from sklearn.metrics import mean_squared_error
from math import sqrt
predictions = []
actual = series.values[1:]
rmse = sqrt(mean_squared_error(actual, predictions))
```

For more help, see the post:

- <a href="">How to Make Baseline Predictions for Time Series Forecasting with Python</a>

## <a id="lesson05">Lesson 05: Autoregressive Forecast Model</a>

Autoregression means developing a linear model that uses observations at previous time steps to predict observations at future time step (“auto” means self in ancient Greek).

Autoregression is a quick and powerful time series forecasting method.

The statsmodels Python library provides the autoregression model in the <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html">AutoReg&nbsp;class</a>

In this lesson, you will develop an autoregressive forecast model for a standard time series dataset.

You can fit an AR model as follows:

```python
model = AutoReg(dataset, lags=2)
model_fit = model.fit()
```

You can predict the next out of sample observation with a fit AR model as follows:

```python
prediction = model_fit.predict(start=len(dataset), end=len(dataset))
```

You may want to experiment by fitting the model on half of the dataset and predicting one or more of the second half of the series, then compare the predictions to the actual observations.

For more help, see the post:

- <a href="">Autoregression Models for Time Series Forecasting With Python</a>

## <a id="lesson06">Lesson 06: ARIMA Forecast Model</a>

The ARIMA is a classical linear model for time series forecasting.

It combines the autoregressive model (AR), differencing to remove trends and seasonality, called integrated (I) and the moving average model (MA) which is an old name given to a model that forecasts the error, used to correct predictions.

The statsmodels Python library provides the <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html">ARIMA class</a>

In this lesson, you will develop an ARIMA model for a standard time series dataset.

The ARIMA class requires an order(p,d,q) that is comprised of three arguments p, d and q for the AR lags, number of differences and MA lags.

You can fit an ARIMA model as follows:

```python
model = ARIMA(dataset, order=(0,1,0))
model_fit = model.fit()
```

You can make a one-step out-of-sample forecast for a fit ARIMA model as follows:

```python
outcome = model_fit.forecast()[0]
```

The shampoo dataset has a trend so I’d recommend a d value of 1. Experiment with different p and q values and evaluate the predictions from resulting models.

For more help, see the post:

- <a href="">How to Create an ARIMA Model for Time Series Forecasting with Python</a>

## <a id="lesson07">Lesson 07: Hello World End-to-End Project</a>

ou now have the tools to work through a time series problem and develop a simple forecast model.

In this lesson, you will use the skills learned from all of the prior lessons to work through a new time series forecasting problem.

Download the monthy car sales dataset in CSV format and save it with the filename “monthly-car-sales.csv“.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv">Download the dataset</a>

Split the data, perhaps extract the last 1 or 2 years to a separate file. Work through the problem and develop forecasts for the missing data, including:

1. Load and explore the dataset.
2. Visualize the dataset.
3. Develop a persistence model.
4. Develop an autoregressive model.
5. Develop an ARIMA model.
6. Visualize forecasts and summarize forecast error.

You discovered:

- How to frame a time series forecasting problem as supervised learning.
- How to load and explore time series data with Pandas.
- How to plot and visualize time series data a number of different ways.
- How to develop a naive forecast called the persistence model as a baseline.
- How to develop an autoregressive forecast model using lagged observations.
- How to develop an ARIMA model including autoregression, integration and moving average elements.
- How to pull all of these elements together into an end-to-end project.
