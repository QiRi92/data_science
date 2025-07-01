# Autoregression Models for Time Series Forecasting With Python

Autoregression is a time series model that uses observations from previous time steps as input to a regression equation to predict the value at the next time step.

It is a very simple idea that can result in accurate forecasts on a range of time series problems.

In this tutorial, you will discover how to implement an autoregressive model for time series forecasting with Python.

After completing this tutorial, you will know:

- How to explore your time series data for autocorrelation.
- How to develop an autocorrelation model and use it to make predictions.
- How to use a developed autocorrelation model to make rolling predictions.

## Autoregression

A regression model, such as linear regression, models an output value based on a linear combination of input values.

For example:

```
yhat = b0 + b1*X1
```

Where yhat is the prediction, b0 and b1 are coefficients found by optimizing the model on training data, and X is an input value.

This technique can be used on time series where input variables are taken as observations at previous time steps, called lag variables.

For example, we can predict the value for the next time step (t+1) given the observations at the last two time steps (t-1 and t-2). As a regression model, this would look as follows:

```
X(t+1) = b0 + b1*X(t-1) + b2*X(t-2)
```

Because the regression model uses data from the same input variable at previous time steps, it is referred to as an autoregression (regression of self).

## Autocorrelation

An autoregression model makes an assumption that the observations at previous time steps are useful to predict the value at the next time step.

This relationship between variables is called correlation.

If both variables change in the same direction (e.g. go up together or down together), this is called a positive correlation. If the variables move in opposite directions as values change (e.g. one goes up and one goes down), then this is called negative correlation.

We can use statistical measures to calculate the correlation between the output variable and values at previous time steps at various different lags. The stronger the correlation between the output variable and a specific lagged variable, the more weight that autoregression model can put on that variable when modeling.

Again, because the correlation is calculated between the variable and itself at previous time steps, it is called an autocorrelation. It is also called serial correlation because of the sequenced structure of time series data.

The correlation statistics can also help to choose which lag variables will be useful in a model and which will not.

Interestingly, if all lag variables show low or no correlation with the output variable, then it suggests that the time series problem may not be predictable. This can be very useful when getting started on a new dataset.

In this tutorial, we will investigate the autocorrelation of a univariate time series then develop an autoregression model and use it to make predictions.

Before we do that, let’s first review the Minimum Daily Temperatures data that will be used in the examples.

## Minimum Daily Temperatures Dataset

This dataset describes the minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia.

The units are in degrees Celsius and there are 3,650 observations. The source of the data is credited as the Australian Bureau of Meteorology.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv">Download the dataset</a>

Download the dataset into your current working directory with the filename “daily-min-temperatures.csv“.

The code below will load the dataset as a Pandas Series.

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
print(series.head())
series.plot()
pyplot.show()
```

Running the example prints the first 5 rows from the loaded dataset.

```
            Temp
Date            
1981-01-01  20.7
1981-01-02  17.9
1981-01-03  18.8
1981-01-04  14.6
1981-01-05  15.8
```

A line plot of the dataset is then created.

<img width="404" alt="image" src="https://github.com/user-attachments/assets/df5aa3fb-1a48-46f9-a2ae-87f0ba79ec89" />

## Quick Check for Autocorrelation

There is a quick, visual check that we can do to see if there is an autocorrelation in our time series dataset.

We can plot the observation at the previous time step (t-1) with the observation at the next time step (t+1) as a scatter plot.

This could be done manually by first creating a lag version of the time series dataset and using a built-in scatter plot function in the Pandas library.

But there is an easier way.

Pandas provides a built-in plot to do exactly this, called the <a href="http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#lag-plot">lag_plot()</a> function.

Below is an example of creating a lag plot of the Minimum Daily Temperatures dataset.

```python
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import lag_plot
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
lag_plot(series)
pyplot.show()
```

Running the example plots the temperature data (t) on the x-axis against the temperature on the previous day (t-1) on the y-axis.

<img width="419" alt="image" src="https://github.com/user-attachments/assets/8ad95bfa-e430-4c60-bae1-615912f05d33" />

We can see a large ball of observations along a diagonal line of the plot. It clearly shows a relationship or some correlation.

This process could be repeated for any other lagged observation, such as if we wanted to review the relationship with the last 7 days or with the same day last month or last year.

Another quick check that we can do is to directly calculate the correlation between the observation and the lag variable.

We can use a statistical test like the <a href="https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient">Pearson correlation coefficient</a>. This produces a number to summarize how correlated two variables are between -1 (negatively correlated) and +1 (positively correlated) with small values close to zero indicating low correlation and high values above 0.5 or below -0.5 showing high correlation.

Correlation can be calculated easily using the <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html">corr()</a> function on the DataFrame of the lagged dataset.

The example below creates a lagged version of the Minimum Daily Temperatures dataset and calculates a correlation matrix of each column with other columns, including itself.

```python
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)
```

This is a good confirmation for the plot above.

It shows a strong positive correlation (0.77) between the observation and the lag=1 value.

```
         t-1      t+1
t-1  1.00000  0.77487
t+1  0.77487  1.00000
```

This is good for one-off checks, but tedious if we want to check a large number of lag variables in our time series.

Next, we will look at a scaled-up version of this approach.

## Autocorrelation Plots

We can plot the correlation coefficient for each lag variable.

This can very quickly give an idea of which lag variables may be good candidates for use in a predictive model and how the relationship between the observation and its historic values changes over time.

We could manually calculate the correlation values for each lag variable and plot the result. Thankfully, Pandas provides a built-in plot called the <a href="http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#autocorrelation-plot">autocorrelation_plot()</a> function.

The plot provides the lag number along the x-axis and the correlation coefficient value between -1 and 1 on the y-axis. The plot also includes solid and dashed lines that indicate the 95% and 99% confidence interval for the correlation values. Correlation values above these lines are more significant than those below the line, providing a threshold or cutoff for selecting more relevant lag values.

```python
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
autocorrelation_plot(series)
pyplot.show()
```

Running the example shows the swing in positive and negative correlation as the temperature values change across summer and winter seasons each previous year.

<img width="420" alt="image" src="https://github.com/user-attachments/assets/f20f5124-311d-4c9d-943a-f4273f148e83" />

The statsmodels library also provides a version of the plot in the plot_acf() function as a line plot.

```python
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
plot_acf(series, lags=31)
pyplot.show()
```

In this example, we limit the lag variables evaluated to 31 for readability.

<img width="397" alt="image" src="https://github.com/user-attachments/assets/d07dee8d-d58d-486a-bd81-dc44354fc10e" />

Now that we know how to review the autocorrelation in our time series, let’s look at modeling it with an autoregression.

Before we do that, let’s establish a baseline performance.

## Persistence Model

Let’s say that we want to develop a model to predict the last 7 days of minimum temperatures in the dataset given all prior observations.

The simplest model that we could use to make predictions would be to persist the last observation. We can call this a persistence model and it provides a baseline of performance for the problem that we can use for comparison with an autoregression model.

We can develop a test harness for the problem by splitting the observations into training and test sets, with only the last 7 observations in the dataset assigned to the test set as “unseen” data that we wish to predict.

The predictions are made using a walk-forward validation model so that we can persist the most recent observations for the next day. This means that we are not making a 7-day forecast, but 7 1-day forecasts.

```python
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0)
# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
	return x

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
```

Running the example prints the mean squared error (MSE).

The value provides a baseline performance for the problem.

```
Test MSE: 3.423
```

The expected values for the next 7 days are plotted (blue) compared to the predictions from the model (red).

<img width="389" alt="image" src="https://github.com/user-attachments/assets/e4139459-fe42-49f9-bd16-4b767e8fa668" />

## Autoregression Model

An autoregression model is a linear regression model that uses lagged variables as input variables.

We could calculate the linear regression model manually using the LinearRegession class in scikit-learn and manually specify the lag input variables to use.

Alternately, the statsmodels library provides an autoregression model where you must specify an appropriate lag value and trains a linear regression model. It is provided in the <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html">AutoReg</a> class.

We can use this model by first creating the model AutoReg() and then calling fit() to train it on our dataset. This returns an <a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoRegResults.html">AutoRegResults</a> object.

Once fit, we can use the model to make a prediction by calling the predict() function for a number of observations in the future. This creates 1 7-day forecast, which is different from the persistence example above.

The complete example is listed below.

```python
# create and evaluate a static autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AutoReg(train, lags=29)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

Running the example the list of coefficients in the trained linear regression model.

The 7 day forecast is then printed and the mean squared error of the forecast is summarized.

```
Coefficients: [ 5.57543506e-01  5.88595221e-01 -9.08257090e-02  4.82615092e-02
  4.00650265e-02  3.93020055e-02  2.59463738e-02  4.46675960e-02
  1.27681498e-02  3.74362239e-02 -8.11700276e-04  4.79081949e-03
  1.84731397e-02  2.68908418e-02  5.75906178e-04  2.48096415e-02
  7.40316579e-03  9.91622149e-03  3.41599123e-02 -9.11961877e-03
  2.42127561e-02  1.87870751e-02  1.21841870e-02 -1.85534575e-02
 -1.77162867e-03  1.67319894e-02  1.97615668e-02  9.83245087e-03
  6.22710723e-03 -1.37732255e-03]
predicted=11.871275, expected=12.900000
predicted=13.053794, expected=14.600000
predicted=13.532591, expected=14.000000
predicted=13.243126, expected=13.600000
predicted=13.091438, expected=13.500000
predicted=13.146989, expected=15.700000
predicted=13.176153, expected=13.000000
Test RMSE: 1.225
```

A plot of the expected (blue) vs the predicted values (red) is made.

The forecast does look pretty good (about 1 degree Celsius out each day), with big deviation on day 5.

<img width="398" alt="image" src="https://github.com/user-attachments/assets/b4cf467f-c250-4774-85de-3565944aa554" />

The statsmodels API does not make it easy to update the model as new observations become available.

One way would be to re-train the AutoReg model each day as new observations become available, and that may be a valid approach, if not computationally expensive.

An alternative would be to use the learned coefficients and manually make predictions. This requires that the history of 29 prior observations be kept and that the coefficients be retrieved from the model and used in the regression equation to come up with new forecasts.

The coefficients are provided in an array with the intercept term followed by the coefficients for each lag variable starting at t-1 to t-n. We simply need to use them in the right order on the history of observations, as follows:

```
yhat = b0 + b1*X1 + b2*X2 ... bn*Xn
```

Below is the complete example.

```python
# create and evaluate an updated autoregressive model
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
series = read_csv('daily-min-temperatures.csv', header=0, index_col=0, parse_dates=True)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

Again, running the example prints the forecast and the mean squared error.

```
predicted=11.871275, expected=12.900000
predicted=13.659297, expected=14.600000
predicted=14.349246, expected=14.000000
predicted=13.427454, expected=13.600000
predicted=13.374877, expected=13.500000
predicted=13.479991, expected=15.700000
predicted=14.765146, expected=13.000000
Test RMSE: 1.204
```

We can see a small improvement in the forecast when comparing the error scores.

<img width="396" alt="image" src="https://github.com/user-attachments/assets/756dbb7f-cef8-403c-a216-2c7394d43582" />

## Further Reading

This section provides some resources if you are looking to dig deeper into autocorrelation and autoregression.

- <a href="https://en.wikipedia.org/wiki/Autocorrelation">Autocorrelation</a> on Wikipedia
- <a href="https://en.wikipedia.org/wiki/Autoregressive_model">Autoregressive</a> model on Wikipedia
- Chapter 7 – Regression-Based Models: Autocorrelation and External Information, <a href="https://amzn.to/33YCJag">Practical Time Series Forecasting with R: A Hands-On Guide</a>.
- Section 4.5 – Autoregressive Models, <a href="https://amzn.to/2XYS8DD">Introductory Time Series with R</a>.

## Summary

In this tutorial, you discovered how to make autoregression forecasts for time series data using Python.

Specifically, you learned:

- About autocorrelation and autoregression and how they can be used to better understand time series data.
- How to explore the autocorrelation in a time series using plots and statistical tests.
- How to train an autoregression model in Python and use it to make short-term and rolling forecasts.
