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
4. [Persistence.](#persistence)
5. [Data Analysis.](#analysis)
6. [ARIMA Models.](#arimamodels)
7. [Model Validation.](#validation)

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

Candidate models will be evaluated using walk-forward validation.

This is because a rolling-forecast type model is required from the problem definition. This is where one-step forecasts are needed given all available data.

The walk-forward validation will work as follows:

- The first 50% of the dataset will be held back to train the model.
- The remaining 50% of the dataset will be iterated and test the model.
- For each step in the test dataset:
    - A model will be trained.
    - A one-step prediction made and the prediction stored for later evaluation.
    - The actual observation from the test dataset will be added to the training dataset for the next iteration.
- The predictions made during the iteration of the test dataset will be evaluated and an RMSE score reported.

Given the small size of the data, we will allow a model to be re-trained given all available data prior to each prediction.

We can write the code for the test harness using simple NumPy and Python code.

Firstly, we can split the dataset into train and test sets directly. We’re careful to always convert a loaded dataset to *float32* in case the loaded data still has some *String* or *Integer* data types.

```python
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
```

Next, we can iterate over the time steps in the test dataset. The train dataset is stored in a Python list as we need to easily append a new observation each iteration and NumPy array concatenation feels like overkill.

The prediction made by the model is called *yhat* for convention, as the outcome or observation is referred to as *y* and *yhat* (a ‘*y*‘ with a mark above) is the mathematical notation for the prediction of the y variable.

The prediction and observation are printed each observation for a sanity check prediction in case there are issues with the model.

```python
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = ...
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
```

### <a id="persistence">4. Persistence</a>

The first step before getting bogged down in data analysis and modeling is to establish a baseline of performance.

This will provide both a template for evaluating models using the proposed test harness and a performance measure by which all more elaborate predictive models can be compared.

The baseline prediction for time series forecasting is called the naive forecast, or persistence.

This is where the observation from the previous time step is used as the prediction for the observation at the next time step.

We can plug this directly into the test harness defined in the previous section.

The complete code listing is provided below.

```python
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
```

Running the test harness prints the prediction and observation for each iteration of the test dataset.

The example ends by printing the RMSE for the model.

In this case, we can see that the persistence model achieved an RMSE of 3186.501. This means that on average, the model was wrong by about 3,186 million sales for each prediction made.

```
>Predicted=4676.000, Expected=5010
>Predicted=5010.000, Expected=4874
>Predicted=4874.000, Expected=4633
>Predicted=4633.000, Expected=1659
>Predicted=1659.000, Expected=5951
RMSE: 3186.501
```

We now have a baseline prediction method and performance; now we can start digging into our data.

### <a id="analysis">5. Data Analysis</a>

We can use summary statistics and plots of the data to quickly learn more about the structure of the prediction problem.

In this section, we will look at the data from five perspectives:

1. [Summary Statistics.](#summary)
2. [Line Plot.](#lineplot)
3. [Seasonal Line Plots.](#seasonallineplots)
4. [Density Plots.](#densityplot)
5. [Box and Whisker Plot.](#boxwhiskerplots)

#### <a id="summary">5.1 Summary Statistics</a>

Summary statistics provide a quick look at the limits of observed values. It can help to get a quick idea of what we are working with.

The example below calculates and prints summary statistics for the time series.

```python
from pandas import read_csv
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
print(series.describe())
```

Running the example provides a number of summary statistics to review.

Some observations from these statistics include:

- The number of observations (count) matches our expectation, meaning we are handling the data correctly.
- The mean is about 4,641, which we might consider our level in this series.
- The standard deviation (average spread from the mean) is relatively large at 2,486 sales.
- The percentiles along with the standard deviation do suggest a large spread to the data.

```
                  1
count     93.000000
mean    4641.118280
std     2486.403841
min     1573.000000
25%     3036.000000
50%     4016.000000
75%     5048.000000
max    13916.000000
```

#### <a id="lineplot">5.2 Line Plot</a>

A line plot of a time series can provide a lot of insight into the problem.

The example below creates and shows a line plot of the dataset.

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
series.plot()
pyplot.show()
```

Run the example and review the plot. Note any obvious temporal structures in the series.

Some observations from the plot include:

- There may be an increasing trend of sales over time.
- There appears to be systematic seasonality to the sales for each year.
- The seasonal signal appears to be growing over time, suggesting a multiplicative relationship (increasing change).
- There do not appear to be any obvious outliers.
- The seasonality suggests that the series is almost certainly non-stationary.

<img width="403" alt="image" src="https://github.com/user-attachments/assets/8e1caae9-5b7b-432c-b7b3-d95f240fafd1" />

There may be benefit in explicitly modeling the seasonal component and removing it. You may also explore using differencing with one or two levels in order to make the series stationary.

The increasing trend or growth in the seasonal component may suggest the use of a log or other power transform.

#### <a id="seasonallineplots">5.3 Seasonal Line Plots</a>

We can confirm the assumption that the seasonality is a yearly cycle by eyeballing line plots of the dataset by year.

The example below takes the 7 full years of data as separate groups and creates one line plot for each. The line plots are aligned vertically to help spot any year-to-year pattern.

```python
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
pyplot.figure()
i = 1
n_groups = len(groups)
for name, group in groups:
	pyplot.subplot((n_groups*100) + 10 + i)
	i += 1
	pyplot.plot(group)
pyplot.show()
```

Running the example creates the stack of 7 line plots.

We can clearly see a dip each August and a rise from each August to December. This pattern appears the same each year, although at different levels.

This will help with any explicitly season-based modeling later.

<img width="403" alt="image" src="https://github.com/user-attachments/assets/15e434c8-c64e-4d44-91d7-8d4a0737a699" />

It might have been easier if all season line plots were added to the one graph to help contrast the data for each year.

#### <a id="densityplot">5.4 Density Plot</a>

Reviewing plots of the density of observations can provide further insight into the structure of the data.

The example below creates a histogram and density plot of the observations without any temporal structure.

```python
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
series.hist()
series.plot(kind='kde')
pyplot.show()
```

Run the example and review the plots.

Some observations from the plots include:

- The distribution is not Gaussian.
- The shape has a long right tail and may suggest an exponential distribution

<img width="379" alt="image" src="https://github.com/user-attachments/assets/8aa23512-ff99-408e-8b40-281b29f781ad" />

<img width="424" alt="image" src="https://github.com/user-attachments/assets/2fe5858f-cc6a-439d-8c00-3390cd61d99d" />

This lends more support to exploring some power transforms of the data prior to modeling.

#### <a id="boxwhiskerplots">5.5 Box and Whisker Plots</a>

We can group the monthly data by year and get an idea of the spread of observations for each year and how this may be changing.

We do expect to see some trend (increasing mean or median), but it may be interesting to see how the rest of the distribution may be changing.

The example below groups the observations by year and creates one box and whisker plot for each year of observations. The last year (1971) only contains 9 months and may not be a useful comparison with the 12 months of observations for other years. Therefore, only data between 1964 and 1970 was plotted.

```python
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
groups = series['1964':'1970'].groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
	years[name.year] = group.values.flatten()
years.boxplot()
pyplot.show()
```

Running the example creates 7 box and whisker plots side-by-side, one for each of the 7 years of selected data.

Some observations from reviewing the plots include:

- The median values for each year (red line) may show an increasing trend.
- The spread or middle 50% of the data (blue boxes) does appear reasonably stable.
- There are outliers each year (black crosses); these may be the tops or bottoms of the seasonal cycle.
- The last year, 1970, does look different from the trend in prior years

<img width="392" alt="image" src="https://github.com/user-attachments/assets/37df963a-5ccf-4700-a3f3-a689e126f698" />

The observations suggest perhaps some growth trend over the years and outliers that may be a part of the seasonal cycle.

This yearly view of the data is an interesting avenue and could be pursued further by looking at summary statistics from year-to-year and changes in summary stats from year-to-year.

### <a id="arimamodels">6. ARIMA Models</a>

In this section, we will develop Autoregressive Integrated Moving Average, or ARIMA, models for the problem.

We will approach modeling by both manual and automatic configuration of the ARIMA model. This will be followed by a third step of investigating the residual errors of the chosen model.

As such, this section is broken down into 3 steps:

1. [Manually Configure the ARIMA.](#manuallyarima)
2. [Automatically Configure the ARIMA.](#gridsearch)
3. [Review Residual Errors.](#residual)

#### <a id="manuallyarima">6.1 Manually Configured ARIMA</a>

The ARIMA(*p,d,q*) model requires three parameters and is traditionally configured manually.

Analysis of the time series data assumes that we are working with a stationary time series.

The time series is almost certainly non-stationary. We can make it stationary this by first differencing the series and using a statistical test to confirm that the result is stationary.

The seasonality in the series is seemingly year-to-year. Seasonal data can be differenced by subtracting the observation from the same time in the previous cycle, in this case the same month in the previous year. This does mean that we will lose the first year of observations as there is no prior year to difference with.

The example below creates a deseasonalized version of the series and saves it to file *stationary.csv*.

```python
from pandas import read_csv
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
X = series.values
X = X.astype('float32').flatten()
# difference data
months_in_year = 12
stationary = difference(X, months_in_year)
stationary.index = series.index[months_in_year:]
# check if stationary
result = adfuller(stationary)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
# save
stationary.to_csv('stationary.csv', header=False)
# plot
stationary.plot()
pyplot.show()
```

Running the example outputs the result of a statistical significance test of whether the differenced series is stationary. Specifically, the augmented Dickey-Fuller test.

The results show that the test statistic value -7.134898 is smaller than the critical value at 1% of -3.515. This suggests that we can reject the null hypothesis with a significance level of less than 1% (i.e. a low probability that the result is a statistical fluke).

Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure.

```
ADF Statistic: -7.134898
p-value: 0.000000
Critical Values:
	1%: -3.515
	5%: -2.898
	10%: -2.586
```

For reference, the seasonal difference operation can be inverted by adding the observation for the same month the year before. This is needed in the case that predictions are made by a model fit on seasonally differenced data. The function to invert the seasonal difference operation is listed below for completeness.

```python
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
```

A plot of the differenced dataset is also created.

The plot does not show any obvious seasonality or trend, suggesting the seasonally differenced dataset is a good starting point for modeling.

We will use this dataset as an input to the ARIMA model. It also suggests that no further differencing may be required, and that the d parameter may be set to 0.

<img width="406" alt="image" src="https://github.com/user-attachments/assets/e6b0c495-1731-4df9-9395-af549f615c13" />

The next first step is to select the lag values for the Autoregression (AR) and Moving Average (MA) parameters, *p* and *q* respectively.

We can do this by reviewing Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.

Note, we are now using the seasonally differenced stationary.csv as our dataset.

The example below creates ACF and PACF plots for the series.

```python
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
series = read_csv('stationary.csv', header=None, index_col=0, parse_dates=True)
pyplot.figure()
pyplot.subplot(211)
plot_acf(series, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(series, ax=pyplot.gca())
pyplot.show()
```

Run the example and review the plots for insights into how to set the p and q variables for the ARIMA model.

Below are some observations from the plots.

- The ACF shows a significant lag for 1 month.
- The PACF shows a significant lag for 1 month, with perhaps some significant lag at 12 and 13 months.
- Both the ACF and PACF show a drop-off at the same point, perhaps suggesting a mix of AR and MA.

A good starting point for the p and q values is also 1.

The PACF plot also suggests that there is still some seasonality present in the differenced data.

We may consider a better model of seasonality, such as modeling it directly and explicitly removing it from the model rather than seasonal differencing.

<img width="385" alt="image" src="https://github.com/user-attachments/assets/833d8cb8-1966-406f-a65b-84fccc47741a" />

This quick analysis suggests an ARIMA(1,0,1) on the stationary data may be a good starting point.

The historic observations will be seasonally differenced prior to the fitting of each ARIMA model. The differencing will be inverted for all predictions made to make them directly comparable to the expected observation in the original sale count units.

Experimentation shows that this configuration of ARIMA does not converge and results in errors by the underlying library. Further experimentation showed that adding one level of differencing to the stationary data made the model more stable. The model can be extended to ARIMA(1,1,1).

We will also disable the automatic addition of a trend constant from the model by setting the ‘*trend*‘ argument to ‘*nc*‘ for no constant in the call to fit(). From experimentation, I find that this can result in better forecast performance on some problems.

The example below demonstrates the performance of this ARIMA model on the test harness.

```python
# evaluate manually configured ARIMA model
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(1,1,1))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
```

Note, you may see a warning message from the underlying linear algebra library; this can be ignored for now.

Running this example results in an RMSE of 956.942, which is dramatically better than the persistence RMSE of 3186.501.

```
>Predicted=3156.598, Expected=5010.000
>Predicted=4693.892, Expected=4874.000
>Predicted=4663.922, Expected=4633.000
>Predicted=2053.645, Expected=1659.000
>Predicted=5440.866, Expected=5951.000
RMSE: 961.619
```

This is a great start, but we may be able to get improved results with a better configured ARIMA model.

#### <a id="gridsearch">6.2 Grid Search ARIMA Hyperparameters</a>

The ACF and PACF plots suggest that an ARIMA(1,0,1) or similar may be the best that we can do.

To confirm this analysis, we can grid search a suite of ARIMA hyperparameters and check that no models result in better out of sample RMSE performance.

In this section, we will search values of *p*, *d*, and *q* for combinations (skipping those that fail to converge), and find the combination that results in the best performance on the test set. We will use a grid search to explore all combinations in a subset of integer values.

Specifically, we will search all combinations of the following parameters:

- *p*: 0 to 6.
- *d*: 0 to 2.
- *q*: 0 to 6.

This is (7 * 3 * 7), or 147, potential runs of the test harness and will take some time to execute.

It may be interesting to evaluate MA models with a lag of 12 or 13 as were noticed as potentially interesting from reviewing the ACF and PACF plots. Experimentation suggested that these models may not be stable, resulting in errors in the underlying mathematical libraries.

The complete worked example with the grid search version of the test harness is listed below.

```python
# grid search ARIMA parameters for time series
import warnings
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.50)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		# difference data
		months_in_year = 12
		diff = difference(history, months_in_year)
		model = ARIMA(diff, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		yhat = inverse_difference(history, yhat, months_in_year)
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# evaluate parameters
p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

Running the example runs through all combinations and reports the results on those that converge without error. The example takes a little over 2 minutes to run on modern hardware.

The results show that the best configuration discovered was ARIMA(0, 0, 1) with an RMSE of 939.464, slightly lower than the manually configured ARIMA from the previous section. This difference may or may not be statistically significant.

```
ARIMA(6, 2, 0) RMSE=1225.748
ARIMA(6, 2, 1) RMSE=1033.275
ARIMA(6, 2, 2) RMSE=1034.780
ARIMA(6, 2, 3) RMSE=1022.608
ARIMA(6, 2, 4) RMSE=1055.113
ARIMA(6, 2, 5) RMSE=1084.804
ARIMA(6, 2, 6) RMSE=1049.919
Best ARIMA(5, 1, 4) RMSE=930.225
```

We will select this ARIMA(5, 1, 4) model going forward.

#### <a id="residual">6.3 Review Residual Errors</a>

A good final check of a model is to review residual forecast errors.

Ideally, the distribution of residual errors should be a Gaussian with a zero mean.

We can check this by using summary statistics and plots to investigate the residual errors from the ARIMA(5, 1, 4) model. The example below calculates and summarizes the residual forecast errors.

```python
# summarize ARIMA forecast residuals
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32').flatten()
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(5,1,4))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

Running the example first describes the distribution of the residuals.

We can see that the distribution has a right shift and that the mean is non-zero at -126.489093.

This is perhaps a sign that the predictions are biased.

```
                 0
count    47.000000
mean   -126.489093
std     931.548799
min   -2327.141995
25%    -542.987301
50%    -292.763350
75%     425.130638
max    2000.046739
```

The distribution of residual errors is also plotted.

The graphs suggest a Gaussian-like distribution with a bumpy left tail, providing further evidence that perhaps a power transform might be worth exploring.

<img width="425" alt="image" src="https://github.com/user-attachments/assets/32e69989-eadc-4ce5-bc6b-b248894117e3" />

We could use this information to bias-correct predictions by adding the mean residual error of -126.489093 to each forecast made.

The example below performs this bias correlation.

```python
# plots of residual errors of bias corrected forecasts
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32').flatten()
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
bias = -126.489093
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(5,1,4))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()
```

The performance of the predictions is improved very slightly from 930.225 to 921.585, which may or may not be significant.

The summary of the forecast residual errors shows that the mean was indeed moved to a value very close to zero.

```
RMSE: 921.585
                  0
count  4.700000e+01
mean  -3.645676e-07
std    9.315488e+02
min   -2.200653e+03
25%   -4.164982e+02
50%   -1.662743e+02
75%    5.516197e+02
max    2.126536e+03
```

Finally, density plots of the residual error do show a small shift towards zero.

It is debatable whether this bias correction is worth it, but we will use it for now.

<img width="426" alt="image" src="https://github.com/user-attachments/assets/6173f312-4d1d-463f-baa5-ee9db8334960" />

It is also a good idea to check the time series of the residual errors for any type of autocorrelation. If present, it would suggest that the model has more opportunity to model the temporal structure in the data.

The example below re-calculates the residual errors and creates ACF and PACF plots to check for any significant autocorrelation.

```python
# ACF and PACF plots of residual errors of bias corrected forecasts
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32').flatten()
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(5,1,4))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
pyplot.show()
```

The results suggest that what little autocorrelation is present in the time series has been captured by the model.

```
                 0
count    47.000000
mean   -126.489093
std     931.548799
min   -2327.141995
25%    -542.987301
50%    -292.763350
75%     425.130638
max    2000.046739
```

<img width="391" alt="image" src="https://github.com/user-attachments/assets/b8fcb726-079d-4dcf-aafb-fc2017c11f53" />

### <a id="validation">7. Model Validation</a>

After models have been developed and a final model selected, it must be validated and finalized.

Validation is an optional part of the process, but one that provides a ‘last check’ to ensure we have not fooled or misled ourselves.

This section includes the following steps:

1. **[Finalize Model](#finalize)**: Train and save the final model.
2. **[Make Prediction](#prediction)**: Load the finalized model and make a prediction.
3. **[Validate Model](#validatemodel)**: Load and validate the final model.

#### <a id="finalize">7.1 Finalize Model</a>

Finalizing the model involves fitting an ARIMA model on the entire dataset, in this case on a transformed version of the entire dataset.

Once fit, the model can be saved to file for later use.

The example below trains an ARIMA(5,1,4) model on the dataset and saves the whole fit object and the bias to file.

The example below saves the fit model to file in the correct state so that it can be loaded successfully later.

```python
# save finalized model
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# load data
series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
# prepare data
X = series.values
X = X.astype('float32').flatten()
# difference data
months_in_year = 12
diff = difference(X, months_in_year)
# fit model
model = ARIMA(diff, order=(5,1,4))
model_fit = model.fit()
# bias constant, could be calculated from in-sample mean residual
bias = -126.489093
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])
```

Running the example creates two local files:

- *model.pkl* This is the ARIMAResult object from the call to *ARIMA.fit()*. This includes the coefficients and all other internal data returned when fitting the model.
- *model_bias.npy* This is the bias value stored as a one-row, one-column NumPy array.

#### <a id="prediction">7.2 Make Prediction</a>

A natural case may be to load the model and make a single forecast.

This is relatively straightforward and involves restoring the saved model and the bias and calling the forecast() method. To invert the seasonal differencing, the historical data must also be loaded.

The example below loads the model, makes a prediction for the next time step, and prints the prediction.

```python
# load finalized model and make a prediction
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMAResults
import numpy

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

series = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
months_in_year = 12
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(series.values, yhat, months_in_year)
print('Predicted: %.3f' % yhat)
```

Running the example prints the prediction of about 6660.

```
Predicted: 6660.516
```

If we peek inside *validation.csv*, we can see that the value on the first row for the next time period is 6981.

The prediction is in the right ballpark.

#### <a id="validatemodel">7.3 Validate Model</a>

We can load the model and use it in a pretend operational manner.

In the test harness section, we saved the final 12 months of the original dataset in a separate file to validate the final model.

We can load this *validation.csv* file now and use it see how well our model really is on “unseen” data.

There are two ways we might proceed:

- Load the model and use it to forecast the next 12 months. The forecast beyond the first one or two months will quickly start to degrade in skill.
- Load the model and use it in a rolling-forecast manner, updating the transform and model for each time step. This is the preferred method as it is how one would use this model in practice as it would achieve the best performance.

As with model evaluation in previous sections, we will make predictions in a rolling-forecast manner. This means that we will step over lead times in the validation dataset and take the observations as an update to the history.

```python
# load and evaluate the finalized model on the validation dataset
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# load and prepare datasets
dataset = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True)
X = dataset.values.astype('float32').flatten()
history = [x for x in X]
months_in_year = 12
validation = read_csv('validation.csv', header=None, index_col=0, parse_dates=True)
y = validation.values.astype('float32').flatten()
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%.3f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# difference data
	months_in_year = 12
	diff = difference(history, months_in_year)
	# predict
	model = ARIMA(diff, order=(5,1,4))
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	yhat = bias + inverse_difference(history, yhat, months_in_year)
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
pyplot.plot(y)
pyplot.plot(predictions, color='red')
pyplot.show()
```

Running the example prints each prediction and expected value for the time steps in the validation dataset.

The final RMSE for the validation period is predicted at 497.300 million sales.

This is much better than the expectation of an error of a little more than 924 million sales per month.

```
>Predicted=6660.516, Expected=6981.000
>Predicted=9796.256, Expected=9851.000
>Predicted=13643.568, Expected=12670.000
>Predicted=3808.467, Expected=4348.000
>Predicted=3011.693, Expected=3564.000
>Predicted=4658.108, Expected=4577.000
>Predicted=5132.485, Expected=4788.000
>Predicted=4891.954, Expected=4618.000
>Predicted=4638.893, Expected=5312.000
>Predicted=5044.506, Expected=4298.000
>Predicted=1730.109, Expected=1413.000
>Predicted=5794.528, Expected=5877.000
RMSE: 497.300
```

A plot of the predictions compared to the validation dataset is also provided.

At this scale on the plot, the 12 months of forecast sales figures look fantastic.

<img width="403" alt="image" src="https://github.com/user-attachments/assets/80c8eb04-45c3-4cb9-be52-681ace2fb740" />

## Summary

In this tutorial, you discovered the steps and the tools for a time series forecasting project with Python.

We have covered a lot of ground in this tutorial; specifically:

- How to develop a test harness with a performance measure and evaluation method and how to quickly develop a baseline forecast and skill.
- How to use time series analysis to raise ideas for how to best model the forecast problem.
- How to develop an ARIMA model, save it, and later load it to make predictions on new data.

## Reference

- <a href="https://github.com/QiRi92/data_science/blob/main/time_series_forecasting/time_forecast_sample.ipynb">Codes</a>
