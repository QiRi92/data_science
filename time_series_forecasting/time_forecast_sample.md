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
