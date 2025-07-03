# How to Grid Search ARIMA Model Hyperparameters with Python

The ARIMA model for time series analysis and forecasting can be tricky to configure.

There are 3 parameters that require estimation by iterative trial and error from reviewing diagnostic plots and using 40-year-old heuristic rules.

We can automate the process of evaluating a large number of hyperparameters for the ARIMA model by using a grid search procedure.

In this tutorial, you will discover how to tune the ARIMA model using a grid search of hyperparameters in Python.

After completing this tutorial, you will know:

- A general procedure that you can use to tune the ARIMA hyperparameters for a rolling one-step forecast.
- How to apply ARIMA hyperparameter optimization on a standard univariate time series dataset.
- Ideas for extending the procedure for more elaborate and robust models.

## Grid Searching Method

Diagnostic plots of the time series can be used along with heuristic rules to determine the hyperparameters of the ARIMA model.

These are good in most, but perhaps not all, situations.

We can automate the process of training and evaluating ARIMA models on different combinations of model hyperparameters. In machine learning this is called a grid search or model tuning.

In this tutorial, we will develop a method to grid search ARIMA hyperparameters for a one-step rolling forecast.

The approach is broken down into two parts:

1. Evaluate an ARIMA model.
2. Evaluate sets of ARIMA parameters.

The code in this tutorial makes use of the scikit-learn, Pandas, and the statsmodels Python libraries.

### 1. Evaluate ARIMA Model

We can evaluate an ARIMA model by preparing it on a training dataset and evaluating predictions on a test dataset.

This approach involves the following steps:

1. Split the dataset into training and test sets.
2. Walk the time steps in the test dataset.
  1. Train an ARIMA model.
  2. Make a one-step prediction.
  3. Store prediction; get and store actual observation.
3. Calculate error score for predictions compared to expected values.

We can implement this in Python as a new standalone function called *evaluate_arima_model()* that takes a time series dataset as input as well as a tuple with the p, d, and q parameters for the model to be evaluated.

The dataset is split in two: 66% for the initial training dataset and the remaining 34% for the test dataset.

Each time step of the test set is iterated. Just one iteration provides a model that you could use to make predictions on new data. The iterative approach allows a new ARIMA model to be trained each time step.

A prediction is made each iteration and stored in a list. This is so that at the end of the test set, all predictions can be compared to the list of expected values and an error score calculated. In this case, a mean squared error score is calculated and returned.

The complete function is listed below.

```python
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error
```

Now that we know how to evaluate one set of ARIMA hyperparameters, let’s see how we can call this function repeatedly for a grid of parameters to evaluate.

### 2. Iterate ARIMA Parameters

Evaluating a suite of parameters is relatively straightforward.

The user must specify a grid of p, d, and q ARIMA parameters to iterate. A model is created for each parameter and its performance evaluated by calling the *evaluate_arima_model()* function described in the previous section.

The function must keep track of the lowest error score observed and the configuration that caused it. This can be summarized at the end of the function with a print to standard out.

We can implement this function called *evaluate_models()* as a series of four loops.

There are two additional considerations. The first is to ensure the input data are floating point values (as opposed to integers or strings), as this can cause the ARIMA procedure to fail.

Second, the statsmodels ARIMA procedure internally uses numerical optimization procedures to find a set of coefficients for the model. These procedures can fail, which in turn can throw an exception. We must catch these exceptions and skip those configurations that cause a problem. This happens more often then you would think.

Additionally, it is recommended that warnings be ignored for this code to avoid a lot of noise from running the procedure. This can be done as follows:

```python
import warnings
warnings.filterwarnings("ignore")
```

Finally, even with all of these protections, the underlying C and Fortran libraries may still report warnings to standard out, such as:

```
** On entry to DLASCL, parameter number 4 had an illegal value
```

These have been removed from the results reported in this tutorial for brevity.

The complete procedure for evaluating a grid of ARIMA hyperparameters is listed below.

```python
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
```

Now that we have a procedure to grid search ARIMA hyperparameters, let’s test the procedure on two univariate time series problems.

We will start with the Shampoo Sales dataset.

## Shampoo Sales Case Study

The Shampoo Sales dataset describes the monthly number of sales of shampoo over a 3-year period.

The units are a sales count and there are 36 observations. The original dataset is credited to Makridakis, Wheelwright, and Hyndman (1998).

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv">Download the dataset</a>.

Download the dataset and place it into your current working directory with the filename “*shampoo-sales.csv*“.

The timestamps in the time series do not contain an absolute year component. We can use a custom date-parsing function when loading the data and baseline the year from 1900, as follows:

```python
import pandas as pd
from datetime import datetime

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, date_parser=parser)
```

Once loaded, we can specify a site of p, d, and q values to search and pass them to the evaluate_models() function.

We will try a suite of lag values (p) and just a few difference iterations (d) and residual error lag values (q).

```python
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

Putting this all together with the generic procedures defined in the previous section, we can grid search ARIMA hyperparameters in the Shampoo Sales dataset.

The complete code example is listed below.

```python
# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import read_csv
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
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
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, date_parser=parser)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

Running the example prints the ARIMA parameters and MSE for each successful evaluation completed.

The best parameters of ARIMA(1, 2, 2) are reported at the end of the run with a root mean squared error of 65.511.

```
ARIMA(0, 0, 0) RMSE=228.966
ARIMA(0, 0, 1) RMSE=195.596
ARIMA(0, 0, 2) RMSE=154.886
ARIMA(0, 1, 0) RMSE=133.156
ARIMA(0, 1, 1) RMSE=104.077
ARIMA(0, 1, 2) RMSE=68.345
ARIMA(0, 2, 0) RMSE=255.187
ARIMA(0, 2, 1) RMSE=134.168
ARIMA(0, 2, 2) RMSE=74.644
ARIMA(1, 0, 0) RMSE=152.028
ARIMA(1, 0, 1) RMSE=111.787
ARIMA(1, 0, 2) RMSE=77.072
ARIMA(1, 1, 0) RMSE=88.631
ARIMA(1, 1, 1) RMSE=87.942
ARIMA(1, 1, 2) RMSE=90.986
ARIMA(1, 2, 0) RMSE=134.576
ARIMA(1, 2, 1) RMSE=86.157
ARIMA(1, 2, 2) RMSE=65.511
ARIMA(2, 0, 0) RMSE=100.879
ARIMA(2, 0, 1) RMSE=98.953
ARIMA(2, 0, 2) RMSE=98.689
ARIMA(2, 1, 0) RMSE=85.063
ARIMA(2, 1, 1) RMSE=88.428
ARIMA(2, 1, 2) RMSE=83.501
ARIMA(2, 2, 0) RMSE=97.829
ARIMA(2, 2, 1) RMSE=76.847
ARIMA(2, 2, 2) RMSE=80.811
ARIMA(4, 0, 0) RMSE=100.975
ARIMA(4, 0, 1) RMSE=101.463
ARIMA(4, 0, 2) RMSE=97.567
ARIMA(4, 1, 0) RMSE=95.068
ARIMA(4, 1, 1) RMSE=84.813
ARIMA(4, 1, 2) RMSE=84.205
ARIMA(4, 2, 0) RMSE=85.397
ARIMA(4, 2, 1) RMSE=74.219
ARIMA(4, 2, 2) RMSE=70.147
ARIMA(6, 0, 0) RMSE=96.000
ARIMA(6, 0, 1) RMSE=85.032
ARIMA(6, 0, 2) RMSE=96.301
ARIMA(6, 1, 0) RMSE=84.633
ARIMA(6, 1, 1) RMSE=78.365
ARIMA(6, 1, 2) RMSE=74.074
ARIMA(6, 2, 0) RMSE=77.305
ARIMA(6, 2, 1) RMSE=77.721
ARIMA(6, 2, 2) RMSE=82.558
ARIMA(8, 0, 0) RMSE=88.828
ARIMA(8, 0, 1) RMSE=94.800
ARIMA(8, 0, 2) RMSE=99.933
ARIMA(8, 1, 0) RMSE=79.987
ARIMA(8, 1, 1) RMSE=79.962
ARIMA(8, 1, 2) RMSE=75.769
ARIMA(8, 2, 0) RMSE=81.882
ARIMA(8, 2, 1) RMSE=83.656
ARIMA(8, 2, 2) RMSE=91.010
ARIMA(10, 0, 0) RMSE=90.833
ARIMA(10, 0, 1) RMSE=91.810
ARIMA(10, 0, 2) RMSE=96.498
ARIMA(10, 1, 0) RMSE=84.839
ARIMA(10, 1, 1) RMSE=84.963
ARIMA(10, 1, 2) RMSE=78.407
ARIMA(10, 2, 0) RMSE=85.105
ARIMA(10, 2, 1) RMSE=75.733
ARIMA(10, 2, 2) RMSE=73.484
Best ARIMA(1, 2, 2) RMSE=65.511
```

## Daily Female Births Case Study

The Daily Female Births dataset describes the number of daily female births in California in 1959.

The units are a count and there are 365 observations. The source of the dataset is credited to Newton (1988).

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv">Download the dataset</a>.

Download the dataset and place it in your current working directory with the filename “*daily-total-female-births.csv*“.

This dataset can be easily loaded directly as a Pandas Series.

```python
# load dataset
series = pd.read_csv('daily-total-female-births.csv', header=0, index_col=0)
```

To keep things simple, we will explore the same grid of ARIMA hyperparameters as in the previous section.

```python
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

Output:

```
ARIMA(0, 0, 0) RMSE=8.189
ARIMA(0, 0, 1) RMSE=7.884
ARIMA(0, 0, 2) RMSE=7.771
ARIMA(0, 1, 0) RMSE=9.151
ARIMA(0, 1, 1) RMSE=7.427
ARIMA(0, 1, 2) RMSE=7.352
ARIMA(0, 2, 0) RMSE=15.670
ARIMA(0, 2, 1) RMSE=9.167
ARIMA(0, 2, 2) RMSE=7.453
ARIMA(1, 0, 0) RMSE=7.802
ARIMA(1, 0, 1) RMSE=7.568
ARIMA(1, 0, 2) RMSE=7.551
ARIMA(1, 1, 0) RMSE=8.106
ARIMA(1, 1, 1) RMSE=7.340
ARIMA(1, 1, 2) RMSE=7.329
ARIMA(1, 2, 0) RMSE=11.968
ARIMA(1, 2, 1) RMSE=8.120
ARIMA(1, 2, 2) RMSE=7.407
ARIMA(2, 0, 0) RMSE=7.697
ARIMA(2, 0, 1) RMSE=7.538
ARIMA(2, 1, 0) RMSE=7.700
ARIMA(2, 1, 1) RMSE=7.332
ARIMA(2, 1, 2) RMSE=7.356
ARIMA(2, 2, 0) RMSE=10.355
ARIMA(2, 2, 1) RMSE=7.714
ARIMA(4, 0, 0) RMSE=7.693
ARIMA(4, 0, 1) RMSE=7.505
ARIMA(4, 0, 2) RMSE=10.707
ARIMA(4, 1, 0) RMSE=7.565
ARIMA(4, 1, 1) RMSE=7.396
ARIMA(4, 1, 2) RMSE=7.320
ARIMA(4, 2, 0) RMSE=8.940
ARIMA(4, 2, 1) RMSE=7.577
ARIMA(6, 0, 0) RMSE=7.666
ARIMA(6, 1, 0) RMSE=7.281
ARIMA(6, 1, 1) RMSE=7.340
ARIMA(6, 1, 2) RMSE=7.433
ARIMA(6, 2, 0) RMSE=8.337
ARIMA(6, 2, 1) RMSE=7.292
ARIMA(8, 0, 0) RMSE=7.549
ARIMA(8, 0, 1) RMSE=7.565
ARIMA(8, 0, 2) RMSE=7.602
ARIMA(8, 1, 0) RMSE=7.555
ARIMA(8, 1, 2) RMSE=7.410
ARIMA(8, 2, 0) RMSE=8.112
ARIMA(8, 2, 1) RMSE=7.560
ARIMA(8, 2, 2) RMSE=7.489
ARIMA(10, 0, 0) RMSE=7.581
ARIMA(10, 0, 1) RMSE=7.622
ARIMA(10, 0, 2) RMSE=7.614
ARIMA(10, 1, 0) RMSE=7.560
ARIMA(10, 1, 1) RMSE=7.402
ARIMA(10, 1, 2) RMSE=7.377
ARIMA(10, 2, 0) RMSE=8.079
ARIMA(10, 2, 1) RMSE=7.565
ARIMA(10, 2, 2) RMSE=7.511
Best ARIMA(6, 1, 0) RMSE=7.281
```

Putting this all together, we can grid search ARIMA parameters on the Daily Female Births dataset. The complete code listing is provided below.

```python
# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
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
series = read_csv('daily-total-female-births.csv', header=0, index_col=0, parse_dates=True)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)
```

Running the example prints the ARIMA parameters and mean squared error for each configuration successfully evaluated.

The best mean parameters are reported as ARIMA(6, 1, 0) with a mean squared error of 53.187.

```
ARIMA(0, 0, 0) RMSE=8.189
ARIMA(0, 0, 1) RMSE=7.884
ARIMA(0, 0, 2) RMSE=7.771
ARIMA(0, 1, 0) RMSE=9.151
ARIMA(0, 1, 1) RMSE=7.427
ARIMA(0, 1, 2) RMSE=7.352
ARIMA(0, 2, 0) RMSE=15.670
ARIMA(0, 2, 1) RMSE=9.167
ARIMA(0, 2, 2) RMSE=7.453
ARIMA(1, 0, 0) RMSE=7.802
ARIMA(1, 0, 1) RMSE=7.568
ARIMA(1, 0, 2) RMSE=7.551
ARIMA(1, 1, 0) RMSE=8.106
ARIMA(1, 1, 1) RMSE=7.340
ARIMA(1, 1, 2) RMSE=7.329
ARIMA(1, 2, 0) RMSE=11.968
ARIMA(1, 2, 1) RMSE=8.120
ARIMA(1, 2, 2) RMSE=7.407
ARIMA(2, 0, 0) RMSE=7.697
ARIMA(2, 0, 1) RMSE=7.538
ARIMA(2, 1, 0) RMSE=7.700
ARIMA(2, 1, 1) RMSE=7.332
ARIMA(2, 1, 2) RMSE=7.356
ARIMA(2, 2, 0) RMSE=10.355
ARIMA(2, 2, 1) RMSE=7.714
ARIMA(4, 0, 0) RMSE=7.693
ARIMA(4, 0, 1) RMSE=7.505
ARIMA(4, 0, 2) RMSE=10.707
ARIMA(4, 1, 0) RMSE=7.565
ARIMA(4, 1, 1) RMSE=7.396
ARIMA(4, 1, 2) RMSE=7.320
ARIMA(4, 2, 0) RMSE=8.940
ARIMA(4, 2, 1) RMSE=7.577
ARIMA(6, 0, 0) RMSE=7.666
ARIMA(6, 1, 0) RMSE=7.281
ARIMA(6, 1, 1) RMSE=7.340
ARIMA(6, 1, 2) RMSE=7.433
ARIMA(6, 2, 0) RMSE=8.337
ARIMA(6, 2, 1) RMSE=7.292
ARIMA(8, 0, 0) RMSE=7.549
ARIMA(8, 0, 1) RMSE=7.565
ARIMA(8, 0, 2) RMSE=7.602
ARIMA(8, 1, 0) RMSE=7.555
ARIMA(8, 1, 2) RMSE=7.410
ARIMA(8, 2, 0) RMSE=8.112
ARIMA(8, 2, 1) RMSE=7.560
ARIMA(8, 2, 2) RMSE=7.489
ARIMA(10, 0, 0) RMSE=7.581
ARIMA(10, 0, 1) RMSE=7.622
ARIMA(10, 0, 2) RMSE=7.614
ARIMA(10, 1, 0) RMSE=7.560
ARIMA(10, 1, 1) RMSE=7.402
ARIMA(10, 1, 2) RMSE=7.377
ARIMA(10, 2, 0) RMSE=8.079
ARIMA(10, 2, 1) RMSE=7.565
ARIMA(10, 2, 2) RMSE=7.511
Best ARIMA(6, 1, 0) RMSE=7.281
```

## Extensions

The grid search method used in this tutorial is simple and can easily be extended.

This section lists some ideas to extend the approach you may wish to explore.

- **Seed Grid**. The classical diagnostic tools of ACF and PACF plots can still be used with the results used to seed the grid of ARIMA parameters to search.
- **Alternate Measures**. The search seeks to optimize the out-of-sample mean squared error. This could be changed to another out-of-sample statistic, an in-sample statistic, such as AIC or BIC, or some combination of the two. You can choose a metric that is most meaningful on your project.
- **Residual Diagnostics**. Statistics can automatically be calculated on the residual forecast errors to provide an additional indication of the quality of the fit. Examples include statistical tests for whether the distribution of residuals is Gaussian and whether there is an autocorrelation in the residuals.
- **Update Model**. The ARIMA model is created from scratch for each one-step forecast. With careful inspection of the API, it may be possible to update the internal data of the model with new observations rather than recreating it from scratch.
- **Preconditions**. The ARIMA model can make assumptions about the time series dataset, such as normality and stationarity. These could be checked and a warning raised for a given of a dataset prior to a given model being trained.

## Summary

In this tutorial, you discovered how to grid search the hyperparameters for the ARIMA model in Python.

Specifically, you learned:

- A procedure that you can use to grid search ARIMA hyperparameters for a one-step rolling forecast.
- How to apply ARIMA hyperparameters tuning on standard univariate time series datasets.
- Ideas on how to further improve grid searching of ARIMA hyperparameters.

Reference

- <a href="https://github.com/QiRi92/data_science/blob/main/time_series_forecasting/arima_hyperparameters.ipynb" rel="noopener" target="_blank">Codes</a>
