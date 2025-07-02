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

