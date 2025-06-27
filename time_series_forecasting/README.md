## INTRODUCTION

In this mini-course, you will discover how you can get started, build accurate models and confidently complete predictive modeling time series forecasting projects using Python.

Below are 7 lessons that will get you started and productive with machine learning in Python:

- [Lesson 01: Time Series as Supervised Learning.](#lesson01)
- [Lesson 02: Load Time Series Data.](#lesson02)
- [Lesson 03: Data Visualization.](#lesson03)
- Lesson 04: Persistence Forecast Model.
- Lesson 05: Autoregressive Forecast Model.
- Lesson 06: ARIMA Forecast Model.
- Lesson 07: Hello World End-to-End Project.

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

## <a id="lesson02">Lesson 02: Load Time Series Data</a>

Before you can develop forecast models, you must load and work with your time series data.

Pandas provides tools to load data in CSV format.

In this lesson, you will download a standard time series dataset, load it in Pandas and explore it.

Download the daily female births dataset in CSV format and save it with the filename “daily-births.csv“.

<a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv">Download the dataset</a>

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

## <a id="lesson03">Lesson 03: Data Visualization</a>
