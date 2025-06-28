# How to Load and Explore Time Series Data in Python

The <a href="https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/">Pandas library</a> in Python provides excellent, built-in support for time series data.

Once loaded, Pandas also provides tools to explore and better understand your dataset.

In this post, you will discover how to load and explore your time series dataset.

After completing this tutorial, you will know:

- How to load your time series dataset from a CSV file using Pandas.
- How to peek at the loaded data and calculate summary statistics.
- How to plot and review your time series data.

## Daily Female Births Dataset

In this post, we will use the Daily Female Births Dataset as an example.

This univariate time series dataset describes the number of daily female births in California in 1959.

The units are a count and there are 365 observations. The source of the dataset is credited to Newton (1988).

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv">Download the dataset</a>

Below is a sample of the first 5 rows of data, including the header row.

```
"Date","Daily total female births in California, 1959"
"1959-01-01",35
"1959-01-02",32
"1959-01-03",30
"1959-01-04",31
"1959-01-05",44
```

Below is a plot of the entire dataset.

<img width="587" alt="image" src="https://github.com/user-attachments/assets/41e8a02b-5193-49bc-ab75-e7a62b31b4cf" />

Download the dataset and place it in your current working directory with the file name “**daily-total-female-births-in-cal.csv**“.

### Load Time Series Data

Pandas represented time series datasets as a Series.

A <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html">Series</a> is a one-dimensional array with a time label for each row.

The series has a name, which is the column name of the data column.

You can see that each row has an associated date. This is in fact not a column, but instead a time index for value. As an index, there can be multiple values for one time, and values may be spaced evenly or unevenly across times.

The main function for loading CSV data in Pandas is the <a href="http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html#pandas.read_csv">read_csv()</a> function. We can use this to load the time series as a Series object, instead of a DataFrame, as follows:

```python
# Load birth data using read_csv
from pandas import read_csv
series = read_csv('daily-total-female-births-in-cal.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
print(type(series))
print(series.head())
```

Note the arguments to the **read_csv()** function.

We provide it a number of hints to ensure the data is loaded as a Series.

- **header=0**: We must specify the header information at row 0.
- **parse_dates=[0]**: We give the function a hint that data in the first column contains dates that need to be parsed. This argument takes a list, so we provide it a list of one element, which is the index of the first column.
- **index_col=0**: We hint that the first column contains the index information for the time series.
- **squeeze=True**: We hint that we only have one data column and that we are interested in a Series and not a DataFrame.

One more argument you may need to use for your own data is **date_parser** to specify the function to parse date-time values. In this example, the date format has been inferred, and this works in most cases. In those few cases where it does not, specify your own date parsing function and use the **date_parser** argument.

Running the example above prints the same output, but also confirms that the time series was indeed loaded as a Series object.

```
<class 'pandas.core.series.Series'>
Date
1959-01-01 35
1959-01-02 32
1959-01-03 30
1959-01-04 31
1959-01-05 44
Name: Daily total female births in California, 1959, dtype: int64
```

It is often easier to perform manipulations of your time series data in a DataFrame rather than a Series object.

In those situations, you can easily convert your loaded Series to a DataFrame as follows:

```python
dataframe = DataFrame(series)
```

## Exploring Time Series Data

