# Time Series Data Visualization with Python

Time series lends itself naturally to visualization.

Line plots of observations over time are popular, but there is a suite of other plots that you can use to learn more about your problem.

The more you learn about your data, the more likely you are to develop a better forecasting model.

In this tutorial, you will discover 6 different types of plots that you can use to visualize time series data with Python.

Specifically, after completing this tutorial, you will know:

- How to explore the temporal structure of time series with line plots, lag plots, and autocorrelation plots.
- How to understand the distribution of observations using histograms and density plots.
- How to tease out the change in distribution over intervals using box and whisker plots and heat map plots.

## Time Series Visualization

Visualization plays an important role in time series analysis and forecasting.

Plots of the raw sample data can provide valuable diagnostics to identify temporal structures like trends, cycles, and seasonality that can influence the choice of model.

A problem is that many novices in the field of time series forecasting stop with line plots.

In this tutorial, we will take a look at 6 different types of visualizations that you can use on your own time series data. They are:

1. Line Plots.
2. Histograms and Density Plots.
3. Box and Whisker Plots.
4. Heat Maps.
5. Lag Plots or Scatter Plots.
6. Autocorrelation Plots.

The focus is on univariate time series, but the techniques are just as applicable to multivariate time series, when you have more than one observation at each time step.

Next, let’s take a look at the dataset we will use to demonstrate time series visualization in this tutorial.

## Minimum Daily Temperatures Dataset

This dataset describes the minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia.

The units are in degrees Celsius and there are 3,650 observations. The source of the data is credited as the Australian Bureau of Meteorology.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv">Download the dataset</a>

Download the dataset and place it in the current working directory with the filename “*daily-minimum-temperatures.csv*“.

Below is an example of loading the dataset as a Panda Series.
