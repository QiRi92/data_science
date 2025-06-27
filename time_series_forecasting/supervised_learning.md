# Time Series Forecasting as Supervised Learning

Time series forecasting can be framed as a supervised learning problem.

This re-framing of your time series data allows you access to the suite of standard linear and nonlinear machine learning algorithms on your problem.

In this post, you will discover how you can re-frame your time series problem as a supervised learning problem for machine learning. After reading this post, you will know:

- What supervised learning is and how it is the foundation for all predictive modeling machine learning algorithms.
- The sliding window method for framing a time series dataset and how to use it.
- How to use the sliding window for multivariate data and multi-step forecasting.

## Supervised Machine Learning

The majority of practical machine learning uses supervised learning.

Supervised learning is where you have input variables (X) and an output variable (y) and you use an algorithm to learn the mapping function from the input to the output.

```
Y = f(X)
```

The goal is to approximate the real underlying mapping so well that when you have new input data (X), you can predict the output variables (y) for that data.

Below is a contrived example of a supervised learning dataset where each row is an observation comprised of one input variable (X) and one output variable to be predicted (y).

```
X, y
5, 0.9
4, 0.8
5, 1.0
3, 0.7
4, 0.9
```

It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process.

We know the correct answers; the algorithm iteratively makes predictions on the training data and is corrected by making updates. Learning stops when the algorithm achieves an acceptable level of performance.

Supervised learning problems can be further grouped into regression and classification problems.

- Classification: A classification problem is when the output variable is a category, such as “red” and “blue” or “disease” and “no disease.”
- Regression: A regression problem is when the output variable is a real value, such as “dollars” or “weight.” The contrived example above is a regression problem.
