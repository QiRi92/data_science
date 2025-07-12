# A Gentle Introduction to XGBoost for Applied Machine Learning

XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

In this post you will discover XGBoost and get a gentle introduction to what is, where it came from and how you can learn more.

After reading this post you will know:

- What XGBoost is and the goals of the project.
- Why XGBoost must be a part of your machine learning toolkit.
- Where you can learn more to start using XGBoost on your next machine learning project.

## What is XGBoost?

XGBoost stands for eXtreme Gradient Boosting.

It is an implementation of gradient boosting machines created by Tianqi Chen, now with contributions from many developers. It belongs to a broader collection of tools under the umbrella of the Distributed Machine Learning Community or <a href="https://github.com/dmlc">DMLC</a> who are also the creators of the popular <a href="https://github.com/dmlc/mxnet">mxnet deep learning library</a>.

Tianqi Chen provides a brief and interesting back story on the creation of XGBoost in the post Story and Lessons Behind the Evolution of XGBoost.

XGBoost is a software library that you can download and install on your machine, then access from a variety of interfaces. Specifically, XGBoost supports the following main interfaces:

- Command Line Interface (CLI).
- C++ (the language in which the library is written).
- Python interface as well as a model in scikit-learn.
- R interface as well as a model in the caret package.
- Julia.
- Java and JVM languages like Scala and platforms like Hadoop.

## XGBoost Features

The library is laser focused on computational speed and model performance, as such there are few frills. Nevertheless, it does offer a number of advanced features.

### Model Features

The implementation of the model supports the features of the scikit-learn and R implementations, with new additions like regularization. Three main forms of gradient boosting are supported:

- **Gradient Boosting** algorithm also called gradient boosting machine including the learning rate.
- **Stochastic Gradient** Boosting with sub-sampling at the row, column and column per split levels.
- **Regularized Gradient Boosting** with both L1 and L2 regularization.

### System Features

The library provides a system for use in a range of computing environments, not least:

- **Parallelization** of tree construction using all of your CPU cores during training.
- **Distributed Computing** for training very large models using a cluster of machines.
- **Out-of-Core Computing** for very large datasets that don’t fit into memory.
- **Cache Optimization** of data structures and algorithm to make best use of hardware.

### Algorithm Features

The implementation of the algorithm was engineered for efficiency of compute time and memory resources. A design goal was to make the best use of available resources to train the model. Some key algorithm implementation features include:

- **Sparse Aware** implementation with automatic handling of missing data values.
- **Block Structure** to support the parallelization of tree construction.
- **Continued Training** so that you can further boost an already fitted model on new data.

XGBoost is free open source software available for use under the permissive Apache-2 license.

## Why Use XGBoost?

The two reasons to use XGBoost are also the two goals of the project:

1. Execution Speed.
2. Model Performance.

### 1. XGBoost Execution Speed

Generally, XGBoost is fast. Really fast when compared to other implementations of gradient boosting.

<a href="https://www.linkedin.com/in/szilard">Szilard Pafka</a> performed some objective benchmarks comparing the performance of XGBoost to other implementations of gradient boosting and bagged decision trees. He wrote up his results in May 2015 in the blog post titled “Benchmarking Random Forest Implementations“.

He also provides <a href="https://github.com/szilard/benchm-ml">all the code on GitHub</a> and a more extensive report of results with hard numbers.

<img width="403" height="278" alt="image" src="https://github.com/user-attachments/assets/e5d3c5c0-03d7-4b7a-a422-1fa35542330f" />

His results showed that XGBoost was almost always faster than the other benchmarked implementations from R, Python Spark and H2O.

### 2. XGBoost Model Performance

XGBoost dominates structured or tabular datasets on classification and regression predictive modeling problems.

The evidence is that it is the go-to algorithm for competition winners on the Kaggle competitive data science platform.

For example, there is an incomplete list of first, second and third place competition winners that used titled: <a href="https://github.com/dmlc/xgboost/tree/master/demo#machine-learning-challenge-winning-solutions">XGBoost: Machine Learning Challenge Winning Solutions</a>.

## What Algorithm Does XGBoost Use?

The XGBoost library implements the <a href="https://en.wikipedia.org/wiki/Gradient_boosting">gradient boosting decision tree algorithm</a>.

This algorithm goes by lots of different names such as gradient boosting, multiple additive regression trees, stochastic gradient boosting or gradient boosting machines.

Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. A popular example is the <a href="https://github.com/QiRi92/data_science/blob/main/XGBoost/1_1_adaboost.md">AdaBoost algorithm</a> that weights data points that are hard to predict.

Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

This approach supports both regression and classification predictive modeling problems.

For more on boosting and gradient boosting, see Trevor Hastie’s talk on <a href="https://www.youtube.com/watch?v=wPqtzj5VZus">Gradient Boosting Machine Learning</a>.

## Official XGBoost Resources

The best source of information on XGBoost is the <a href="https://github.com/dmlc/xgboost">official GitHub repository for the project</a>.

From there you can get access to the <a href="https://github.com/dmlc/xgboost/issues">Issue Tracker</a> and the User Group that can be used for asking questions and reporting bugs.

A great source of links with example code and help is the <a href="https://github.com/dmlc/xgboost/tree/master/demo">Awesome XGBoost page</a>.

There is also an <a href="https://xgboost.readthedocs.io/en/latest/">official documentation page</a> that includes a getting started guide for a range of different languages, tutorials, how-to guides and more.

## Installing XGBoost

There is a comprehensive installation guide on the <a href="http://xgboost.readthedocs.io/en/latest/build.html">XGBoost documentation website</a>.

It covers installation for Linux, Mac OS X and Windows.

It also covers installation on platforms such as R and Python.

### XGBoost in Python

Installation instructions are available on the Python section of the XGBoost installation guide.

The official <a href="http://xgboost.readthedocs.io/en/latest/python/python_intro.html">Python Package Introduction</a> is the best place to start when working with XGBoost in Python.

To get started quickly, you can type:

```
sudo pip install xgboost
```

There is also an excellent list of sample source code in Python on the <a href="https://github.com/tqchen/xgboost/tree/master/demo/guide-python">XGBoost Python Feature Walkthrough</a>.

## Summary

In this post you discovered the XGBoost algorithm for applied machine learning.

You learned:

- That XGBoost is a library for developing fast and high performance gradient boosting tree models.
- That XGBoost is achieving the best performance on a range of difficult machine learning tasks.
- That you can use this library from the command line, Python and R and how to get started.
