# Data Preparation for Machine Learning

Data preparation involves transforming raw data into a form that is more appropriate for modeling.

Preparing data may be the most important part of a predictive modeling project and the most time-consuming, although it seems to be the least discussed. Instead, the focus is on machine learning algorithms, whose usage and parameterization has become quite routine.

Practical data preparation requires knowledge of data cleaning, feature selection data transforms, dimensionality reduction, and more.

In this crash course, you will discover how you can get started and confidently prepare data for a predictive modeling project with Python in seven days.

## Crash-Course Overview

This crash course is broken down into seven lessons.

You could complete one lesson per day (recommended) or complete all of the lessons in one day (hardcore). It really depends on the time you have available and your level of enthusiasm.

Below is a list of the seven lessons that will get you started and productive with data preparation in Python:

- **Lesson 01**: Importance of Data Preparation
- **Lesson 02**: Fill Missing Values With Imputation
- **Lesson 03**: Select Features With RFE
- **Lesson 04**: Scale Data With Normalization
- **Lesson 05**: Transform Categories With One-Hot Encoding
- **Lesson 06**: Transform Numbers to Categories With kBins
- **Lesson 07**: Dimensionality Reduction with PCA

## Lesson 01: Importance of Data Preparation

In this lesson, you will discover the importance of data preparation in predictive modeling with machine learning.

Predictive modeling projects involve learning from data.

Data refers to examples or cases from the domain that characterize the problem you want to solve.

On a predictive modeling project, such as classification or regression, raw data typically cannot be used directly.

There are four main reasons why this is the case:

- **Data Types**: Machine learning algorithms require data to be numbers.
- **Data Requirements**: Some machine learning algorithms impose requirements on the data.
- **Data Errors**: Statistical noise and errors in the data may need to be corrected.
- **Data Complexity**: Complex nonlinear relationships may be teased out of the data.

The raw data must be pre-processed prior to being used to fit and evaluate a machine learning model. This step in a predictive modeling project is referred to as “*data preparation*.”

There are common or standard tasks that you may use or explore during the data preparation step in a machine learning project.

These tasks include:

- **Data Cleaning**: Identifying and correcting mistakes or errors in the data.
- **Feature Selection**: Identifying those input variables that are most relevant to the task.
- **Data Transforms**: Changing the scale or distribution of variables.
- **Feature Engineering**: Deriving new variables from available data.
- **Dimensionality Reduction**: Creating compact projections of the data.

Each of these tasks is a whole field of study with specialized algorithms.

## Lesson 02: Fill Missing Values With Imputation

In this lesson, you will discover how to identify and fill missing values in data.

Real-world data often has <a href="">missing values</a>.

Data can have missing values for a number of reasons, such as observations that were not recorded and data corruption. Handling missing data is important as many machine learning algorithms do not support data with missing values.

Filling missing values with data is called data imputation and a popular approach for data imputation is to calculate a statistical value for each column (such as a mean) and replace all missing values for that column with the statistic.

The <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv">horse colic dataset</a> describes medical characteristics of horses with colic and whether they lived or died. It has missing values marked with a question mark ‘?’. We can load the dataset with the <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html">read_csv() function</a> and ensure that question mark values are marked as NaN.

Once loaded, we can use the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html">SimpleImputer</a> class to transform all missing values marked with a NaN value with the mean of the column.

The complete example is listed below.

```python
# statistical imputation transform for the horse colic dataset
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# print total missing
print('Missing: %d' % sum(isnan(X).flatten()))
# define imputer
imputer = SimpleImputer(strategy='mean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
# print total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))
```

Output:

```
Missing: 1605
Missing: 0
```

## Lesson 03: Select Features With RFE

In this lesson, you will discover how to select the most important features in a dataset.

Feature selection is the process of reducing the number of input variables when developing a predictive model.

It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

Recursive Feature Elimination, or RFE for short, is a popular feature selection algorithm.

RFE is popular because it is easy to configure and use and because it is effective at selecting those features (columns) in a training dataset that are more or most relevant in predicting the target variable.

The scikit-learn Python machine learning library provides an implementation of <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html">RFE</a> for machine learning. RFE is a transform. To use it, first, the class is configured with the chosen algorithm specified via the “estimator” argument and the number of features to select via the “*n_features_to_select*” argument.

The example below defines a synthetic classification dataset with five redundant input features. RFE is then used to select five features using the decision tree algorithm.

```python
# report which features were selected by RFE
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
# fit RFE
rfe.fit(X, y)
# summarize all features
for i in range(X.shape[1]):
	print('Column: %d, Selected=%s, Rank: %d' % (i, rfe.support_[i], rfe.ranking_[i]))
```

Output:

```
Column: 0, Selected=False, Rank: 5
Column: 1, Selected=False, Rank: 4
Column: 2, Selected=True, Rank: 1
Column: 3, Selected=True, Rank: 1
Column: 4, Selected=True, Rank: 1
Column: 5, Selected=False, Rank: 6
Column: 6, Selected=True, Rank: 1
Column: 7, Selected=False, Rank: 3
Column: 8, Selected=True, Rank: 1
Column: 9, Selected=False, Rank: 2
```

## Lesson 04: Scale Data With Normalization

In this lesson, you will discover how to scale numerical data for machine learning.

Many machine learning algorithms perform better when numerical input variables are scaled to a standard range.

This includes algorithms that use a weighted sum of the input, like linear regression, and algorithms that use distance measures, like k-nearest neighbors.

One of the most popular techniques for scaling numerical data prior to modeling is normalization. Normalization scales each input variable separately to the range 0-1, which is the range for floating-point values where we have the most precision. It requires that you know or are able to accurately estimate the minimum and maximum observable values for each variable. You may be able to estimate these values from your available data.

You can normalize your dataset using the scikit-learn object <a href="http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html">MinMaxScaler</a>.

The example below defines a synthetic classification dataset, then uses the MinMaxScaler to normalize the input variables.

```python
# example of normalizing input data
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)
# summarize data before the transform
print(X[:3, :])
# define the scaler
trans = MinMaxScaler()
# transform the data
X_norm = trans.fit_transform(X)
# summarize data after the transform
print(X_norm[:3, :])
```

Output:

```
[[ 2.39324489 -5.77732048 -0.59062319 -2.08095322  1.04707034]
 [-0.45820294  1.94683482 -2.46471441  2.36590955 -0.73666725]
 [ 2.35162422 -1.00061698 -0.5946091   1.12531096 -0.65267587]]
[[0.77608466 0.0239289  0.48251588 0.18352101 0.59830036]
 [0.40400165 0.79590304 0.27369632 0.6331332  0.42104156]
 [0.77065362 0.50132629 0.48207176 0.5076991  0.4293882 ]]
```

