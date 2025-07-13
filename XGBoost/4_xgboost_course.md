# 7 Step Mini-Course to Get Started with XGBoost in Python

XGBoost is an implementation of gradient boosting that is being used to win machine learning competitions.

It is powerful but it can be hard to get started.

In this post, you will discover a 7-part crash course on XGBoost with Python.

This mini-course is designed for Python machine learning practitioners that are already comfortable with scikit-learn and the SciPy ecosystem.

## Who Is This Mini-Course For?

Before we get started, let’s make sure you are in the right place. The list below provides some general guidelines as to who this course was designed for.

Don’t panic if you don’t match these points exactly, you might just need to brush up in one area or another to keep up.

- **Developers that know how to write a little code**. This means that it is not a big deal for you to get things done with Python and know how to setup the SciPy ecosystem on your workstation (a prerequisite). It does not mean your a wizard coder, but it does mean you’re not afraid to install packages and write scripts.
- **Developers that know a little machine learning**. This means you know about the basics of machine learning like cross validation, some algorithms and the bias-variance trade-off. It does not mean that you are a machine learning PhD, just that you know the landmarks or know where to look them up.

This mini-course is not a textbook on XGBoost. There will be no equations.

It will take you from a developer that knows a little machine learning in Python to a developer who can get results and bring the power of XGBoost to your own projects.

## Mini-Course Overview (what to expect)

This mini-course is divided into 7 parts.

Each lesson was designed to take the average developer about 30 minutes. You might finish some much sooner and others you may choose to go deeper and spend more time.

You can complete each part as quickly or as slowly as you like. A comfortable schedule may be to complete one lesson per day over a one week period. Highly recommended.

The topics you will cover over the next 7 lessons are as follows:

- **Lesson 01**: Introduction to Gradient Boosting.
- **Lesson 02**: Introduction to XGBoost.
- **Lesson 03**: Develop Your First XGBoost Model.
- **Lesson 04**: Monitor Performance and Early Stopping.
- **Lesson 05**: Feature Importance with XGBoost.
- **Lesson 06**: How to Configure Gradient Boosting.
- **Lesson 07**: XGBoost Hyperparameter Tuning.

## Lesson 01: Introduction to Gradient Boosting

Gradient boosting is one of the most powerful techniques for building predictive models.

The idea of boosting came out of the idea of whether a weak learner can be modified to become better. The first realization of boosting that saw great success in application was Adaptive Boosting or AdaBoost for short. The weak learners in AdaBoost are decision trees with a single split, called decision stumps for their shortness.

AdaBoost and related algorithms were recast in a statistical framework and became known as Gradient Boosting Machines. The statistical framework cast boosting as a numerical optimization problem where the objective is to minimize the loss of the model by adding weak learners using a gradient descent like procedure, hence the name.

The Gradient Boosting algorithm involves three elements:

1. **A loss function to be optimized**, such as cross entropy for classification or mean squared error for regression problems.
2. **A weak learner to make predictions**, such as a greedily constructed decision tree.
3. **An additive model**, used to add weak learners to minimize the loss function.

New weak learners are added to the model in an effort to correct the residual errors of all previous trees. The result is a powerful predictive modeling algorithm, perhaps more powerful than random forest.

In the next lesson we will take a closer look at the XGBoost implementation of gradient boosting.

## Lesson 02: Introduction to XGBoost

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

XGBoost stands for e**X**treme **G**radient **Boost**ing.

It was developed by Tianqi Chen and is laser focused on computational speed and model performance, as such there are few frills.

In addition to supporting all key variations of the technique, the real interest is the speed provided by the careful engineering of the implementation, including:

- **Parallelization** of tree construction using all of your CPU cores during training.
- **Distributed Computing** for training very large models using a cluster of machines.
- **Out-of-Core Computing** for very large datasets that don’t fit into memory.
- **Cache Optimization** of data structures and algorithms to make best use of hardware.

Traditionally, gradient boosting implementations are slow because of the sequential nature in which each tree must be constructed and added to the model.

The on performance in the development of XGBoost has resulted in one of the best predictive modeling algorithms that can now harness the full capability of your hardware platform, or very large computers you might rent in the cloud.

As such, XGBoost has been a cornerstone in competitive machine learning, being the technique used to win and recommended by winners. For example, here is what some recent Kaggle competition winners have said:

In the next lesson, we will develop our first XGBoost model in Python.

## Lesson 03: Develop Your First XGBoost Model

Assuming you have a working SciPy environment, XGBoost can be installed easily using pip.

For example:

```python
sudo pip install xgboost
```

You can learn more about installing and building XGBoost on your platform in the <a href="http://xgboost.readthedocs.io/en/latest/build.html">XGBoost Installation Instructions</a>.

XGBoost models can be used directly in the scikit-learn framework using the wrapper classes, **XGBClassifier** for classification and **XGBRegressor** for regression problems.

This is the recommended way to use XGBoost in Python.

Download the Pima Indians onset of diabetes dataset.

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv">Dataset File</a>
- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names">Dataset Details</a>

It is a good test dataset for binary classification as all input variables are numeric, meaning the problem can be modeled directly with no data preparation.

We can train an XGBoost model for classification by constructing it and calling the **model.fit()** function:

```python
model = XGBClassifier()
model.fit(X_train, y_train)
```

This model can then be used to make predictions by calling the **model.predict()** function on new data.

```python
y_pred = model.predict(X_test)
```

We can tie this all together as follows:

```python
# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

In the next lesson we will look at how we can use early stopping to limit overfitting.

## Lesson 04: Monitor Performance and Early Stopping

The XGBoost model can evaluate and report on the performance on a test set for the model during training.

It supports this capability by specifying both a test dataset and an evaluation metric on the call to **model.fit()** when training the model and specifying verbose output (**verbose=True**).

For example, we can report on the binary classification error rate (**error**) on a standalone test set (**eval_set**) while training an XGBoost model as follows:

```python
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
```

Running a model with this configuration will report the performance of the model after each tree is added. For example:

```python
...
[89] validation_0-error:0.204724
[90] validation_0-error:0.208661
```

We can use this evaluation to stop training once no further improvements have been made to the model.

We can do this by setting the **early_stopping_rounds** parameter when calling **model.fit()** to the number of iterations that no improvement is seen on the validation dataset before training is stopped.

The full example using the Pima Indians Onset of Diabetes dataset is provided below.

```python
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Split into training and testing
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=7)

# Further split training into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    verbosity=1,
    random_state=42,
    early_stopping_rounds=10,   # ✅ Set here
    eval_metric="logloss"       # ✅ Set here
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Make predictions
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

In the next lesson, we will look at how we calculate the importance of features using XGBoost

## Lesson 05: Feature Importance with XGBoost
