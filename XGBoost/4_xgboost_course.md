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

A benefit of using ensembles of decision tree methods like gradient boosting is that they can automatically provide estimates of feature importance from a trained predictive model.

A trained XGBoost model automatically calculates feature importance on your predictive modeling problem.

These importance scores are available in the **feature_importances_** member variable of the trained model. For example, they can be printed directly as follows:

```python
print(model.feature_importances_)
```

The XGBoost library provides a built-in function to plot features ordered by their importance.

The function is called **plot_importance()** and can be used as follows:

```python
plot_importance(model)
pyplot.show()
```

These importance scores can help you decide what input variables to keep or discard. They can also be used as the basis for automatic feature selection techniques.

The full example of plotting feature importance scores using the Pima Indians Onset of Diabetes dataset is provided below.

```python
# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()
```

Output:

<img width="448" height="331" alt="image" src="https://github.com/user-attachments/assets/052f0e40-4414-4f7c-84da-20237c19f3c1" />

In the next lesson we will look at heuristics for best configuring the gradient boosting algorithm.

## Lesson 06: How to Configure Gradient Boosting

Gradient boosting is one of the most powerful techniques for applied machine learning and as such is quickly becoming one of the most popular.

But how do you configure gradient boosting on your problem?

A number of configuration heuristics were published in the original gradient boosting papers. They can be summarized as:

- Learning rate or shrinkage (**learning_rate** in XGBoost) should be set to 0.1 or lower, and smaller values will require the addition of more trees.
- The depth of trees (**max_depth** in XGBoost) should be configured in the range of 2-to-8, where not much benefit is seen with deeper trees.
- Row sampling (**subsample** in XGBoost) should be configured in the range of 30% to 80% of the training dataset, and compared to a value of 100% for no sampling.

These are a good starting points when configuring your model.

A good general configuration strategy is as follows:

1. Run the default configuration and review plots of the learning curves on the training and validation datasets.
2. If the system is overlearning, decrease the learning rate and/or increase the number of trees.
3. If the system is underlearning, speed the learning up to be more aggressive by increasing the learning rate and/or decreasing the number of trees.

<a href="https://goo.gl/OqIRIc">Owen Zhang</a>, the former #1 ranked competitor on Kaggle and now CTO at Data Robot proposes an interesting strategy to configure XGBoost.

He suggests to set the number of trees to a target value such as 100 or 1000, then tune the learning rate to find the best model. This is an efficient strategy for quickly finding a good model.

In the next and final lesson, we will look at an example of tuning the XGBoost hyperparameters.

## Lesson 07: XGBoost Hyperparameter Tuning

The scikit-learn framework provides the capability to search combinations of parameters.

This capability is provided in the **GridSearchCV** class and can be used to discover the best way to configure the model for top performance on your problem.

For example, we can define a grid of the number of trees (**n_estimators**) and tree sizes (**max_depth**) to evaluate by defining a grid as:

```python
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
```

And then evaluate each combination of parameters using 10-fold cross validation as:

```python
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
result = grid_search.fit(X, label_encoded_y)
```

We can then review the results to determine the best combination and the general trends in varying the combinations of parameters.

This is the best practice when applying XGBoost to your own problems. The parameters to consider tuning are:

- The number and size of trees (**n_estimators** and **max_depth**).
- The learning rate and number of trees (**learning_rate** and **n_estimators**).
- The row and column subsampling rates (**subsample**, **colsample_bytree** and **colsample_bylevel**).

Below is a full example of tuning just the **learning_rate** on the Pima Indians Onset of Diabetes dataset.

```python
# Tune learning_rate
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]
# grid search
model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
```

Output:

```
Best: -0.517018 using {'learning_rate': 0.01}
-0.643877 (0.002168) with: {'learning_rate': 0.0001}
-0.620317 (0.006926) with: {'learning_rate': 0.001}
-0.517018 (0.036275) with: {'learning_rate': 0.01}
-0.557111 (0.115078) with: {'learning_rate': 0.1}
-0.662438 (0.149560) with: {'learning_rate': 0.2}
-0.750694 (0.199802) with: {'learning_rate': 0.3}
```

## XGBoost Learning Mini-Course Review

Take a moment and look back at how far you have come:

- You learned about the gradient boosting algorithm and the XGBoost library.
- You developed your first XGBoost model.
- You learned how to use advanced features like early stopping and feature importance.
- You learned how to configure gradient boosted models and how to design controlled experiments to tune XGBoost hyperparameters.

## Reference

- <a href="https://github.com/QiRi92/data_science/blob/main/XGBoost/4_xgboost_course.ipynb" rel="noopener" target="_blank">Codes</a>
