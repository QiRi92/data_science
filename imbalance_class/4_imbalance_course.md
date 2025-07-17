# Imbalanced Classification With Python (7-Day Mini-Course)

Classification predictive modeling is the task of assigning a label to an example.

Imbalanced classification are those classification tasks where the distribution of examples across the classes is not equal.

Practical imbalanced classification requires the use of a suite of specialized techniques, data preparation techniques, learning algorithms, and performance metrics.

In this crash course, you will discover how you can get started and confidently work through an imbalanced classification project with Python in seven days.

## Crash-Course Overview

This crash course is broken down into seven lessons.

You could complete one lesson per day (*recommended*) or complete all of the lessons in one day (*hardcore*). It really depends on the time you have available and your level of enthusiasm.

Below is a list of the seven lessons that will get you started and productive with imbalanced classification in Python:

**- Lesson 01**: Challenge of Imbalanced Classification
**- Lesson 02**: Intuition for Imbalanced Data
**- Lesson 03**: Evaluate Imbalanced Classification Models
**- Lesson 04**: Undersampling the Majority Class
**- Lesson 05**: Oversampling the Minority Class
**- Lesson 06**: Combine Data Undersampling and Oversampling
**- Lesson 07**: Cost-Sensitive Algorithms

## Lesson 01: Challenge of Imbalanced Classification

In this lesson, you will discover the challenge of imbalanced classification problems.

Imbalanced classification problems pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class.

This results in models that have poor predictive performance, specifically for the minority class. This is a problem because typically, the minority class is more important and therefore the problem is more sensitive to classification errors for the minority class than the majority class.

**- Majority Class**: More than half of the examples belong to this class, often the negative or normal case.
**- Minority Class**: Less than half of the examples belong to this class, often the positive or abnormal case.

A classification problem may be a little skewed, such as if there is a slight imbalance. Alternately, the classification problem may have a severe imbalance where there might be hundreds or thousands of examples in one class and tens of examples in another class for a given training dataset.

**- Slight Imbalance**. Where the distribution of examples is uneven by a small amount in the training dataset (e.g. 4:6).
**- Severe Imbalance**. Where the distribution of examples is uneven by a large amount in the training dataset (e.g. 1:100 or more).

Many of the classification predictive modeling problems that we are interested in solving in practice are imbalanced.

As such, it is surprising that imbalanced classification does not get more attention than it does.

## Lesson 02: Intuition for Imbalanced Data

In this lesson, you will discover how to develop a practical intuition for imbalanced classification datasets.

A challenge for beginners working with imbalanced classification problems is what a specific skewed class distribution means. For example, what is the difference and implication for a 1:10 vs. a 1:100 class ratio?

The <a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html">make_classification()</a> scikit-learn function can be used to define a synthetic dataset with a desired class imbalance. The “*weights*” argument specifies the ratio of examples in the negative class, e.g. [0.99, 0.01] means that 99 percent of the examples will belong to the majority class, and the remaining 1 percent will belong to the minority class.

```python
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0)
```

Once defined, we can summarize the class distribution using a Counter object to get an idea of exactly how many examples belong to each class.

```python
# summarize class distribution
counter = Counter(y)
print(counter)
```

We can also create a scatter plot of the dataset because there are only two input variables. The dots can then be colored by each class. This plot provides a visual intuition for what exactly a 99 percent vs. 1 percent majority/minority class imbalance looks like in practice.

The complete example of creating and summarizing an imbalanced classification dataset is listed below.

```python

```
