# Boosting and AdaBoost for Machine Learning

Boosting is an ensemble technique that attempts to create a strong classifier from a number of weak classifiers.

In this post you will discover the AdaBoost Ensemble method for machine learning. After reading this post, you will know:

- What the boosting ensemble method is and generally how it works.
- How to learn to boost decision trees using the AdaBoost algorithm.
- How to make predictions using the learned AdaBoost model.
- How to best prepare your data for use with the AdaBoost algorithm

## Boosting Ensemble Method

<a href="https://en.wikipedia.org/wiki/Boosting_(machine_learning)">Boosting</a> is a general ensemble method that creates a strong classifier from a number of weak classifiers.

This is done by building a model from the training data, then creating a second model that attempts to correct the errors from the first model. Models are added until the training set is predicted perfectly or a maximum number of models are added.

<a href="https://en.wikipedia.org/wiki/AdaBoost">AdaBoost</a> was the first really successful boosting algorithm developed for binary classification. It is the best starting point for understanding boosting.

Modern boosting methods build on AdaBoost, most notably <a href="https://en.wikipedia.org/wiki/Gradient_boosting">stochastic gradient boosting machines</a>.

## Learning An AdaBoost Model From Data

AdaBoost is best used to boost the performance of decision trees on binary classification problems.

AdaBoost was originally called AdaBoost.M1 by the authors of the technique Freund and Schapire. More recently it may be referred to as discrete AdaBoost because it is used for classification rather than regression.

AdaBoost can be used to boost the performance of any machine learning algorithm. It is best used with weak learners. These are models that achieve accuracy just above random chance on a classification problem.

The most suited and therefore most common algorithm used with AdaBoost are decision trees with one level. Because these trees are so short and only contain one decision for classification, they are often called decision stumps.

Each instance in the training dataset is weighted. The initial weight is set to:

                                                      weight(xi) = 1/n

Where xi is the i’th training instance and n is the number of training instances.

## How To Train One Model

