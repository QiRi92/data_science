# A Gentle Introduction to Imbalanced Classification

Classification predictive modeling involves predicting a class label for a given observation.

An imbalanced classification problem is an example of a classification problem where the distribution of examples across the known classes is biased or skewed. The distribution can vary from a slight bias to a severe imbalance where there is one example in the minority class for hundreds, thousands, or millions of examples in the majority class or classes.

Imbalanced classifications pose a challenge for predictive modeling as most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class. This results in models that have poor predictive performance, specifically for the minority class. This is a problem because typically, the minority class is more important and therefore the problem is more sensitive to classification errors for the minority class than the majority class.

In this tutorial, you will discover imbalanced classification predictive modeling.

After completing this tutorial, you will know:

- Imbalanced classification is the problem of classification when there is an unequal distribution of classes in the training dataset.
- The imbalance in the class distribution may vary, but a severe imbalance is more challenging to model and may require specialized techniques.
- Many real-world classification problems have an imbalanced class distribution, such as fraud detection, spam detection, and churn prediction.

## Tutorial Overview

This tutorial is divided into five parts; they are:

1. Classification Predictive Modeling
2. Imbalanced Classification Problems
3. Causes of Class Imbalance
4. Challenge of Imbalanced Classification
5. Examples of Imbalanced Classification

## Classification Predictive Modeling

Classification is a predictive modeling problem that involves assigning a class label to each observation.

Each example is comprised of both the observations and a class label.

- **Example**: An observation from the domain (*input*) and an associated class label (*output*).

For example, we may collect measurements of a flower and classify the species of flower (label) from the measurements. The number of classes for a predictive modeling problem is typically fixed when the problem is framed or described, and typically, the number of classes does not change.

We may alternately choose to predict a probability of class membership instead of a crisp class label.

This allows a predictive model to share uncertainty in a prediction across a range of options and allow the user to interpret the result in the context of the problem.

For example, given measurements of a flower (*observation*), we may predict the likelihood (*probability*) of the flower being an example of each of twenty different species of flower.

The number of classes for a predictive modeling problem is typically fixed when the problem is framed or described, and usually, the number of classes does not change.

A classification predictive modeling problem may have two class labels. This is the simplest type of classification problem and is referred to as two-class classification or binary classification. Alternately, the problem may have more than two classes, such as three, 10, or even hundreds of classes. These types of problems are referred to as multi-class classification problems.

- **Binary Classification Problem**: A classification predictive modeling problem where all examples belong to one of two classes.
- **Multiclass Classification Problem**: A classification predictive modeling problem where all examples belong to one of three classes.

When working on classification predictive modeling problems, we must collect a training dataset.

A training dataset is a number of examples from the domain that include both the input data (e.g. measurements) and the output data (e.g. class label).

- **Training Dataset**: A number of examples collected from the problem domain that include the input observations and output class labels.

Depending on the complexity of the problem and the types of models we may choose to use, we may need tens, hundreds, thousands, or even millions of examples from the domain to constitute a training dataset.

The training dataset is used to better understand the input data to help best prepare it for modeling. It is also used to evaluate a suite of different modeling algorithms. It is used to tune the hyperparameters of a chosen model. And finally, the training dataset is used to train a final model on all available data that we can use in the future to make predictions for new examples from the problem domain.

Now that we are familiar with classification predictive modeling, let’s consider an imbalance of classes in the training dataset.

## Imbalanced Classification Problems

The number of examples that belong to each class may be referred to as the class distribution.

Imbalanced classification refers to a classification predictive modeling problem where the number of examples in the training dataset for each class label is not balanced.

That is, where the class distribution is not equal or close to equal, and is instead biased or skewed.

- **Imbalanced Classification**: A classification predictive modeling problem where the distribution of examples across the classes is not equal.

For example, we may collect measurements of flowers and have 80 examples of one flower species and 20 examples of a second flower species, and only these examples comprise our training dataset. This represents an example of an imbalanced classification problem.

We refer to these types of problems as “*imbalanced classification*” instead of “*unbalanced classification*“. Unbalance refers to a class distribution that was balanced and is now no longer balanced, whereas imbalanced refers to a class distribution that is inherently not balanced.

There are other less general names that may be used to describe these types of classification problems, such as:

- Rare event prediction.
- Extreme event prediction.
- Severe class imbalance.

The imbalance of a problem is defined by the distribution of classes in a specific training dataset.

It is common to describe the imbalance of classes in a dataset in terms of a ratio.

For example, an imbalanced binary classification problem with an imbalance of 1 to 100 (1:100) means that for every one example in one class, there are 100 examples in the other class.

Another way to describe the imbalance of classes in a dataset is to summarize the class distribution as percentages of the training dataset. For example, an imbalanced multiclass classification problem may have 80 percent examples in the first class, 18 percent in the second class, and 2 percent in a third class.

Now that we are familiar with the definition of an imbalanced classification problem, let’s look at some possible reasons as to why the classes may be imbalanced.

## Causes of Class Imbalance

The imbalance to the class distribution in an imbalanced classification predictive modeling problem may have many causes.

There are perhaps two main groups of causes for the imbalance we may want to consider; they are data sampling and properties of the domain.

It is possible that the imbalance in the examples across the classes was caused by the way the examples were collected or sampled from the problem domain. This might involve biases introduced during data collection, and errors made during data collection.

- Biased Sampling.
- Measurement Errors.

For example, perhaps examples were collected from a narrow geographical region, or slice of time, and the distribution of classes may be quite different or perhaps even collected in a different way.

Errors may have been made when collecting the observations. One type of error might have been applying the wrong class labels to many examples. Alternately, the processes or systems from which examples were collected may have been damaged or impaired to cause the imbalance.

Often in cases where the imbalance is caused by a sampling bias or measurement error, the imbalance can be corrected by improved sampling methods, and/or correcting the measurement error. This is because the training dataset is not a fair representation of the problem domain that is being addressed.

The imbalance might be a property of the problem domain.

For example, the natural occurrence or presence of one class may dominate other classes. This may be because the process that generates observations in one class is more expensive in time, cost, computation, or other resources. As such, it is often infeasible or intractable to simply collect more samples from the domain in order to improve the class distribution. Instead, a model is required to learn the difference between the classes.

Now that we are familiar with the possible causes of a class imbalance, let’s consider why imbalanced classification problems are challenging.

## Challenge of Imbalanced Classification

The imbalance of the class distribution will vary across problems.

A classification problem may be a little skewed, such as if there is a slight imbalance. Alternately, the classification problem may have a severe imbalance where there might be hundreds or thousands of examples in one class and tens of examples in another class for a given training dataset.

- **Slight Imbalance**. An imbalanced classification problem where the distribution of examples is uneven by a small amount in the training dataset (e.g. 4:6).
- **Severe Imbalance**. An imbalanced classification problem where the distribution of examples is uneven by a large amount in the training dataset (e.g. 1:100 or more).

A slight imbalance is often not a concern, and the problem can often be treated like a normal classification predictive modeling problem. A severe imbalance of the classes can be challenging to model and may require the use of specialized techniques.

The class or classes with abundant examples are called the major or majority classes, whereas the class with few examples (and there is typically just one) is called the minor or minority class.

- **Majority Class**: The class (or classes) in an imbalanced classification predictive modeling problem that has many examples.
- **Minority Class**: The class in an imbalanced classification predictive modeling problem that has few examples.

When working with an imbalanced classification problem, the minority class is typically of the most interest. This means that a model’s skill in correctly predicting the class label or probability for the minority class is more important than the majority class or classes.

The minority class is harder to predict because there are few examples of this class, by definition. This means it is more challenging for a model to learn the characteristics of examples from this class, and to differentiate examples from this class from the majority class (or classes).

The abundance of examples from the majority class (or classes) can swamp the minority class. Most machine learning algorithms for classification predictive models are designed and demonstrated on problems that assume an equal distribution of classes. This means that a naive application of a model may focus on learning the characteristics of the abundant observations only, neglecting the examples from the minority class that is, in fact, of more interest and whose predictions are more valuable.

Imbalanced classification is not “*solved*.”

It remains an open problem generally, and practically must be identified and addressed specifically for each training dataset.

This is true even in the face of more data, so-called “*big data*,” large neural network models, so-called “*deep learning*,” and very impressive competition-winning models, so-called “*xgboost*.”

Now that we are familiar with the challenge of imbalanced classification, let’s look at some common examples.

## Examples of Imbalanced Classification

Many of the classification predictive modeling problems that we are interested in solving in practice are imbalanced.

As such, it is surprising that imbalanced classification does not get more attention than it does.

Below is a list of ten examples of problem domains where the class distribution of examples is inherently imbalanced.

Many classification problems may have a severe imbalance in the class distribution; nevertheless, looking at common problem domains that are inherently imbalanced will make the ideas and challenges of class imbalance concrete.

- Fraud Detection.
- Claim Prediction
- Default Prediction.
- Churn Prediction.
- Spam Detection.
- Anomaly Detection.
- Outlier Detection.
- Intrusion Detection
- Conversion Prediction.

The list of examples sheds light on the nature of imbalanced classification predictive modeling.

Each of these problem domains represents an entire field of study, where specific problems from each domain can be framed and explored as imbalanced classification predictive modeling. This highlights the multidisciplinary nature of class imbalanced classification, and why it is so important for a machine learning practitioner to be aware of the problem and skilled in addressing it.

Notice that most, if not all, of the examples are likely binary classification problems. Notice too that examples from the minority class are rare, extreme, abnormal, or unusual in some way.

Also notice that many of the domains are described as “detection,” highlighting the desire to discover the minority class amongst the abundant examples of the majority class.

We now have a robust overview of imbalanced classification predictive modeling.

## Further Reading

This section provides more resources on the topic if you are looking to go deeper.

### Tutorials

- <a href="https://github.com/QiRi92/data_science/blob/main/imbalance_class/1_1_way_handle_imbalance.md" rel="noopener" target="_blank">8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset</a>
- <a href="" rel="noopener" target="_blank">How to Develop and Evaluate Naive Classifier Strategies Using Probability</a>

## Summary

In this tutorial, you discovered imbalanced classification predictive modeling.

Specifically, you learned:

- Imbalanced classification is the problem of classification when there is an unequal distribution of classes in the training dataset.
- The imbalance in the class distribution may vary, but a severe imbalance is more challenging to model and may require specialized techniques.
- Many real-world classification problems have an imbalanced class distribution such as fraud detection, spam detection, and churn prediction.
