# Applied Deep Learning in Python Mini-Course

Deep learning is a fascinating field of study and the techniques are achieving world class results in a range of challenging machine learning problems.

It can be hard to get started in deep learning.

Which library should you use and which techniques should you focus on?

In this post you will discover a 14-part crash course into deep learning in Python with the easy to use and powerful Keras library.

This mini-course is intended for python machine learning practitioners that are already comfortable with scikit-learn on the SciPy ecosystem for machine learning.

## Mini-Course Overview (what to expect)

This mini-course is divided into 14 parts.

Each lesson was designed to take the average developer about 30 minutes. You might finish some much sooner and other you may choose to go deeper and spend more time.

You can can complete each part as quickly or as slowly as you like. A comfortable schedule may be to complete one lesson per day over a two week period. Highly recommended.

The topics you will cover over the next 14 lessons are as follows:

**- Lesson 01**: Introduction to Theano.
**- Lesson 02**: Introduction to TensorFlow.
**- Lesson 03**: Introduction to Keras.
**- Lesson 04**: Crash Course in Multi-Layer Perceptrons.
**- Lesson 05**: Develop Your First Neural Network in Keras.
**- Lesson 06**: Use Keras Models With Scikit-Learn.
**- Lesson 07**: Plot Model Training History.
**- Lesson 08**: Save Your Best Model During Training With Checkpointing.
**- Lesson 09**: Reduce Overfitting With Dropout Regularization.
**- Lesson 10**: Lift Performance With Learning Rate Schedules.
**- Lesson 11**: Crash Course in Convolutional Neural Networks.
**- Lesson 12**: Handwritten Digit Recognition.
**- Lesson 13**: Object Recognition in Small Photographs.
**- Lesson 14**: Improve Generalization With Data Augmentation.

## Lesson 01: Introduction to Theano

Theano is a Python library for fast numerical computation to aid in the development of deep learning models.

At it’s heart Theano is a compiler for mathematical expressions in Python. It knows how to take your structures and turn them into very efficient code that uses NumPy and efficient native libraries to run as fast as possible on CPUs or GPUs.

The actual syntax of Theano expressions is symbolic, which can be off-putting to beginners used to normal software development. Specifically, expression are defined in the abstract sense, compiled and later actually used to make calculations.

In this lesson your goal is to install Theano and write a small example that demonstrates the symbolic nature of Theano programs.

For example, you can install Theano using pip as follows:

```python
sudo pip install Theano
```

A small example of a Theano program that you can use as a starting point is listed below:

```python
import theano
from theano import tensor
# declare two symbolic floating-point scalars
a = tensor.dscalar()
b = tensor.dscalar()
# create a simple expression
c = a + b
# convert the expression into a callable object that takes (a,b)
# values as input and computes a value for c
f = theano.function([a,b], c)
# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
result = f(1.5, 2.5)
print(result)
```

## Lesson 02: Introduction to TensorFlow

TensorFlow is a Python library for fast numerical computing created and released by Google. Like Theano, TensorFlow is intended to be used to develop deep learning models.

With the backing of Google, perhaps used in some of it’s production systems and used by the Google DeepMind research group, it is a platform that we cannot ignore.

Unlike Theano, TensorFlow does have more of a production focus with a capability to run on CPUs, GPUs and even very large clusters.

In this lesson your goal is to install TensorFlow become familiar with the syntax of the symbolic expressions used in TensorFlow programs.

For example, you can install TensorFlow using pip:

```python
sudo pip install TensorFlow
```

A small example of a TensorFlow program that you can use as a starting point is listed below:

```python
# Example of TensorFlow library
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# declare two symbolic floating-point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# create a simple symbolic expression using the add function
add = tf.add(a, b)
# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)
```

Output:

```
4.0
```

Learn more about TensorFlow on the <a href="https://www.tensorflow.org/">TensorFlow homepage</a>.

## Lesson 03: Introduction to Keras

A difficulty of both Theano and TensorFlow is that it can take a lot of code to create even very simple neural network models.

These libraries were designed primarily as a platform for research and development more than for the practical concerns of applied deep learning.

The Keras library addresses these concerns by providing a wrapper for both Theano and TensorFlow. It provides a clean and simple API that allows you to define and evaluate deep learning models in just a few lines of code.

Because of the ease of use and because it leverages the power of Theano and TensorFlow, Keras is quickly becoming the go-to library for applied deep learning.

The focus of Keras is the concept of a model. The life-cycle of a model can be summarized as follows:

1. Define your model. Create a Sequential model and add configured layers.
2. Compile your model. Specify loss function and optimizers and call the compile()
function on the model.
3. Fit your model. Train the model on a sample of data by calling the fit() function on
the model.
4. Make predictions. Use the model to generate predictions on new data by calling functions such as evaluate() or predict() on the model.

Your goal for this lesson is to install Keras.

For example, you can install Keras using pip:

```python
sudo pip install keras
```

Start to familiarize yourself with the Keras library ready for the upcoming lessons where we will implement our first model.

You can learn more about the Keras library on the <a href="http://keras.io/">Keras homepage</a>.

## Lesson 04: Crash Course in Multi-Layer Perceptrons

