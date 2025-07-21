# Your First Deep Learning Project in Python with Keras Step-by-Step

**Keras** is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning **models**.

It is part of the TensorFlow library and allows you to define and train neural network models in just a few lines of code.

In this tutorial, you will discover how to create your first deep learning neural network model in Python using Keras.

## Keras Tutorial Overview

There is not a lot of code required, but we will go over it slowly so that you will know how to create your own models in the future.

*The steps you will learn in this tutorial are as follows*:

1. Load Data
2. Define Keras Model
3. Compile Keras Model
4. Fit Keras Model
5. Evaluate Keras Model
6. Tie It All Together
7. Make Predictions

**This Keras tutorial makes a few assumptions. You will need to have**:

1. Python 2 or 3 installed and configured
2. SciPy (including NumPy) installed and configured
3. Keras and a backend (Theano or TensorFlow) installed and configured

Create a new file called **keras_first_network.py** and type or copy-and-paste the code into the file as you go.

## 1. Load Data

The first step is to define the functions and classes you intend to use in this tutorial.

You will use the <a href="https://www.numpy.org/">NumPy library</a> to load your dataset and two classes from the <a href="https://www.tensorflow.org/api_docs/python/tf/keras">Keras library</a> to define your model.

The imports required are listed below.

```python
# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

You can now load our dataset.

In this Keras tutorial, you will use the Pima Indians onset of diabetes dataset. This is a standard machine learning dataset from the UCI Machine Learning repository. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

As such, it is a binary classification problem (onset of diabetes as 1 or not as 0). All of the input variables that describe each patient are numerical. This makes it easy to use directly with neural networks that expect numerical input and output values and is an ideal choice for our first neural network in Keras.

The dataset is available here:

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv">Dataset CSV File (pima-indians-diabetes.csv)</a>

- <a href="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names">Dataset Details</a>

Download the dataset and place it in your local working directory, the same location as your Python file.

Save it with the filename:

*pima-indians-diabetes.csv*

You can now load the file as a matrix of numbers using the NumPy function <a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.loadtxt.html">loadtxt()</a>.

There are eight input variables and one output variable (the last column). You will be learning a model to map rows of input variables (X) to an output variable (y), which is often summarized as *y = f(X)*.

The variables can be summarized as follows:

Input Variables (X):

1. Number of times pregnant
2. Plasma glucose concentration at 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)

Output Variables (y):

1. Class variable (0 or 1)

Once the CSV file is loaded into memory, you can split the columns of data into input and output variables.

The data will be stored in a 2D array where the first dimension is rows and the second dimension is columns, e.g., [rows, columns].

You can split the array into two arrays by selecting subsets of columns using the standard NumPy slice operator or “:”. You can select the first eight columns from index 0 to index 7 via the slice 0:8. We can then select the output column (the 9th variable) via index 8.

```python
# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
```

You are now ready to define your neural network model.

**Note**: The dataset has nine columns, and the range 0:8 will select columns from 0 to 7, stopping before index 8. If this is new to you, then you can learn more about array slicing and ranges in this post:

## 2. Define Keras Model

Models in Keras are defined as a sequence of layers.

We create a <a href="https://keras.io/models/sequential/">*Sequential model*</a> and add layers one at a time until we are happy with our network architecture.

The first thing to get right is to ensure the input layer has the correct number of input features. This can be specified when creating the first layer with the **input_shape** argument and setting it to (8,) for presenting the eight input variables as a vector.

How do we know the number of layers and their types?

This is a tricky question. There are heuristics that you can use, and often the best network structure is found through a process of trial and error experimentation (I explain more about this here). Generally, you need a network large enough to capture the structure of the problem.

In this example, let’s use a fully-connected network structure with three layers.

Fully connected layers are defined using the <a href="https://keras.io/layers/core/">Dense class</a>. You can specify the number of neurons or nodes in the layer as the first argument and the activation function using the **activation** argument.

Also, you will use the rectified linear unit activation function referred to as ReLU on the first two layers and the Sigmoid function in the output layer.

It used to be the case that Sigmoid and Tanh activation functions were preferred for all layers. These days, better performance is achieved using the ReLU activation function. Using a sigmoid on the output layer ensures your network output is between 0 and 1 and is easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

You can piece it all together by adding each layer:

- The model expects rows of data with 8 variables (the *input_shape=(8,)* argument).
- The first hidden layer has 12 nodes and uses the relu activation function.
- The second hidden layer has 8 nodes and uses the relu activation function.
- The output layer has one node and uses the sigmoid activation function.

```python
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

**Note**:  The most confusing thing here is that the shape of the input to the model is defined as an argument on the first hidden layer. This means that the line of code that adds the first Dense layer is doing two things, defining the input or visible layer and the first hidden layer.

## 3. Compile Keras Model
