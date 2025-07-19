# Introduction to the Python Deep Learning Library TensorFlow

TensorFlow is a Python library for fast numerical computing created and released by Google.

It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.

## What Is TensorFlow?

TensorFlow is an open-source library for fast numerical computing.

It was created and is maintained by Google and was released under the Apache 2.0 open source license. The API is nominally for the Python programming language, although there is access to the underlying C++ API.

Unlike other numerical libraries intended for use in Deep Learning like Theano, TensorFlow was designed for use both in research and development and in production systems, not least of which is <a href="https://en.wikipedia.org/wiki/RankBrain">RankBrain in Google search</a> and the fun <a href="https://en.wikipedia.org/wiki/DeepDream">DeepDream project</a>.

It can run on single CPU systems and GPUs, as well as mobile devices and large-scale distributed systems of hundreds of machines.

## How to Install TensorFlow

Installation of TensorFlow is straightforward if you already have a Python SciPy environment.

TensorFlow works with Python 3.3+. You can follow the <a href="https://www.tensorflow.org/install">Download and Setup instructions</a> on the TensorFlow website. Installation is probably simplest via PyPI, and specific instructions of the pip command to use for your Linux or Mac OS X platform are on the Download and Setup webpage. In the simplest case, you just need to enter the following in your command line:

```python
pip install tensorflow
```

An exception would be on the newer Mac with an Apple Silicon CPU. The package name for this specific architecture is tensorflow-macos instead:

```python
pip install tensorflow-macos
```

There are also <a href="http://docs.python-guide.org/en/latest/dev/virtualenvs/">virtualenv</a> and <a href="https://www.docker.com/">docker images</a> that you can use if you prefer.

To make use of the GPU, you need to have the Cuda Toolkit installed as well.

## Your First Examples in TensorFlow

Computation is described in terms of data flow and operations in the structure of a directed graph.

**- Nodes**: Nodes perform computation and have zero or more inputs and outputs. Data that moves between nodes are known as tensors, which are multi-dimensional arrays of real values.
**- Edges**: The graph defines the flow of data, branching, looping, and updates to state. Special edges can be used to synchronize behavior within the graph, for example, waiting for computation on a number of inputs to complete.
**- Operation**: An operation is a named abstract computation that can take input attributes and produce output attributes. For example, you could define an add or multiply operation.

### Computation with TensorFlow

This first example is a modified version of the example on the <a href="https://github.com/tensorflow/tensorflow">TensorFlow website</a>. It shows how you can define values as **tensors** and execute an operation.

```python
import tensorflow as tf
a = tf.constant(10)
b = tf.constant(32)
print(a+b)
```

Running this example displays:

```
tf.Tensor(42, shape=(), dtype=int32)
```

### Linear Regression with TensorFlow

This next example comes from the introduction in the TensorFlow tutorial.

This example shows how you can define variables (e.g., W and b) as well as variables that are the result of the computation (y).

We get some sense that TensorFlow separates the definition and declaration of the computation. Below, there is automatic differentiation under the hood. When we use the function <code>mse_loss()</code> to compute the difference between y and <code>y_data</code>, there is a graph created connecting the value produced by the function to the TensorFlow variables W and b. TensorFlow uses this graph to deduce how to update the variables inside the <code>minimize()</code> function.

```python
import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.zeros([1]))

# Minimize the mean squared errors.
optimizer = tf.keras.optimizers.Adam()
for step in range(5000):
    with tf.GradientTape() as tape:
        y = W*x_data+b
        loss = tf.reduce_mean(tf.square(y - y_data))

    grads = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))

    if step % 500 ==0:
        print(f"Step {step}: W = {W.numpy()}, b = {b.numpy()}, loss = {loss.numpy()}")

# Learns best fit is W: [0.1], b: [0.3]
```

Running this example prints the following output:

```
Step 0: W = [-0.41399738], b = [0.00099999], loss = 0.34826985001564026
Step 500: W = [-0.0830213], b = [0.32231948], loss = 0.008317295461893082
Step 1000: W = [-0.00971244], b = [0.3611552], loss = 0.00099891796708107
Step 1500: W = [0.01962436], b = [0.3464727], loss = 0.000547263422049582
Step 2000: W = [0.04682668], b = [0.330749], loss = 0.00023966144362930208
Step 2500: W = [0.06941903], b = [0.3176839], loss = 7.931893924251199e-05
Step 3000: W = [0.08524626], b = [0.3085315], loss = 1.847640669438988e-05
Step 3500: W = [0.09430827], b = [0.30329126], loss = 2.7525704808795126e-06
Step 4000: W = [0.09835245], b = [0.30095273], loss = 2.309380988663179e-07
Step 4500: W = [0.09967104], b = [0.3001902], loss = 9.221054142471985e-09
```

You can learn more about the mechanics of TensorFlow in the <a href="https://www.tensorflow.org/guide/basics">Basic Usage guide</a>.

## More Deep Learning Models

