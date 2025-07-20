# Introduction to Python Deep Learning with Keras

Two of the top numerical platforms in Python that provide the basis for Deep Learning research and development are Theano and TensorFlow.

Both are very powerful libraries, but both can be difficult to use directly for creating deep learning models.

In this post, you will discover the Keras Python library that provides a clean and convenient way to create a range of deep learning models on top of Theano or TensorFlow.

## What is Keras?

Keras is a minimalist Python library for deep learning that can run on top of Theano or TensorFlow.

It was developed to make implementing deep learning models as fast and easy as possible for research and development.

It runs on Python 2.7 or 3.5 and can seamlessly execute on GPUs and CPUs given the underlying frameworks. It is released under the permissive MIT license.

Keras was developed and maintained by <a href="https://www.linkedin.com/in/fchollet">François Chollet</a>, a Google engineer using four guiding principles:

- **Modularity**: A model can be understood as a sequence or a graph alone. All the concerns of a deep learning model are discrete components that can be combined in arbitrary ways.
- **Minimalism**: The library provides just enough to achieve an outcome, no frills and maximizing readability.
- **Extensibility**: New components are intentionally easy to add and use within the framework, intended for researchers to trial and explore new ideas.
- **Python**: No separate model files with custom file formats. Everything is native Python.

## How to Install Keras

Keras is relatively straightforward to install if you already have a working Python and SciPy environment.

You must also have an installation of Theano or TensorFlow on your system already.

Keras can be installed easily using PyPI, as follows:

```
pip install keras
```

## Theano and TensorFlow Backends for Keras

Assuming you have both Theano and TensorFlow installed, you can configure the backend used by Keras.

The easiest way is by adding or editing the Keras configuration file in your home directory:

```
~/.keras/keras.json
```

Which has the format:

```
{
    "image_data_format": "channels_last",
    "backend": "tensorflow",
    "epsilon": 1e-07,
    "floatx": "float32"
}
```

In this configuration file you can change the “backend” property from “tensorflow” (the default) to “theano“. Keras will then use the configuration the next time it is run.

You can confirm the backend used by Keras using the following snippet on the command line:

```
python -c "from keras import backend; print(backend.backend())"
```

Running this with default configuration you will see:

```
Using TensorFlow backend.
tensorflow
```

You can also specify the backend to use by Keras on the command line by specifying the KERAS_BACKEND environment variable, as follows:

```
KERAS_BACKEND=theano python -c "from keras import backend; print(backend.backend())"
```

Running this example prints:

```
Using Theano backend.
theano
```

## Build Deep Learning Models with Keras

The focus of Keras is the idea of a model.

The main type of model is called a Sequence which is a linear stack of layers.

You create a sequence and add layers to it in the order that you wish for the computation to be performed.

Once defined, you compile the model which makes use of the underlying framework to optimize the computation to be performed by your model. In this you can specify the loss function and the optimizer to be used.

Once compiled, the model must be fit to data. This can be done one batch of data at a time or by firing off the entire model training regime. This is where all the compute happens.

Once trained, you can use your model to make predictions on new data.

We can summarize the construction of deep learning models in Keras as follows:

1. **Define your model**. Create a sequence and add layers.
2. **Compile your model**. Specify loss functions and optimizers.
3. **Fit your model**. Execute the model using data.
4. **Make predictions**. Use the model to generate predictions on new data.

## Keras Resources

The list below provides some additional resources that you can use to learn more about Keras.

- <a href="http://keras.io/">Keras Official Homepage</a>

- <a href="https://github.com/fchollet/keras">Keras Project on GitHub</a>

- <a href="https://groups.google.com/forum/#!forum/keras-users">Keras User Group</a>

## Summary

In this post, you discovered the Keras Python library for deep learning research and development.

You discovered that Keras is designed for minimalism and modularity allowing you to very quickly define deep learning models and run them on top of a Theano or TensorFlow backend.
