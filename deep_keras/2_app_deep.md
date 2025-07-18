# 8 Inspirational Applications of Deep Learning

It is hyperbole to say deep learning is achieving state-of-the-art results across a range of difficult problem domains. A fact, but also hyperbole.

There is a lot of excitement around artificial intelligence, machine learning and deep learning at the moment. It is also an amazing opportunity to get on on the ground floor of some really powerful tech.

I try hard to convince friends, colleagues and students to get started in deep learning and bold statements like the above are not enough. It requires stories, pictures and research papers.

In this post you will discover amazing and recent applications of deep learning that will inspire you to get started in deep learning.

Getting started in deep learning does not have to mean go and study the equations for the next 2-3 years, it could mean download Keras and start running your first model in 5 minutes flat. Start applied deep learning. Build things. Get excited and turn it into code and systems.

## Overview

Below is the list of the specific examples we are going to look at in this post.

Not all of the examples are technology that is ready for prime time, but guaranteed, they are all examples that will get you excited.

Some are examples that seem ho hum if you have been around the field for a while. In the broader context, they are not ho hum. Not at all.

Frankly, to an old AI hacker like me, some of these examples are a slap in the face. Problems that I simply did not think we could tackle for decades, if at all.

I’ve focused on visual examples because we can look at screenshots and videos to immediately get an idea of what the algorithm is doing, but there are just as many if not more examples in natural language with text and audio data that are not listed.

Here’s the list:

1. Colorization of Black and White Images.
2. Adding Sounds To Silent Movies.
3. Automatic Machine Translation.
4. Object Classification in Photographs.
5. Automatic Handwriting Generation.
6. Character Text Generation.
7. Image Caption Generation.
8. Automatic Game Playing.

## 1. Automatic Colorization of Black and White Images

Image colorization is the problem of adding color to black and white photographs.

Traditionally this was done by hand with human effort because it is such a difficult task.

Deep learning can be used to use the objects and their context within the photograph to color the image, much like a human operator might approach the problem.

A visual and highly impressive feat.

This capability leverages of the high quality and very large convolutional neural networks trained for ImageNet and co-opted for the problem of image colorization.

Generally the approach involves the use of very large convolutional neural networks and supervised layers that recreate the image with the addition of color.

<img width="733" height="507" alt="image" src="https://github.com/user-attachments/assets/6e1e606f-e8c6-454f-bdd2-d9c3bc1dfdef" />

Impressively, the same approach can be used to colorize still frames of black and white movies

### Further Reading

- <a href="http://tinyclouds.org/colorize/">Automatic Colorization</a>

- <a href="https://news.developer.nvidia.com/automatic-image-colorization-of-grayscale-images/">Automatic Colorization of Grayscale Images</a>

## 2. Automatically Adding Sounds To Silent Movies

In this task the system must synthesize sounds to match a silent video.

The system is trained using 1000 examples of video with sound of a drum stick striking different surfaces and creating different sounds. A deep learning model associates the video frames with a database of pre-rerecorded sounds in order to select a sound to play that best matches what is happening in the scene.

The system was then evaluated using a turing-test like setup where humans had to determine which video had the real or the fake (synthesized) sounds.

A very cool application of both convolutional neural networks and LSTM recurrent neural networks.

### Further Reading

- <a href="http://news.mit.edu/2016/artificial-intelligence-produces-realistic-sounds-0613">Artificial intelligence produces realistic sounds that fool humans</a>

- <a href="https://www.engadget.com/2016/06/13/machines-can-generate-sound-effects-that-fool-humans">Machines can generate sound effects that fool humans</a>

## 3. Automatic Machine Translation

This is a task where given words, phrase or sentence in one language, automatically translate it into another language.

Automatic machine translation has been around for a long time, but deep learning is achieving top results in two specific areas:

- Automatic Translation of Text.
- Automatic Translation of Images.

Text translation can be performed without any preprocessing of the sequence, allowing the algorithm to learn the dependencies between words and their mapping to a new language. Stacked networks of large LSTM recurrent neural networks are used to perform this translation.

As you would expect, convolutional neural networks are used to identify images that have letters and where the letters are in the scene. Once identified, they can be turned into text, translated and the image recreated with the translated text. This is often called instant visual translation.

### Further Reading

- <a href="https://research.googleblog.com/2015/07/how-google-translate-squeezes-deep.html">How Google Translate squeezes deep learning onto a phone</a>

## 4. Object Classification and Detection in Photographs

This task requires the classification of objects within a photograph as one of a set of previously known objects.

State-of-the-art results have been achieved on benchmark examples of this problem using very large convolutional neural networks. A breakthrough in this problem by Alex Krizhevsky et al. results on the ImageNet classification problem called AlexNet.

A more complex variation of this task called object detection involves specifically identifying one or more objects within the scene of the photograph and drawing a box around them.

### Further Reading

- <a href="https://research.googleblog.com/2014/09/building-deeper-understanding-of-images.html">Building a deeper understanding of images</a>

- <a href="https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet">AlexNet</a>

- <a href="http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html">ConvNetJS: CIFAR-10 demo</a>

## 5. Automatic Handwriting Generation

This is a task where given a corpus of handwriting examples, generate new handwriting for a given word or phrase.

The handwriting is provided as a sequence of coordinates used by a pen when the handwriting samples were created. From this corpus the relationship between the pen movement and the letters is learned and new examples can be generated ad hoc.

What is fascinating is that different styles can be learned and then mimicked. I would love to see this work combined with some forensic hand writing analysis expertise.

### Further Reading

- <a href="http://www.cs.toronto.edu/~graves/handwriting.html">Interactive Handwriting Generation Demo</a>

## 6. Automatic Text Generation

This is an interesting task, where a corpus of text is learned and from this model new text is generated, word-by-word or character-by-character.

The model is capable of learning how to spell, punctuate, form sentiences and even capture the style of the text in the corpus.

Large recurrent neural networks are used to learn the relationship between items in the sequences of input strings and then generate text. More recently LSTM recurrent neural networks are demonstrating great success on this problem using a character-based model, generating one character at time.

Andrej Karpathy provides many examples in his popular blog post on the topic including:

- Paul Graham essays
- Shakespeare
- Wikipedia articles (including the markup)
- Algebraic Geometry (with LaTeX markup)
- Linux Source Code
- Baby Names

### Further Reading

- <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/">The Unreasonable Effectiveness of Recurrent Neural Networks</a>

- <a href="https://larseidnes.com/2015/10/13/auto-generating-clickbait-with-recurrent-neural-networks/">Auto-Generating Clickbait With Recurrent Neural Networks</a>

## 7. Automatic Image Caption Generation

Automatic image captioning is the task where given an image the system must generate a caption that describes the contents of the image.

In 2014, there were an explosion of deep learning algorithms achieving very impressive results on this problem, leveraging the work from top models for object classification and object detection in photographs.

Once you can detect objects in photographs and generate labels for those objects, you can see that the next step is to turn those labels into a coherent sentence description.

This is one of those results that knocked my socks off and still does. Very impressive indeed.

Generally, the systems involve the use of very large convolutional neural networks for the object detection in the photographs and then a recurrent neural network like an LSTM to turn the labels into a coherent sentence.

### Further Reading

- <a href="https://research.googleblog.com/2014/11/a-picture-is-worth-thousand-coherent.html">A picture is worth a thousand (coherent) words: building a natural description of images</a>

- <a href="https://blogs.technet.microsoft.com/machinelearning/2014/11/18/rapid-progress-in-automatic-image-captioning/">Rapid Progress in Automatic Image Captioning</a>

## 8. Automatic Game Playing

This is a task where a model learns how to play a computer game based only on the pixels on the screen.

This very difficult task is the domain of deep reinforcement models and is the breakthrough that <a href="https://en.wikipedia.org/wiki/Google_DeepMind">DeepMind</a> (now part of google) is renown for achieving.

This work was expanded and culminated in Google DeepMind’s <a href="https://en.wikipedia.org/wiki/AlphaGo">AlphaGo</a> that beat the world master at the game Go.

### Further Reading

- <a href="https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html">Deep Q Learning Demo</a>

## More Resources

There are a lot of great resources, talks and more to help you get excited about the capabilities and potential for deep learning.

Below are a few additional resources to help get you excited.

- <a href="http://rodrigob.github.io/are_we_there_yet/build/">Which algorithm has achieved the best results</a>

## Summary

In this post you have discovered 8 applications of deep learning that are intended to inspire you.

This *show* rather than *tell* approach is expect to cut through the hyperbole and give you a clearer idea of the current and future capabilities of deep learning technology.
