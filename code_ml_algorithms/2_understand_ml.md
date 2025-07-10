# Understand Machine Learning Algorithms By Implementing Them From Scratch

Implementing machine learning algorithms from scratch seems like a great way for a programmer to understand machine learning.

And maybe it is.

But there some downsides to this approach too.

In this post you will discover some great resources that you can use to implement machine learning algorithms from scratch.

You will also discover some of the limitations of this seemingly perfect approach.

## Benefits of Implementing Machine Learning Algorithms From Scratch

I promote the idea of implementing machine learning algorithms from scratch.

I think you can learn a lot about how algorithms work. I also think that as a developer, it provides a bridge into learning the mathematical notations, descriptions and intuitions used in machine learning.

In the post I listed the benefits as:

1. the understanding you gain
2. the starting point it provides
3. the ownership of the algorithm and code it forces

I’ve helped a lot of programmers get started in machine learning over the last few years. From my experience, I list 5 of the most common stumbling blocks that I see tripping up programmers and the tactics that you can use to over come them.

Finally, you will discover 3 quick tips to getting the most from code tutorials and going from a copy-paste programmer (if you happen to be one) to truly diving down the rabbit hole of machine learning algorithms.

## Great Books You Can Use To Implement Algorithms

I have implemented a lot of algorithms from scratch, directly from research papers. It can be very difficult.

It is a much gentler start to follow someone else’s tutorial.

There are many excellent resources that you can use to get started implementing machine learning algorithms from scratch.

### Data Science from Scratch: First Principles with Python by Joel Grus

This truly is from scratch, working through visualization, stats, probability, working with data and then 12 or so different machine learning algorithms.

This is one of my favorite beginner machine learning books from this year.

![image](https://github.com/user-attachments/assets/7eaf9e36-8429-4ede-9da8-3f8c213d0387)

### Machine Learning: An Algorithmic Perspective by Stephen Marsland

This is the long awaited second edition to this popular book. This covers a large number of diverse machine learning algorithms with implementations.

I like that it gives a mix of mathematical description, pseudo code as well as working source code.

![image](https://github.com/user-attachments/assets/79113ba0-b435-4bce-87dc-3936af73053b)

### Machine Learning in Action by Peter Harrington

This book works through the 10 most popular machine learning algorithms providing case study problems and worked code examples in Python.

I like that there is a good effort to tie the code to the descriptions using numbering and arrows.

![image](https://github.com/user-attachments/assets/3e16ba3c-69a3-4277-a120-3ea8dbd2e413)

## 5 Stumbling Blocks When Implementing Algorithms From Scratch (and how to overcome them)

Implementing machine learning algorithms from scratch using tutorials is a lot of fun.

But there can be stumbling blocks, and if you’re not careful, they may trip you up and kill your motivation.

In this section I want to point out the 5 most common stumbling blocks that I see and how to roll with them and not let them hold you up. I want you to get unstuck and plow on (or move on to another tutorial).

Some good general advice for avoiding the stumbling blocks below is to carefully check the reviews of books (or the comments on blog posts) before diving into a tutorial. You want to be sure that the code works and that you’re not wasting your time.

Another general tactic is to dive-in no matter what and figure out the parts that are not working and re-implement them yourself. This is a great hack to force understanding, but it’s probably not for the beginner and you may require a good technical reference close at hand.

Anyway, let’s dive into the 5 common stumbling blocks with machine learning from scratch tutorials:

### 1) The Code Does Not Work

The worst and perhaps most common stumbling block is that the code in the example does not work.

In fact, if you spend some time in the book reviews on Amazon for some texts or in the comments of big blog posts, it’s clear that this problem is more prevalent than you think.

How does this happen? A few reasons come to mind that might give you clues to applying your own fixes and carrying on:

- **The code never worked**. This means that the book was published without being carefully edited. Not much you can do here other than perhaps getting into the mind of the author and trying to figure out what they meant. Maybe even try contacting the author or the publisher.
- **The language has moved on**. This can happen, especially if the post is old or the book has been in print for a long time. Two good examples are the version of Ruby moving from 1.x to 2.x and Python moving from 2.x to 3.x.
- **The third-party libraries have moved on**. This is for those cases where the implementations were not totally from scratch and some utility libraries were used, such as for plotting. This is often not that bad. You can often just update the code to use the latest version of the library and modify the arguments to meet the API changes. It may even be possible to install an older version of the library (if there are few or no dependencies that you might break in your development environment).
- **The dataset has moved on**. This can happen if the data file is a URL and is no longer available (perhaps you can find the file elsewhere). It is much worse if the example is coded against a third-party API data source like Facebook or Twitter. These APIs can change a lot and quickly. Your best bet is to understand the most recent version of the API and adapt the code example, if possible.

A good general tactic if the code does not work is to look for the associated errata if it is a book, GitHub repository, code downloads or similar. Sometimes the problems have been fixed and are available on the book or author’s website. Some simple Googling should turn it up.

### 2) Poor Descriptions Of Code

I think the second worst stumbling block when implementing algorithms from scratch is when the descriptions provided with the code are bad.

These types of problems are particularly not good for a beginner, because you are trying your best to stay motivated and actually learn something from the exercise. All of that goes down in smoke if the code and text do not align.

I (perhaps kindly) call them “*bad descriptions*” because there may be many symptoms and causes. For example:

- **A mismatch between code and description**. This may have been caused by the code and text being prepared at different times and not being correctly edited together. It may be something small like a variable name change or it may be whole function names or functions themselves.
- **Missing explanations**. Sometimes you are given large slabs of code that you are expected to figure out. This is frustrating, especially in a book where it’s page after page of code that would be easier to understand on the screen. If this is the case, you might be better off finding the online download for the code and working with it directly.
- **Terse explanations**. Sometimes you get explanations of the code, but they are too brief, like “uses information gain” or whatever. Frustrating! You still may have enough to research the term, but it would be much easier if the author had included an explanation in the context and relevant to the example.

A good general tactic is to look up description for the algorithm in other resources and try to map them onto the code you are working with. Essentially, try to build your own descriptions for the code.

This just might not be an option for a beginner and you may need to move on to another resource.

### 3) Code is not Idiomatic

We programmers can be pedantic about the “correct” use of our languages (e.g. Python code is not Pythonic). This is a good thing, it shows good attention to detail and best practices.

When sample code is not idiomatic to the language in which it is written it can be off putting. Sometimes it can be so distracting that the code can be unreadable.

There are many reasons that this may be the case, for example:

- **Port from another language**. The sample code may be a port from another programming language. Such as FORTRAN in Java or C in Python. To a trained eye, this can be obvious.
- **Author is learning the language**. Sometimes the author may use a book or tutorial project to learn a language. This can be manifest by inconsistency throughout the code examples. This can be frustrating and even distracting when examples are verbose making poor use of language features and API.
- **Author has not used the language professionally**. This can be more subtle to spot and can be manifest by the use of esoteric language features and APIs. This can be confusing when you have to research or decode the strange code.

If idiomatic code is deeply important to you, these stumbling blocks could be an opportunity. You could port the code from the “Java-Python” hybrid (or whatever) to a pure Pythonic implementation.

In so doing, you would gain a deeper understanding for the algorithm and more ownership over the code.

### 4) Code is not Connected to the Math

A good code example or tutorial will provide a bridge from the mathematical description to the code.

This is important because it allows you to travel across and start to build an intuition for the notation and the concise mathematical descriptions.

There problem is, sometimes this bridge may be broken or missing completely.

- **Errors in the math**. This is insidious for the beginner that is already straining to build connections from the math to the code. Incorrect math can mislead or worse consume vast amounts of time with no pay off. Knowing that it is possible, is a good start.
- **Terse mathematical description**. Equations may be littered around the sample code, leaving it to you to figure out what it is and how it relates to the code. You have few options, you could just treat it as a math free example and refer to a different more complete reference text, or you could put in effort to relate the math to the code yourself. This is more likely by authors that are not familiar with the mathematical description of the algorithm and seemingly drop it in as an after thought.
- **Missing mathematics**. Some references are math free, by design. In this case you may need to find your own reference text and build the bridge yourself. This is probably not for beginners, but it is a skill well worth investing the time into.

### 5) Incomplete Code Listing

We saw in 2) that you can have no descriptions and long listings of code. This problem can be inverted where you don’t have enough code. This is the case when the code listing is incomplete.

I am a big believer in complete code listings. I think the code listing should give you everything you need to give a “*complete*” and working implementation, even if it is the simplest possible case.

You can build on a simple case, you can’t run an incomplete example. You have to put in work and tie it all together.

Some reasons that this stumbling block may be the case, are:

- **Elaborate descriptions**. Verbose writing can be a sign of incomplete thinking. Not always, but sometimes. If something is not well understood there may be an implicit attempt to cover it up with a wash of words. If there is no code at all, you could take it as a challenge to design the algorithm from the description and corroborate it from other descriptions and resources.
- **Code snippets**. Concepts may be elaborately described then demonstrated with a small code snippet. This can help to closely tie the concept to the code snippet, but it requires a lot of work on your behalf to tie it all together into a working system.
- **No sample output**. A key aspect often missing from code examples is a sample output. If present, this can give you an unambiguous idea of what to expect when you run it. Without a sample output, it’s a total guess.

In some situations, having to tie code together yourself might present an interesting challenge. Again, not suitable for the beginner, but perhaps a fun exercise later once you have some algorithms under your belt.

## 3 Tips to Get The Most From Implementing Algorithms

You may implement a fair number of algorithms. Once you do a few you may do a few more and before you know it, you’ve built your own little library of algorithms that you understand intimately.

In this section I wan to give you 3 quick tips that you can use to get the most out of your experiences implementing machine learning algorithms.

1. **Add advanced features**. Take your working code example and build on it. If the tutorial is any good, it will list ideas for extension. If not, you can research some yourself. List a number of candidate extensions to the algorithm and implement them, one-by-one. This will force you to at least understand the code enough to make the modification.
2. **Adapt to another problem**. Run the algorithm on a different dataset. Fix any issues if it breaks. Go further and adapt the implementation to a different problem. If the code example was two-class classification, update it for multi-class classification or regression.
3. **Visualize algorithm behavior**. I find plotting algorithm performance and behavior in real-time a very valuable learning tool, even today. You can start out by plotting at the epoch-level (all algorithms are iterative at some level) accuracy on the test and training datasets. From there, you can pick out algorithm specific visualizations, like the 2D grids of a self-organizing map, the coefficients on a time series in regression, and a voronoi tessellation for a k-Nearest Neighbors algorithm.

I think these tips will allow you to go a lot further than the tutorials and code examples.

This last point especially will give you deep insights into algorithm behavior that few practitioners take the time to acquire.

