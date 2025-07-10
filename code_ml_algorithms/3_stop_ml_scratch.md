# Stop Coding Machine Learning Algorithms From Scratch

**You Don’t Have To Implement Algorithms
…if you’re a beginner and just getting started.**

Stop.

Are you implementing a machine learning algorithm at the moment?

Why?

Implementing algorithms from scratch is one of the biggest mistakes I see beginners make.

In this post you will discover:

- The algorithm implementation trap that beginners fall into.
- The very real difficulty of engineering world-class implementations of machine learning algorithms.
- Why you should be using off-the-shelf implementations.

## Caught In The Implementation Trap

It seems that a lot of developers get caught in this challenge.

They are told or imply that:

**Algorithms must be implemented
before being used.**

Or that:

**You can only learn machine learning by
implementing algorithms.**

Here are some similar questions I stumbled across:

- *Why is there a need to manually implement machine learning algorithms when there are many advanced APIs like tensorflow available?* (<a href="https://www.quora.com/Why-is-there-a-need-to-manually-implement-machine-learning-algorithms-when-there-are-many-advanced-APIs-like-tensorflow-available">on Quora</a>)
- *Is there any value implementing machine learning algorithms by yourself or should you use libraries?* (<a href="https://www.quora.com/Is-there-any-value-implementing-machine-learning-algorithms-by-yourself-or-should-you-use-libraries">on Quora</a>)
- *Is it useful to implement machine learning algorithms?* (<a href="https://www.quora.com/Is-it-useful-to-implement-machine-learning-algorithms">on Quora</a>)
- *Which programming language should I use to implement Machine Learning algorithms?* (<a href="https://www.quora.com/Which-programming-language-should-I-use-to-implement-Machine-Learning-algorithms">on Quora</a>)
- *Why do you and other people sometimes implement machine learning algorithms from scratch?* (<a href="https://github.com/rasbt/python-machine-learning-book/blob/master/faq/implementing-from-scratch.md">on GitHub</a>)

## You’re Probably Doing it Wrong

You don’t have to implement machine learning algorithms from scratch.

This is a part of the bottom-up approach traditionally used to teach machine learning.

1. Learn Math.
2. Learn Theory.
3. Implement Algorithm From Scratch.
4. ??? (*magic happens here*).
5. Apply Machine Learning.

It is a lot easier to apply machine learning algorithms to a problem and get a result than it is to implement them from scratch.

**A Lot Easier!**

Learning how to use an algorithm rather than implement an algorithm is not only easier, it is a more valuable skill. A skill that you can start using to make a real impact very quickly.

There’s a lot of low-hanging fruit that you can pick with applied machine learning.

## Implementing Machine Learning Algorithms Well …is Really Hard!

Algorithms that you use to solve business problems need to be **fast** and **correct**.

### Fast Algorithms

The more sophisticated nonlinear methods require a lot more data than their linear counterparts.

This means they need to do a lot of work, which may take a long time.

Algorithms need to be fast to process through all of this data. Especially, at scale.

This may require a re-interpretation of the linear algebra that underlies the method in such a way that best suits a specific matrix operation in an underlying library.

It may require specialized knowledge of caching to make the most of your hardware.

These are not ad hoc tricks that come together after you get a “*hello world*” implementation working. These are engineering challenges that encompass the algorithm implementation project.

### Correct Algorithms

Machine learning algorithms will give you a result, even when their implementation is crippled.

You get a number. An output. A prediction.

Sometimes the prediction is correct and sometimes it is not.

Machine learning algorithms use randomness. They are stochastic algorithms.

This is not just a matter of unit tests, it is a matter of having a deep understanding of the technique and devising cases to prove the implementation is as expected and edge cases are handled.

## Use An Off-The-Shelf Implementation

You may be an excellent engineer.

But your “*hello world*” implementation of an algorithm will probably not cut-it when compared to an off-the-shelf implementation.

Your implementation will probably be based on a textbook description, meaning it will be naive and slow. And you may or may not have the expertise to devise tests to ensure the correctness of your implementation.

Off-the-shelf implementations in open source libraries are built for speed and/or robustness.

**How could you not use a standard machine learning library?**

They may be tailored to a very narrow problem type intended to be as fast as possible. They may also be intended for general purpose use, ensuring they operate correctly on a wide range of problems, beyond those you have considered.

### Libraries Are Not All Created Equal

Not all algorithm implementations you download off the Internet are created equal.

The code snippet from GitHub maybe a grad students “hello world” implementation, or it may be the highly optimized implementation contributed to by the entire research team at a large organization.

You need to evaluate the source of the code you are using. Some sources are better or more reliable than others.

General purposes libraries are often more robust at the cost of some speed.

Lighting fast implementations by hacker-engineers often suffer poor documentation and are highly pedantic when it comes to their expectations.

Consider this when you pick your implementation.

### Recommendations

When asked, I typically recommend one of three platforms:

1. **Weka**. A graphical user interface that does not require any code. Perfect if you want to focus on the machine learning first and learning how to work through problems.
2. **Python**. The ecosystem including pandas and scikit-learn. Excellent for stitching together a solution to a machine learning problem in development that is robust enough to also be deployed into operations.
3. **R**. The more advanced platform that although has an esoteric language and sometimes buggy packages, offers access to state-of-the-art methods written directly by academics. Great for one-off projects and R&D.

These are just my recommendations, there are many more machine learning platforms to choose from.

## Sometimes You Must Implement

You do not have to implement machine learning algorithms when getting started in machine learning.

But you can.

And there can be very good reasons for doing so.

For example here are 3 big reasons:

- You want to implement to learn how the algorithm works.
- There is no available implementation of the algorithm you need.
- There is no suitable (fast enough, etc.) implementation of the algorithm you need.

The first is my favorite. It’s the one that may have confused you.

You can implement machine learning algorithms to learn how they work. I recommend it. It’s very efficient for developers to learn this way.

But.

You do not have to **start** by implementing machine learning algorithms. You will build your confidence and skill in machine learning a lot faster by learning how to use machine learning algorithms before implementing them.

The implementation and any research required to complete the implementation would then be an improvement on your understanding. An addition that would help you to get better results the next time you used that algorithm.

## Summary

In this post, you discovered that beginners fall into the trap of implementing machine learning algorithms from scratch.

**They are told that it’s the only way.**

You discovered that engineering fast and robust implementations of machine learning algorithms is a tough challenge.

You learned that it is much easier and more desirable to learn how to use machine learning algorithms before implementing them. You also learned that implementing algorithms is a great way to learn more about how they work and get more from them, but only after you know how to use them.

