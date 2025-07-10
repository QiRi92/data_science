# Don’t Start with Open-Source Code When Implementing Machine Learning Algorithms

<a href="https://www.linkedin.com/pub/edward-raff/40/920/99">Edward Raff</a> is the author of the Java Machine Learning library called <a href="https://code.google.com/p/java-statistical-analysis-tool/">JSAT</a> (which is an acronym for Java Statistical Analysis Tool).

Edward has implemented many algorithms in creating this library and I recently reached out to him and asked what advice he could give to beginners implementing machine learning algorithms from scratch.

In this post we take a look at tips on implementing machine learning algorithms based on Edwards advice.

## Don’t Read Other Peoples Source Code

At least, not initially.

What drew me to ask Edward questions about his advice on implementing machine learning algorithms from scratch was his comment on a Reddit question, titled appropriately “<a href="http://www.reddit.com/r/MachineLearning/comments/2h94uj/implementing_machine_learning_algorithms/">Implementing Machine Learning Algorithms</a>“.

In <a href="http://www.reddit.com/r/MachineLearning/comments/2h94uj/implementing_machine_learning_algorithms/ckqrn1t">his comment</a>, Edward suggested that beginners avoid looking at source code of other open source implementations as much as possible. He knew this was counter to most advice (even my own) and it really caught my attention.

Edward start’s out suggesting that there are two quite different tasks when implementing machine learning algorithms:

1. **Implementing Well Known Algorithms**. These are well described in many papers, books, websites lecture notes and so on. You have many sources, they algorithms are relatively straight forward and they are good case studies for self education.
2. **Implementing Algorithms From Papers**. These are algorithms that have limited and sparse descriptions in literature and require significant work and prior knowledge to understand and implement.

### Implementing Well Known Algorithms

Edward suggests reading code is a bad idea if you are interested in implementing well known algorithms.

I distilled at least 3 key reasons for why this is the case from his comments:

- **Code Optimization**: Code in open source libraries is very likely highly optimized for execution speed, memory and accuracy. This means that it implements all kinds of mathematical and programming tricks. As a result, the code will be very difficult to follow. You will spend the majority of your time figuring out the tricks rather than figuring out the algorithm, which was your goal in the first place.
- **Project Centric**: The code will not be a generic implementation ready for you to run in isolation, it will be carefully designed to work within the project’s framework. It is also very likely that details will be abstracted and hidden from you “conveniently” by the framework. You will spend your time learning that framework and it’s design in order to understand the algorithm implementation.
- **Limited Understanding**: Studying an implementation of an algorithm does not help you in your ambition to understand the algorithm, it can teach you tricks of efficient algorithm implementation. In the beginning, the most critical time, other peoples code will confuse you.

I think there is deep wisdom here.

I would point out that open source implementations can sometimes help in the understanding of a specific technical detail, such as an update rule or other modular piece of mathematics that may be poorly described, but realized in a single function in code. I have experienced this myself many times, but it is a heuristic, not a rule.

Edward suggests algorithms like k-means and stochastic gradient descent as good examples to start with.

### Implementing Algorithms From Papers

Edward suggests that implementing machine learning algorithms from papers is a big jump if you have not first implemented well known algorithms, as described above.

From Edwards comments you can sketch out a process for learning machine learning algorithms by implementing them from scratch. My interpretation of that process looks something like the following:

1. Implement the algorithm yourself from scratch.
2. Compare performance to off-the-shelf implementations.
3. Work hard to meet performance and results.
4. Look at open source code to understand advanced tips and tricks.

He suggests that creating your own un-optimized implementation will point out to you where the inefficiencies are, motivate you to fix them, motivating you to understand them in depth and seek out how they have been solved elsewhere.

He further suggests that simply coping an implementation won’t teach you what you need to know, that you will miss out on that deep understanding of the algorithm and its unoptimized performance characteristics and how optimizations can be generalized across algorithms of a similar class.

## Advice for Beginners

After some discussion over email, Edward expanded on his comments and wrote up his thoughts in a blog post titled “<a href="http://jsatml.blogspot.com.au/2014/10/beginner-advice-on-learning-to.html">Beginner Advice on Learning to Implement ML Algorithms</a>“.

This is a great post. In it he addresses three key questions: how to implement machine learning algorithms from scratch, common traps for beginners and resources that may help.

The post is not just great because the advice comes from hard earned wisdom (Edward does machine learning the hard way – he practices it, as you should), but there are few if any posts out there like it. No one is talking about how to implement machine learning algorithms from scratch. It is my mission to work on this problem at the moment.

Edward’s key message is that you need to practice. Implementing machine learning algorithms requires that you understand the background to each algorithm, including the theory, mathematics and history of the field and the algorithm. This does not come quickly or easily. You must work at it, iterate on your understanding and practice, a lot.

If you are a professional programmer, than you know, mastery takes nothing less.

### Tips for Implementing Algorithms

In his blog post, Edward provides 4 master tips that can help you implement machine learning algorithms from scratch. In summary, they are:

1. **Read the whole paper**. Read the whole paper, slowly. Then soak in the ideas for a while, say a few days. Read it again later, but not until you have a first-cut of your own mental model for how the algorithm works, the data flow and how it all hangs together. Read with purpose. Subsequent reads must correct and improve upon your existing understanding of the algorithm.
2. **Devise a test problem**. Locate, select or devise a test problem that is simple enough for you to understand and visualize the results or behavior of the algorithm, but complex enough to force the procedure to exhibit a differentiating characteristic or result. This problem will be your litmus test, telling you when the implementation is correct and when optimizations have not introduced fatal bugs. Edward calls it a “useful unit test of macro functionality“.
3. **Optimize last**. Understand the procedure and logic of the algorithm first by implementing the whole thing from scratch, leveraging little existing code or tricks. Only after you understand and have a working implementation should you consider improving performance in terms of space or time complexity, or accuracy with algorithm extensions.
4. **Understand the foundations**. When it comes to production-grade implementations, you can leverage existing libraries. Edward points to examples such as <a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">LIBSVM</a> and <a href="http://www.csie.ntu.edu.tw/~cjlin/liblinear/">LIBLINEAR</a>. These are powerful libraries that incorporate decades of bug fixing and optimizations. Before adopting them, be confident you understand exactly what it is you are leveraging, how it works and the characterize the benefits it provides. Optimize your implementations purposefully, use the best and understand what it does.

Again, there is great wisdom in these tips. I could not have put it better myself. In particular. I strongly agree with the need for implementing algorithms inefficiently from scratch to maximize learning. Algorithm optimization is an import but wholly different skill for a wholly different purpose.

Remember this.

### Avoid the Beginner Pitfalls

Edward goes on to highlight common traps that beginners fall into. In summary, they are:

- Don’t assume the research paper is correct, peer review is not perfect and mistakes (sometimes large ones) make into publications.
- Don’t try and get a math-free understanding of an algorithm, maths can describe salient parts of an algorithm process efficiently and unambiguously and this is critically important.
- Don’t start with other peoples source code, as described above.
- You cannot know how to apply an algorithm to a problem effectively a priori, look for transferable application ideas from similar papers.
- Default random number generates often don’t cut it, use something better, but not cryptographic strength.
- Scripting languages don’t cut the mustard when optimizing (His personal, and stated possibly controversial, opinion. I personally find static types save a lot of headache in large production systems).

## Summary

Implementing machine learning algorithms is an excellent (if not the best) way of learning machine learning. The knowledge is visceral because you have to sweat the details, they become intimate. This helps when you are trying to get the most from an algorithm.

In this post you discovered that the often suggested advice of “*read open source implementations*” is not wrong, but needs to fit carefully within your learning strategy.

Edward suggests that you learn machine learning algorithms *the hard way*, figure them out yourself so that you grow, then turn to open source implementations to learn the efficient mathematical and programatic tricks to optimize the implementation, if and when you need those efficiencies.

This is nuanced and valuable advice.
