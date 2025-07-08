# Testing Assumptions in Real Estate: A Dive into Hypothesis Testing with the Ames Housing Dataset

In doing inferential statistics, you often want to test your assumptions. Indeed there is a way to quantitatively test an assumption that you thought of. Using the Ames Housing dataset, you’ll delve deep into the concept of hypothesis testing and explore if the presence of an air conditioner affects the sale price of a house.

Let’s get started.

## Overview

This post unfolds through the following segments:

- The Role of Hypothesis Testing in Inferential Statistics.
- How does Hypothesis Testing work?
- Does Air Conditioning Affect Sale Price?

## The Role of Hypothesis Testing in Inferential Statistics

Inferential Statistics uses a sample of data to make inferences about the population from which it was drawn. Hypothesis testing, a fundamental component of inferential statistics, is crucial when making informed decisions about a population based on sample data, especially when studying the entire population is unfeasible. Hypothesis testing is a way to make a statement about the data.

Imagine you’ve come across a claim stating that houses with air conditioners sell at a higher price than those without. To verify this claim, you’d gather data on house sales and analyze if there’s a significant difference in prices based on the presence of air conditioning. This process of testing claims or assumptions about a population using sample data is known as hypothesis testing. In essence, hypothesis testing allows us to make an informed decision (either rejecting or failing to reject a starting assumption) based on evidence from the sample and the likelihood that the observed effect occurred by chance.

## How does Hypothesis Testing work?

- **Null Hypothesis** (*H*0): The **default state** of no effect or no different. A statement that you aim to test against.
- **Alternative Hypothesis** (*H*a): What you want to prove. It is what you believe if the null hypothesis is wrong.
- **Test Statistic**: A value computed from the sample data that’s used to test the null hypothesis.
- **P-value**: The probability that the observed effect in the sample occurred by random chance under the null hypothesis situation.

Performing hypothesis testing is like a detective: Ordinarily, you assume something should happen (*H*0), but you suspect something else is actually happening (*H*1). Then you collect your evidence (the test statistic) to argue why *H*0 is not reasonable; hence *H*1 should be the truth.

In a typical hypothesis test:

1. You state the null and alternative hypotheses. You should carefully design these hypotheses to reflect a reasonable assumption about the reality.
2. You choose a significance level (a); it is common to use a=0.05 in statistical hypothesis tests.
3. You collect and analyze the data to get our test statistic and p-value, based on the situation of *H*0.
4. You make a decision based on the p-value: You reject the null hypothesis and accept the alternative hypothesis if and only if the p-value is less than a.

## Does Air Conditioning Affect Sales Price?

Based on the <a href="https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv" target="_blank" rel="noopener">Ames Dataset</a>, we want to know if the presence of air conditioning can affect the price.

To explore the impact of air conditioning on sales prices, you’ll set our hypotheses as:

- *H*0: The average sales price of houses with air conditioning is the same as those without.
- *H*a: The average sales price of houses with air conditioning is not the same as those without.

Before performing the hypothesis test, let’s visualize our data to get a preliminary understanding.

```python
# Loading the dataset and essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Ames = pd.read_csv('Ames.csv')

# Data separation
ac_prices = Ames[Ames['CentralAir'] == 'Y']['SalePrice']
no_ac_prices = Ames[Ames['CentralAir'] == 'N']['SalePrice']

# Setting up the visualization
plt.figure(figsize=(10, 6))

# Histograms for sale prices based on air conditioning
# Plotting 'With AC' first for the desired order in the legend
plt.hist(ac_prices, bins=30, alpha=0.7, color='blue', edgecolor='blue', lw=0.5,
         label='Sales Prices With AC')
mean_ac = np.mean(ac_prices)
plt.axvline(mean_ac, color='blue', linestyle='dashed', linewidth=1.5,
            label=f'Mean (With AC): ${mean_ac:.2f}')

plt.hist(no_ac_prices, bins=30, alpha=0.7, color='red', edgecolor='red', lw=0.5,
         label='Sales Prices Without AC')
mean_no_ac = np.mean(no_ac_prices)
plt.axvline(mean_no_ac, color='red', linestyle='dashed', linewidth=1.5,
            label=f'Mean (Without AC): ${mean_no_ac:.2f}')

plt.title('Distribution of Sales Prices based on Presence of Air Conditioning', fontsize=18)
plt.xlabel('Sales Price', fontsize=15)
plt.ylabel('Number of Houses', fontsize=15)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```

Output:

<img width="584" alt="image" src="https://github.com/user-attachments/assets/81f10689-8efa-420b-b79c-e441802fd590" />

The code above called plt.hist() twice with different data to show two overlapped histograms, one for the distribution of sales price with air conditioning (AC) and one without. Here are a few observations that can be made from the visual:

- **Distinct Peaks**: Both distributions exhibit a **distinct** peak, which indicates the most frequent sale prices in their respective categories.
- **Mean Sale Price**: The mean sale price of houses with AC is higher than that of houses without AC, as indicated by the vertical dashed lines.
- **Spread and Skewness**: The distribution of sale prices for houses with AC appears slightly right-skewed, indicating that while most houses are sold at a lower price, there are some properties with significantly higher prices. In contrast, the distribution for houses without AC is more compact, with a smaller range of prices.
- **Overlap**: Despite the differences in means, there’s an overlap in the price range of houses with and without AC. This suggests that while AC may influence price, other factors are also at play in determining a house’s value.

Given these insights, the presence of AC seems to be associated with a higher sale price. The next step would be to perform the hypothesis test to numerically determine if this difference is significant.

```python
# Import an additional library
import scipy.stats as stats

# Performing a two-sample t-test
t_stat, p_value = stats.ttest_ind(ac_prices, no_ac_prices, equal_var=False)

# Printing the results
if p_value < 0.05:
    result = "reject the null hypothesis"
else:
    result = "fail to reject the null hypothesis"
print(f"With a p-value of {p_value:.5f}, we {result}.")
```

This shows:

```
With a p-value of 0.00000, we reject the null hypothesis.
```

The p-value is less than a. The p-value says that it is very unlikely, under *H*0, that the difference in the price is by chance. This indicates that there’s a statistically significant difference in the average sale prices of houses with air conditioning compared to those without. This aligns with our visual observations from the histogram. Thus, the presence of an air conditioner does seem to have a significant effect on the sale price of houses in the Ames dataset.

This p-value is computed using t-test. It is a statistic aimed at comparing the **means of two groups**. There are many statistics available, and t-test is a suitable one here because our hypotheses *H*0, *H*a are about the average sales price.

Note that the alternative hypothesis *H*a defined above can be changed. You can make it mean “the average sales price of houses with air conditioning is **less than** those without”; however, this is counter-intuitive to the reality.  You can also make it mean “the average sales price of houses with air conditioning is **more than** those without”; which you should change the t-test in the code to include the extra argument alternative='greater':

```python
# Performing a one-sided t-test
t_stat, p_value = stats.ttest_ind(ac_prices, no_ac_prices, equal_var=False, alternative='greater')

# Printing the results
if p_value < 0.05:
    result = "reject the null hypothesis"
else:
    result = "fail to reject the null hypothesis"
print(f"With a p-value of {p_value:.5f}, we {result}.")
```

Output:

```
With a p-value of 0.00000, we reject the null hypothesis.
```

This changes the two-sided t-test to one-sided t-test, but the resulting outcome is the same. Switching from a two-sided to a one-sided t-test but arriving at the same conclusion implies that we had a clear expectation of the direction of the difference from the start, or the data strongly supported one direction of difference, making the outcome consistent across both test types.

The setup of the null hypothesis (*H*0) and alternative hypothesis (*H*a) is fundamental to the design of statistical tests, influencing the test’s directionality (one-sided vs. two-sided), the interpretation of results (how we understand p-values and evidence), and decision-making processes (especially when the p-value is close to the significance level a). This framework determines not only what we are testing for but also how we interpret and act on the statistical evidence obtained.

### Resources

- <a href="https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv" target="_blank" rel="noopener">Ames Dataset</a>
- <a href="https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt" target="_blank" rel="noopener">Ames Data Dictionary</a>

## Summary

In this exploration, you delved into the world of hypothesis testing using the Ames Housing dataset. You examined how the presence of an air conditioner might impact the sale price of a house. Through rigorous statistical testing, you found that houses with air conditioning tend to have a higher sale price than those without, a result that holds statistical significance. This not only underscores the importance of amenities like air conditioning in the real estate market but also showcases the power of hypothesis testing in making informed decisions based on data.

Specifically, you learned:

- The importance of hypothesis testing within inferential statistics.
- How to set up and evaluate null and alternative hypothesis using detailed methods of hypothesis testing.
- The practical implications of hypothesis testing in real-world scenarios, exemplified by the presence of air conditioning on property values in the Ames housing market.

## Reference

- <a href="https://github.com/QiRi92/data_science/blob/main/data_science/3_testing_assumptions.ipynb" rel="noopener" target="_blank">Codes</a>
