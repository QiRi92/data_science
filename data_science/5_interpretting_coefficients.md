# Interpreting Coefficients in Linear Regression Models

Linear regression models are foundational in machine learning. Merely fitting a straight line and reading the coefficient tells a lot. But how do we extract and interpret the coefficients from these models to understand their impact on predicted outcomes? This post will demonstrate how one can interpret coefficients by exploring various scenarios. We’ll explore the analysis of a single numerical feature, examine the role of categorical variables, and unravel the complexities introduced when these features are combined. Through this exploration, we aim to equip you with the skills needed to leverage linear regression models effectively, enhancing your analytical capabilities across different data-driven domains.

##Overview

This post is divided into three parts; they are:

- Interpreting Coefficients in Linear Models with a Single Numerical Feature
- Interpreting Coefficients in Linear Models with a Single Categorical Feature
- Discussion on Combining Numerical and Categorical Features

## Interpreting Coefficients in Linear Models with a Single Numerical Feature

In this section, we focus on a single numerical feature from the Ames Housing dataset, “GrLivArea” (above-ground living area in square feet), to understand its direct impact on “SalePrice”. We employ K-Fold Cross-Validation to validate our model’s performance and extract the coefficient of “GrLivArea”. This coefficient estimates how much the house price is expected to increase for every additional square foot of living area under the assumption that all other factors remain constant. This is a fundamental aspect of linear regression analysis, ensuring that the effect of “GrLivArea” is isolated from other variables.

Here is how we set up our regression model to achieve this:

```python
# Set up to obtain CV model performance and coefficient using K-Fold
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

Ames = pd.read_csv("Ames.csv")
X = Ames[["GrLivArea"]].values  # get 2D matrix
y = Ames["SalePrice"].values    # get 1D vector

model = LinearRegression()
kf = KFold(n_splits=5)
coefs = []
scores = []

# Manually perform K-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model, obtain fold performance and coefficient
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    coefs.append(model.coef_)

mean_score = np.mean(scores)
print(f"Mean CV R² = {mean_score:.4f}")

mean_coefs = np.mean(coefs)
print(f"Mean Coefficient = {mean_coefs:.4f}")
```

The output from this code block provides two key pieces of information: the mean R² score across the folds and the mean coefficient for “GrLivArea.” The R² score gives us a general idea of how well our model fits the data across different subsets, indicating the model’s consistency and reliability. Meanwhile, the mean coefficient quantifies the average effect of “GrLivArea” on “SalePrice” across all the validation folds.

```
Mean CV R² = 0.5127
Mean Coefficient = 110.5214
```

The coefficient of “GrLivArea” can be directly interpreted as the price change per square foot. Specifically, it indicates that for each square foot increase in “GrLivArea,” the sale price of the house is expected to rise by approximately $110.52 (not to be confused with the price per square foot since the coefficient refers to the **marginal price**). Conversely, a decrease in living area by one square foot would typically lower the sale price by the same amount.

## Interpreting Coefficients in Linear Models with a Single Categorical Feature

While numerical features like “GrLivArea” can be directly used in our regression model, categorical features require a different approach. Proper encoding of these categorical variables is crucial for accurate model training and ensuring the results are interpretable. In this section, we’ll explore One Hot Encoding—a technique that prepares categorical variables for linear regression by transforming them into a format that is interpretable within the model’s framework. We will specifically focus on how to interpret the coefficients that result from these transformations, including the strategic selection of a reference category to simplify these interpretations.

Choosing an appropriate reference category when applying One Hot Encoding is crucial as it sets the baseline against which other categories are compared. This baseline category’s mean value often serves as the intercept in our regression model. Let’s explore the distribution of sale prices across neighborhoods to select a reference category that will make our model both interpretable and meaningful:

```python
# Rank neighborhoods by their mean sale price
Ames = pd.read_csv("Ames.csv")
neighbor_stats = Ames.groupby("Neighborhood")["SalePrice"].agg(["count", "mean"]).sort_values(by="mean")
print(neighbor_stats.round(0).astype(int))
```

This output will inform our choice by highlighting the neighborhoods with the lowest and highest average prices, as well as indicating the neighborhoods with sufficient data points (count) to ensure robust statistical analysis:

```
              count    mean
Neighborhood               
MeadowV          34   96836
BrDale           29  106095
IDOTRR           76  108103
BrkSide         103  126030
OldTown         213  126939
Edwards         165  133152
SWISU            42  133576
Landmrk           1  137000
Sawyer          139  137493
NPkVill          22  140743
Blueste          10  143590
NAmes           410  145087
Mitchel         104  162655
SawyerW         113  188102
Gilbert         143  189440
NWAmes          123  190372
Greens            8  193531
Blmngtn          23  196237
CollgCr         236  198133
Crawfor          92  202076
ClearCr          40  213981
Somerst         143  228762
Timber           54  242910
Veenker          23  251263
GrnHill           2  280000
StoneBr          43  305308
NridgHt         121  313662
NoRidge          67  326114
```

Choosing a neighborhood like “MeadowV” as our reference sets a clear baseline, interpreting other neighborhoods’ coefficients straightforward: they show how much more expensive houses are than “MeadowV”.

Having identified “MeadowV” as our reference neighborhood, we are now ready to apply One Hot Encoding to the “Neighborhood” feature, explicitly excluding “MeadowV” to establish it as our baseline in the model. This step ensures that all subsequent neighborhood coefficients are interpreted in relation to “MeadowV,” providing a clear comparative analysis of house pricing across different areas. The next block of code will demonstrate this encoding process, fit a linear regression model using K-Fold cross-validation, and calculate the average coefficients and Y-intercept. These calculations will help quantify the additional value or deficit associated with each neighborhood compared to our baseline, offering actionable insights for market evaluation.

```python
# Build on initial set up and block of code above
# Import OneHotEncoder to preprocess a categorical feature
from sklearn.preprocessing import OneHotEncoder

# One Hot Encoding for "Neighborhood", Note: drop=["MeadowV"]

# For scikit-learn >= 1.2
encoder = OneHotEncoder(sparse_output=False, drop=["MeadowV"])

# For scikit-learn < 1.2 (deprecated)
# encoder = OneHotEncoder(sparse=False, drop=["MeadowV"])

X = encoder.fit_transform(Ames[["Neighborhood"]])
y = Ames["SalePrice"].values

# Setup KFold and initialize storage
kf = KFold(n_splits=5)
scores = []
coefficients = []
intercept = []

# Perform the KFold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Append the results for each fold
    scores.append(model.score(X_test, y_test))
    coefficients.append(model.coef_)
    intercept.append(model.intercept_)

mean_score = np.mean(scores)
print(f"Mean CV R² = {mean_score:.4f}")
mean_coefficients = np.mean(coefficients, axis=0)
mean_intercept = np.mean(intercept)
print(f"Mean Y-intercept = {mean_intercept:.0f}")

# Retrieve neighborhood names from the encoder, adjusting for the dropped category
neighborhoods = encoder.categories_[0]
if "MeadowV" in neighborhoods:
    neighborhoods = [name for name in neighborhoods if name != "MeadowV"]

# Create a DataFrame to nicely display neighborhoods with their average coefficients
import pandas as pd

coefficients_df = pd.DataFrame({
    "Neighborhood": neighborhoods,
    "Average Coefficient": mean_coefficients.round(0).astype(int)
})

# Print or return the DataFrame
print(coefficients_df.sort_values(by="Average Coefficient").reset_index(drop=True))
```

The mean R² will remain consistent at 0.5408 regardless of what feature we “dropped” when we One Hot Encoded.

The Y-intercept provides a specific quantitative benchmark. Representing the average sale price in “MeadowV,” this Y-intercept forms the foundational price level against which all other neighborhoods’ premiums or discounts are measured.

<img width="169" alt="image" src="https://github.com/user-attachments/assets/5df0d606-b51a-4fa3-b38f-07247e1a7745" />

Each neighborhood’s coefficient, calculated relative to “MeadowV,” reveals its premium or deficit in house pricing. By setting “MeadowV” as the reference category in our One Hot Encoding process, its average sale price effectively becomes the intercept of our model. The coefficients calculated for other neighborhoods then measure the difference in expected sale prices relative to “MeadowV.” For instance, a positive coefficient for a neighborhood indicates that houses there are more expensive than those in “MeadowV” by the coefficient’s value, assuming all other factors are constant. This arrangement allows us to directly assess and compare the impact of different neighborhoods on the “SalePrice,” providing a clear and quantifiable understanding of each neighborhood’s relative market value.

## Discussion on Combining Numerical and Categorical Features

So far, we have examined how numerical and categorical features influence our predictions separately. However, real-world data often require more sophisticated models that can handle multiple types of data simultaneously to capture the complex relationships within the market. To achieve this, it is essential to become familiar with tools like the ColumnTransformer, which allows for the simultaneous processing of different data types, ensuring that each feature is optimally prepared for modeling. Let’s now demonstrate an example where we combine the living area (“GrLivArea”) with the neighborhood classification to see how these factors together affect our model performance.

```python
# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
Ames = pd.read_csv("Ames.csv")

# Select features and target
features = Ames[["GrLivArea", "Neighborhood"]]
target = Ames["SalePrice"]

# Preprocess features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["GrLivArea"]),
        ("cat", OneHotEncoder(sparse_output=False, drop=["MeadowV"], handle_unknown="ignore"), ["Neighborhood"])
    ])

# Fit and transform the features
X_transformed = preprocessor.fit_transform(features)
feature_names = ["GrLivArea"] + list(preprocessor.named_transformers_["cat"].get_feature_names_out())

# Initialize KFold
kf = KFold(n_splits=5)

# Initialize variables to store results
coefficients_list = []
intercepts_list = []
scores = []

# Perform the KFold cross-validation
for train_index, test_index in kf.split(X_transformed):
    X_train, X_test = X_transformed[train_index], X_transformed[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Store coefficients and intercepts
    coefficients_list.append(model.coef_)
    intercepts_list.append(model.intercept_)

    # Evaluate the model
    scores.append(model.score(X_test, y_test))

# Calculate the mean of scores, coefficients, and intercepts
average_score = np.mean(scores)
average_coefficients = np.mean(coefficients_list, axis=0)
average_intercept = np.mean(intercepts_list)

# Display the average R² score and Y-Intercept across all folds
# The Y-Intercept represents the baseline sale price in "MeadowV" with no additional living area
print(f"Mean CV R² Score of Combined Model: {average_score:.4f}")
print(f"Mean Y-intercept = {average_intercept:.0f}")

# Create a DataFrame for the coefficients
df_coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Average Coefficient": average_coefficients
    }).sort_values(by="Average Coefficient").reset_index(drop=True)

# Display the DataFrame
print("Coefficients for Combined Model:")
print(df_coefficients)
```

The code above should output:

```
Mean CV R² Score of Combined Model: 0.7375
Mean Y-intercept = 11786
Coefficients for Combined Model:
                 Feature  Average Coefficient
0     Neighborhood_SWISU         -3728.929853
1    Neighborhood_IDOTRR         -1498.971239
2              GrLivArea            78.938757
3   Neighborhood_OldTown          2363.805796
4    Neighborhood_BrDale          6551.114637
5   Neighborhood_BrkSide         16521.117849
6   Neighborhood_Landmrk         16921.529665
7   Neighborhood_Edwards         17520.110407
8   Neighborhood_NPkVill         30034.541748
9     Neighborhood_NAmes         31717.960146
10   Neighborhood_Sawyer         32009.140024
11  Neighborhood_Blueste         39908.310031
12   Neighborhood_NWAmes         44409.237736
13  Neighborhood_Mitchel         48013.229999
14  Neighborhood_SawyerW         48204.606372
15  Neighborhood_Gilbert         49255.248193
16  Neighborhood_Crawfor         55701.500795
17  Neighborhood_ClearCr         61737.497483
18  Neighborhood_CollgCr         69781.161291
19  Neighborhood_Blmngtn         72456.245569
20  Neighborhood_Somerst         90020.562168
21   Neighborhood_Greens         90219.452164
22   Neighborhood_Timber         97021.781128
23  Neighborhood_Veenker         98829.786236
24  Neighborhood_NoRidge        120717.748175
25  Neighborhood_StoneBr        147811.849406
26  Neighborhood_NridgHt        150129.579392
27  Neighborhood_GrnHill        157858.199004
```

Combining “GrLivArea” and “Neighborhood” into a single model has significantly improved the R² score, rising to 0.7375 from the individual scores of 0.5127 and 0.5408, respectively. This substantial increase illustrates that integrating multiple data types provides a more accurate reflection of the complex factors influencing real estate prices.

However, this integration introduces new complexities into the model. The interaction effects between features like “GrLivArea” and “Neighborhood” can significantly alter the coefficients. For instance, the coefficient for “GrLivArea” decreased from 110.52 in the single-feature model to 78.93 in the combined model. This change illustrates how the value of living area is influenced by the characteristics of different neighborhoods. Incorporating multiple variables requires adjustments in the coefficients to account for overlapping variances between predictors, resulting in coefficients that often differ from those in single-feature models.

The mean Y-intercept calculated for our combined model is $11,786. This value represents the predicted sale price for a house in the “MeadowV” neighborhood with the base living area (as accounted for by “GrLivArea”) adjusted to zero. This intercept serves as a foundational price point, enhancing our interpretation of how different neighborhoods compare to “MeadowV” in terms of cost, once adjusted for the size of the living area. Each neighborhood’s coefficient, therefore, informs us about the additional cost or savings relative to our baseline, “MeadowV,” providing clear and actionable insights into the relative value of properties across different areas.

## Further Reading

### APIs
- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html" target="_blank" rel="noopener">sklearn.compose.ColumnTransformer</a> API

### Tutorials
- <a href="https://www.theanalysisfactor.com/interpreting-regression-coefficients/" target="_blank" rel="noopener">Interpreting Regression Coefficients</a>

- <a href="https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv" target="_blank" rel="noopener">Ames Dataset</a>
- <a href="https://github.com/Padre-Media/dataset/blob/main/Ames%20Data%20Dictionary.txt" target="_blank" rel="noopener">Ames Data Dictionary</a>

## Summary

This post has guided you through interpreting coefficients in linear regression models with clear, practical examples using the Ames Housing dataset. We explored how different types of features—numerical and categorical—affect the predictability and clarity of models. Moreover, we addressed the challenges and benefits of combining these features, especially in the context of interpretation.

Specifically, you learned:

- **The Direct Impact of Single Numerical Features**: How the “GrLivArea” coefficient directly quantifies the increase in “SalePrice” for each additional square foot, providing a clear measure of its predictive value in a straightforward model.
- **Handling Categorical Variables**: The importance of One Hot Encoding in dealing with categorical features like “Neighborhood”, illustrating how choosing a baseline category impacts the interpretation of coefficients and sets a foundation for comparison across different areas.
- **Combining Features to Enhance Model Performance**: The integration of “GrLivArea” and “Neighborhood” not only improved the predictive accuracy (R² score) but also introduced a complexity that affects how each feature’s coefficient is interpreted. This part emphasized the trade-off between achieving high predictive accuracy and maintaining model interpretability, which is crucial for making informed decisions in the real estate market.

## Reference

- <a href="https://github.com/QiRi92/data_science/blob/main/data_science/5_interpretting_coefficients.ipynb" rel="noopener" target="_blank">Codes</a>
