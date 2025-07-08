# Exploring Dictionaries, Classifying Variables, and Imputing Data in the Ames Dataset

The real estate market is a complex ecosystem driven by numerous variables such as location, property features, market trends, and economic indicators. One dataset that offers a deep dive into this complexity is the Ames Housing dataset. Originating from Ames, Iowa, this dataset comprises various properties and their characteristics, ranging from the type of alley access to the overall condition of the property.

In this post, you aim to take a closer look at this dataset using data science techniques. Specifically, you’ll focus on how to identify categorical and numerical variables, as understanding these variables is crucial for any data-driven decision-making process.

Let’s get started.

## Overview

This post is divided into three parts; they are:

- The Importance of a Data Dictionary
- Identifying Categorical and Numerical Variables
- Missing Data Imputation

## The Importance of a Data Dictionary

A crucial first step in analyzing the Ames Housing dataset is utilizing its <a href="https://jse.amstat.org/v19n3/decock/DataDocumentation.txt" target="_blank" rel="noopener">data dictionary</a>. This version does more than list and define the features; it categorizes them into **nominal**, **ordinal**, **discrete**, and **continuous** types, guiding our analysis approach.

- **Nominal Variables** are categories without an order like ‘Neighborhood’. They help in identifying segments for grouping analysis.
- **Ordinal Variables** have a clear order (e.g ‘KitchenQual’). They allow for ranking and order-based analysis but don’t imply equal spacing between categories.
- **Discrete Variables** are countable numbers, like ‘Bedroom’. They are integral to analyses that sum or compare quantities.
- **Continuous Variables** measure on a continuous scale, like ‘Lot Area’. They enable a wide range of statistical analyses that depend on granular detail.

Understanding these variable types also guides the selection of appropriate visualization techniques. **Nominal and ordinal variables** are well-suited to bar charts, which can effectively highlight categorical differences and rankings. In contrast, **discrete and continuous variables** are best represented through histograms, scatter plots, and line charts, which illustrate distributions, relationships, and trends within the data.

## Identifying Categorical and Numerical Variables

Building on our understanding of the data dictionary, let’s delve into how we can practically distinguish between categorical and numerical variables within the Ames dataset using Python’s pandas library. This step is crucial for informing our subsequent data processing and analysis strategies.

```python
# Load and obtain the data types from the Ames dataset
import pandas as pd
Ames = pd.read_csv('Ames.csv')

print(Ames.dtypes)
print(Ames.dtypes.value_counts())
```

Executing the above code will yield the following output, categorizing each feature by its data type:

```
PID                int64
GrLivArea          int64
SalePrice          int64
MSSubClass         int64
MSZoning          object
                  ...   
SaleCondition     object
GeoRefNo         float64
Prop_Addr         object
Latitude         float64
Longitude        float64
Length: 85, dtype: object
object     44
int64      27
float64    14
Name: count, dtype: int64
```

This output reveals that the dataset comprises object (44 variables), int64 (27 variables), and float64 (14 variables) data types. Here, object typically indicates nominal variables, which are categorical data without an inherent order. Meanwhile, int64 and float64 suggest numerical data, which could be either discrete (int64 for countable numbers) or continuous (float64 for measurable quantities on a continuous scale).

Now we can leverage pandas’ select_dtypes() method to explicitly separate numerical and categorical features within the Ames dataset.

```python
# Build on the above block of code
# Separating numerical and categorical features
numerical_features = Ames.select_dtypes(include=['int64', 'float64']).columns
categorical_features = Ames.select_dtypes(include=['object']).columns

# Displaying the separated lists
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)
```

The numerical_features captures variables stored as int64 and float64, indicative of countable and measurable quantities, respectively. Conversely, the categorical_features comprises variables of type object, typically representing nominal or ordinal data without a quantitative value:

```
Numerical Features: Index(['PID', 'GrLivArea', 'SalePrice', 'MSSubClass', 'LotFrontage', 'LotArea',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
       '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', 'GeoRefNo', 'Latitude', 'Longitude'],
      dtype='object')
Categorical Features: Index(['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition', 'Prop_Addr'],
      dtype='object')
```

Notably, some variables, like ‘MSSubClass’ despite being encoded numerically, actually serve as categorical data, underscoring the importance of referring back to our data dictionary for accurate classification. Similarly, features like ‘MoSold’ (Month Sold) and ‘YrSold’ (Year Sold) are numerical in nature, but they can often be treated as categorical variables, especially when there is no interest in performing mathematical operations on them. We can use the astype() method in pandas to convert these to categorical features.

```python
# Building on the above 2 blocks of code
Ames['MSSubClass'] = Ames['MSSubClass'].astype('object')
Ames['YrSold'] = Ames['YrSold'].astype('object')
Ames['MoSold'] = Ames['MoSold'].astype('object')
print(Ames.dtypes.value_counts())
```

After performing this conversion, the count of columns with the object data type has increased to 47 (from the previous 44), while int64 has dropped to 24 (from 27).

```
object     47
int64      24
float64    14
Name: count, dtype: int64
```

A careful assessment of the data dictionary, the nature of the dataset, and domain expertise can contribute to properly reclassifying data types.

## Missing Data Imputation

Dealing with missing data is a challenge that every data scientist faces. Ignoring missing values or handling them inadequately can lead to skewed analysis and incorrect conclusions. The choice of imputation technique often depends on the nature of the data—categorical or numerical. In addition, information in the data dictionary will be useful (such as the case for Pool Quality) where a missing value (“NA”) has a meaning, namely the absence of this feature for a particular property.

**Data Imputation For Categorical Features with Missing Values**

You can identify categorical data types and rank them in the order in which they are most affected by missing data.

```python
# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
data_type = Ames.dtypes

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage,
                             'Data Type':data_type})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values of 'object' data type
print(missing_info[(missing_info['Missing Values'] > 0) & (missing_info['Data Type'] == 'object')])
```

```
              Missing Values  Percentage Data Type
PoolQC                  2570   99.651028    object
MiscFeature             2482   96.238852    object
Alley                   2411   93.485847    object
Fence                   2054   79.643273    object
MasVnrType              1572   60.953858    object
FireplaceQu             1241   48.119426    object
GarageCond               129    5.001939    object
GarageFinish             129    5.001939    object
GarageQual               129    5.001939    object
GarageType               127    4.924389    object
BsmtExposure              71    2.753005    object
BsmtFinType2              70    2.714230    object
BsmtFinType1              69    2.675456    object
BsmtQual                  69    2.675456    object
BsmtCond                  69    2.675456    object
Prop_Addr                 20    0.775494    object
Electrical                 1    0.038775    object
```

The data dictionary indicates that missing values for the entire list of categorical features above indicate the absence of that feature for a given property, except for “Electrical”. With this insight, we can impute with the “mode” for the 1 missing data point for the electrical system and impute all others using "None" (with quotations to make it a Python string).

```python
# Building on the above block of code
# Imputing Missing Categorical Data

mode_value = Ames['Electrical'].mode()[0]
Ames['Electrical'].fillna(mode_value, inplace=True)

missing_categorical = missing_info[(missing_info['Missing Values'] > 0)
                           & (missing_info['Data Type'] == 'object')]

for item in missing_categorical.index.tolist():
    Ames[item].fillna("None", inplace=True)

print(Ames[missing_categorical.index].isnull().sum())
```

This confirms that there are now no more missing values for categorical features:

```
PoolQC          0
MiscFeature     0
Alley           0
Fence           0
MasVnrType      0
FireplaceQu     0
GarageCond      0
GarageFinish    0
GarageQual      0
GarageType      0
BsmtExposure    0
BsmtFinType2    0
BsmtFinType1    0
BsmtQual        0
BsmtCond        0
Prop_Addr       0
Electrical      0
dtype: int64
```

**Data Imputation For Numerical Features with Missing Values**

We can apply the same technique demonstrated above to identify numerical data types and rank them in the order in which they are most affected by missing data.

```python
# Build on the above blocks of code
# Import Numpy
import numpy as np

# Calculating the percentage of missing values for each column
missing_data = Ames.isnull().sum()
missing_percentage = (missing_data / len(Ames)) * 100
data_type = Ames.dtypes

# Combining the counts and percentages into a DataFrame for better visualization
missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage,
                             'Data Type':data_type})

# Sorting the DataFrame by the percentage of missing values in descending order
missing_info = missing_info.sort_values(by='Percentage', ascending=False)

# Display columns with missing values of numeric data type
print(missing_info[(missing_info['Missing Values'] > 0)
                   & (missing_info['Data Type'] == np.number)])
```

```
              Missing Values  Percentage Data Type
LotFrontage              462   17.913920   float64
GarageYrBlt              129    5.001939   float64
Longitude                 97    3.761148   float64
Latitude                  97    3.761148   float64
GeoRefNo                  20    0.775494   float64
MasVnrArea                14    0.542846   float64
BsmtFullBath               2    0.077549   float64
BsmtHalfBath               2    0.077549   float64
BsmtFinSF2                 1    0.038775   float64
GarageArea                 1    0.038775   float64
BsmtFinSF1                 1    0.038775   float64
BsmtUnfSF                  1    0.038775   float64
TotalBsmtSF                1    0.038775   float64
GarageCars                 1    0.038775   float64
```

The above illustrates that there are fewer instances of missing numerical data versus missing categorical data. However, the data dictionary is not as useful for a straightforward imputation. Whether or not to impute missing data in data science largely depends on the goal of the analysis. Often, a data scientist may generate multiple imputations to account for the uncertainty in the imputation process. Common multiple imputation methods include (but are not limited to) mean, median, and regression imputation. As a baseline, we will illustrate how to employ mean imputation here, but may refer to other techniques depending on the task at hand.

```python
# Build on the above blocks of code
# Initialize a DataFrame to store the concise information
concise_info = pd.DataFrame(columns=['Feature', 'Missing Values After Imputation', 
                                     'Mean Value Used to Impute'])

# Identify and impute missing numerical values, and store the related concise information
missing_numeric_df = missing_info[(missing_info['Missing Values'] > 0)
                           & (missing_info['Data Type'] == np.number)]

for item in missing_numeric_df.index.tolist():
    mean_value = Ames[item].mean(skipna=True)
    Ames[item].fillna(mean_value, inplace=True)

    # Append the concise information to the concise_info DataFrame
    concise_info.loc[len(concise_info)] = pd.Series({
        'Feature': item,
        'Missing Values After Imputation': Ames[item].isnull().sum(),
        # This should be 0 as we are imputing all missing values
        'Mean Value Used to Impute': mean_value
    })

# Display the concise_info DataFrame
print(concise_info)
```

Output:

```
         Feature Missing Values After Imputation Mean Value Used to Impute
0    LotFrontage                               0                 68.510628
1    GarageYrBlt                               0               1976.997143
2      Longitude                               0                -93.642535
3       Latitude                               0                 42.034556
4       GeoRefNo                               0          713676171.462681
5     MasVnrArea                               0                 99.346979
6   BsmtFullBath                               0                   0.43539
7   BsmtHalfBath                               0                  0.062088
8     BsmtFinSF2                               0                 53.259503
9     GarageArea                               0                466.864624
10    BsmtFinSF1                               0                444.285105
11     BsmtUnfSF                               0                539.194725
12   TotalBsmtSF                               0               1036.739333
13    GarageCars                               0                  1.747867
```

At times, we may also opt to leave the missing value without any imputation to retain the authenticity of the original dataset and remove the observations that do not have complete and accurate data if required. Alternatively, you may also try to build a machine learning model to **guess** the missing value based on some other data in the same rows, which is the principle behind imputation by regression. As a final step of the above baseline imputation, let us cross-check if there are any missing values.

```python
# Build on the above blocks of code
missing_values_count = Ames.isnull().sum().sum()
print(f'The DataFrame has a total of {missing_values_count} missing values.')
```

Output:

```
The DataFrame has a total of 0 missing values.
```

Congratulations! We have successfully imputed every missing value in the Ames dataset using baseline operations. It’s important to note that numerous other techniques exist for imputing missing data. As a data scientist, exploring various options and determining the most appropriate method for the given context is crucial to producing reliable and meaningful results.

## Further Reading

### Resources

- <a href="https://raw.githubusercontent.com/Padre-Media/dataset/main/Ames.csv" target="_blank" rel="noopener">Ames Dataset</a>
- <a href="https://jse.amstat.org/v19n3/decock/DataDocumentation.txt" target="_blank" rel="noopener">Ames Data Dictionary (Expanded Version)</a>

## Summary

In this tutorial, we explored the Ames Housing dataset through the lens of data science techniques. We discussed the importance of a data dictionary in understanding the dataset’s variables and dove into Python code snippets that help identify and handle these variables effectively.

Understanding the nature of the variables you’re working with is crucial for any data-driven decision-making process. As we’ve seen, the Ames data dictionary serves as a valuable guide in this respect. Coupled with Python’s powerful data manipulation libraries, navigating complex datasets like the Ames Housing dataset becomes a much more manageable task.

Specifically, you learned: 

- The importance of a data dictionary when assessing data types and imputation strategies.
- Identification and reclassification methods for numerical and categorical features.
- How to impute missing categorical and numerical features using the pandas library.

## Reference

- <a href="https://github.com/QiRi92/data_science/blob/main/data_science/2_explore_classify_impute.ipynb" rel="noopener" target="_blank">Codes</a>
