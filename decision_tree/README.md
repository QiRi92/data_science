# Decision Tree Classifier for Gender Prediction

This project demonstrates a simple yet effective example of using a **Decision Tree Classifier** from the `scikit-learn` library to predict gender based on physical attributes such as height, weight, and shoe size. It is implemented in the file `intro.py`.

## 🧠 Project Overview

**Machine Learning Algorithm**: Decision Tree  
**Input Features**:
- Height (in cm)
- Weight (in kg)
- Shoe Size (EU)

**Target Variable**: Gender (`'male'` or `'female'`)

This project showcases the classification power of a decision tree—a non-parametric supervised learning method used for classification and regression tasks. Decision trees are favored for their interpretability and ability to handle both numerical and categorical data without feature scaling or normalization.

## 🧪 Dataset Description

The dataset used in this script is a small, handcrafted dataset designed for illustrative purposes:

| Height (cm) | Weight (kg) | Shoe Size | Gender  |
|-------------|-------------|------------|----------|
| 181         | 80          | 44         | male     |
| 177         | 70          | 43         | female   |
| 160         | 60          | 38         | female   |
| 154         | 54          | 37         | female   |
| 166         | 65          | 40         | male     |
| 190         | 90          | 47         | male     |
| 175         | 64          | 39         | male     |
| 177         | 70          | 40         | female   |
| 159         | 55          | 37         | male     |
| 171         | 75          | 42         | female   |
| 181         | 85          | 43         | male     |

Although the dataset is small, it is sufficient to illustrate the decision boundary learning capability of a decision tree classifier.

## 🔍 Technical Details

### What is a Decision Tree?

A **Decision Tree** splits the dataset into branches based on feature thresholds that lead to the highest information gain (using metrics like Gini impurity or entropy). It recursively builds branches until it reaches leaves that represent a final class label.

### Why Decision Trees?

- **No need for feature scaling or normalization**
- **Handles non-linear relationships well**
- **Easily interpretable and visualizable**
- **Performs well on small to medium-sized datasets**

### Model Training

The classifier is trained using the `fit()` method on feature matrix `X` and label vector `Y`:

```python
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
