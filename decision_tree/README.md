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
```

### Prediction

After training, the classifier is used to predict the gender for a new data point:

```python
prediction = clf.predict([[190, 70, 43]])
```

### Output

The result will be either `['male']` or `['female']`, depending on the learned model structure.

## 📊 Model Evaluation

Although the current script does not include a model evaluation pipeline, it's crucial to validate the performance of any machine learning model. Here's how you can evaluate a Decision Tree Classifier:

### 1. Train-Test Split

Split your data into training and test sets to evaluate generalization performance:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

### 2. Accuracy Score

Measure the proportion of correctly predicted labels:

```python
from sklearn.metrics import accuracy_score
predictions = clf.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, predictions))
```

### 3. Confusion Matrix

Provides insights into true positives, false positives, true negatives, and false negatives:

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, predictions))
```

### 4. Classification Report

Shows precision, recall, F1-score, and support for each class:

```python
from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))
```
> ⚠️ **Note**: With a small dataset like this, results may not be statistically robust.  
> For more reliable evaluation, it is highly recommended to:
> - Use **cross-validation** techniques (e.g., K-Fold Cross-Validation).
> - Train and test the model on a **larger, more diverse dataset**.

## ▶️ How to Run

1. Clone or download this repository.

2. Ensure Python 3 and `pip` are installed.

3. Install dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

4. Run the script:

##📦 Dependencies

- `scikit-learn`: Machine learning library for Python

- `numpy`: (implicitly used by scikit-learn)

## 📈 Future Enhancements

- Replace the handcrafted dataset with a larger, real-world dataset (e.g., from UCI or Kaggle).

- Evaluate model performance using accuracy, precision, recall, F1-score.

- Visualize the decision tree structure using `graphviz` or `matplotlib`.
