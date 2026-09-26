# DS605: Fundamentals of Machine Learning

## Lab Assignment 5 — Machine Learning with Scikit-learn and From Scratch

### Student Information

**Course:** DS605 – Fundamentals of Machine Learning
**Lab:** Lab Assignment – 5
**Dataset:** UCI Productivity Prediction of Garment Employees
**Programming Language:** Python
**Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib

---

## 1. Objective

The objective of this lab is to implement and compare Machine Learning workflows using:

1. **Scikit-learn**
2. **From-scratch implementation using NumPy and Pandas**

Two Machine Learning tasks are performed:

* **Regression:** Predict `actual_productivity` using Linear Regression.
* **Classification:** Predict whether an employee/team meets the productivity target using Logistic Regression.

The implementations are compared based on:

* Predictive performance
* Training time
* Prediction time
* Implementation efficiency

The manual implementation is further optimized using vectorization, learning-rate tuning, convergence control, and regularization.

---

## 2. Dataset

The dataset used is the **Productivity Prediction of Garment Employees** dataset from the UCI Machine Learning Repository.

The dataset contains information related to garment production, including:

* Department
* Quarter
* Day
* Team
* Targeted productivity
* Standard minute value (SMV)
* Work in progress (WIP)
* Overtime
* Incentive
* Idle time
* Idle workers
* Number of style changes
* Number of workers
* Actual productivity

The main target for regression is:

```text
actual_productivity
```

---

## 3. Machine Learning Tasks

### 3.1 Regression

The regression task predicts:

```text
actual_productivity
```

using Linear Regression.

The following metrics are used:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

A lower MAE and RMSE indicate lower prediction error, while a higher R² indicates better explanatory performance.

---

### 3.2 Classification

A new binary target called `MeetsTarget` is created.

The definition is:

```text
MeetsTarget = 1
if actual_productivity >= targeted_productivity

MeetsTarget = 0
otherwise
```

Logistic Regression is then used to predict `MeetsTarget`.

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-score

### Important

`actual_productivity` is **not used as an input feature for classification**, because `MeetsTarget` is directly derived from `actual_productivity`. Using it would result in target leakage.

---

## 4. Project Workflow

The overall workflow is:

```text
Raw Dataset
     |
     v
Data Inspection
     |
     v
Missing Value Handling
     |
     v
Categorical Encoding
     |
     v
Feature Scaling
     |
     v
Fixed Train-Test Split
     |
     +-------------------------+
     |                         |
     v                         v
Regression               Classification
     |                         |
     v                         v
Linear Regression       Logistic Regression
     |                         |
     v                         v
Prediction              Prediction
     |                         |
     v                         v
MAE, RMSE, R²          Accuracy, Precision,
                       Recall, F1
     |                         |
     +------------+------------+
                  |
                  v
             Comparison
                  |
                  v
             Optimization
```

---

## 5. Part A — Scikit-learn Implementation

The first implementation uses Scikit-learn.

### Preprocessing

The following preprocessing operations are performed:

* Missing numerical values are replaced using the median.
* Missing categorical values are replaced using the most frequent value.
* Categorical features are converted using One-Hot Encoding.
* Numerical features are standardized using StandardScaler.

Scikit-learn `Pipeline` and `ColumnTransformer` are used to organize the preprocessing workflow.

### Regression Model

```text
LinearRegression
```

### Classification Model

```text
LogisticRegression
```

Training and prediction times are recorded separately for both models.

---

## 6. Part B — From-Scratch Implementation

The second implementation recreates the Machine Learning workflow using only:

```python
NumPy
Pandas
```

Scikit-learn preprocessing, models, metrics, and train-test utilities are not used in this section.

The same fixed train-test samples from Part A are reused.

### Manual preprocessing includes:

1. Missing-value handling
2. Categorical encoding
3. Feature scaling
4. Conversion to NumPy arrays

### Manual Linear Regression

Linear Regression is implemented using the closed-form least-squares solution:

$$
\beta = (X^TX)^{-1}X^Ty
$$

A pseudoinverse is used for numerical stability:

```python
beta = np.linalg.pinv(X.T @ X) @ X.T @ y
```

Predictions are calculated using:

```python
y_pred = X @ beta
```

### Manual Logistic Regression

Logistic Regression is implemented using gradient descent.

The sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

The implementation includes:

* Weight initialization
* Bias initialization
* Sigmoid probability calculation
* Binary cross-entropy loss
* Gradient calculation
* Parameter updates
* Probability prediction
* Threshold-based classification

The default classification threshold is:

```text
0.5
```

---

## 7. Manual Evaluation Metrics

All required evaluation metrics are manually implemented.

### Regression

#### MAE

$$
MAE=\frac{1}{n}\sum |y_i-\hat{y_i}|
$$

#### RMSE

$$
RMSE=
\sqrt{\frac{1}{n}\sum(y_i-\hat{y_i})^2}
$$

#### R²

$$
R^2 =
1-
\frac{\sum(y_i-\hat{y_i})^2}
{\sum(y_i-\bar{y})^2}
$$

### Classification

The following values are calculated manually:

* True Positive
* True Negative
* False Positive
* False Negative

These are then used to calculate:

$$
Accuracy=
\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
Precision=
\frac{TP}{TP+FP}
$$

$$
Recall=
\frac{TP}{TP+FN}
$$

$$
F1=
\frac{2(Precision)(Recall)}
{Precision+Recall}
$$

---

## 8. Part C — Comparison and Optimization

The Scikit-learn and manual implementations are compared using the same:

* Train-test split
* Input features
* Target definitions
* Evaluation metrics

### Regression comparison

| Implementation      | MAE | RMSE | R² | Training Time | Prediction Time |
| ------------------- | --: | ---: | -: | ------------: | --------------: |
| Scikit-learn        |   — |    — |  — |             — |               — |
| Manual NumPy/Pandas |   — |    — |  — |             — |               — |

### Classification comparison

| Implementation      | Accuracy | Precision | Recall | F1 | Training Time | Prediction Time |
| ------------------- | -------: | --------: | -----: | -: | ------------: | --------------: |
| Scikit-learn        |        — |         — |      — |  — |             — |               — |
| Manual NumPy/Pandas |        — |         — |      — |  — |             — |               — |
| Optimized Manual    |        — |         — |      — |  — |             — |               — |

The actual values are generated when the notebook is executed.

---

## 9. Optimization

The manual Logistic Regression implementation is optimized using:

### Vectorization

NumPy matrix operations are used instead of unnecessary Python loops.

For example:

```python
X @ weights
```

is used for vectorized computation.

### Learning-rate tuning

Several learning rates are tested:

```text
0.001
0.005
0.01
0.05
0.1
```

The resulting performance and training times are compared.

### Convergence control

Training can stop when the change in loss becomes sufficiently small.

### L2 Regularization

An L2 regularization term is added to reduce unnecessarily large model weights and improve optimization stability.

---

## 10. Reproducibility

A fixed random seed is used:

```python
random_state = 42
```

The train-test split is created once and reused for the subsequent comparisons.

This ensures that the Scikit-learn and manual implementations are evaluated on the same samples.

---

## 11. Project Structure

```text
DS605_Lab5/
│
├── productivity.csv
│
├── Lab5_Productivity_ML.ipynb
│
├── regression_comparison.csv
│
├── classification_comparison.csv
│
├── logistic_optimization_results.csv
│
└── README.md
```

If additional Python source files are used, they can be organized as:

```text
DS605_Lab5/
│
├── data/
│   └── productivity.csv
│
├── notebooks/
│   └── Lab5_Productivity_ML.ipynb
│
├── results/
│   ├── regression_comparison.csv
│   ├── classification_comparison.csv
│   └── logistic_optimization_results.csv
│
├── README.md
└── requirements.txt
```

---

## 12. Requirements

Install the required Python libraries using:

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

Alternatively, create a `requirements.txt` file containing:

```text
numpy
pandas
scikit-learn
matplotlib
jupyter
```

---

## 13. How to Run

### Step 1 — Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2 — Open the project

```bash
cd DS605_Lab5
```

### Step 3 — Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 4 — Open

```text
Lab5_Productivity_ML.ipynb
```

### Step 5 — Run all cells

The notebook will:

1. Load the dataset.
2. Inspect and preprocess the data.
3. Create the `MeetsTarget` classification target.
4. Create a fixed train-test split.
5. Train Scikit-learn Linear Regression.
6. Train Scikit-learn Logistic Regression.
7. Evaluate both models.
8. Implement preprocessing manually.
9. Implement Linear Regression using NumPy.
10. Implement Logistic Regression using NumPy.
11. Calculate metrics manually.
12. Compare execution times.
13. Optimize the manual Logistic Regression.
14. Generate final comparison tables.

---

## 14. Important Assignment Constraint

The manual implementation does **not** use Scikit-learn for:

* Preprocessing
* Model training
* Prediction
* Evaluation metrics
* Train-test splitting

The manual section uses only NumPy and Pandas for the Machine Learning implementation.

Python's `time` module is used only for measuring
