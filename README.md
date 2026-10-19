# Logistic Regression Classifier Workshop

**CSCN8010 — Foundations of Machine Learning**  
**Group:** Final_Project_Group 2  
**Use case:** Predicting student pass/fail outcomes from study hours

This project demonstrates **binary classification with logistic regression** using a small educational dataset. It combines mathematical explanations, visualizations, a **vectorized NumPy implementation from scratch**, and a comparison against scikit-learn.

> **Scope:** The reported performance is measured on the *training data*. These results illustrate the algorithm and should not be treated as evidence of accuracy on new students.

## Project objectives

- Distinguish binary classification from linear regression.
- Convert linear model outputs into probabilities with the **sigmoid function**.
- Explain features, labels, the **0.5 classification threshold**, and the decision boundary.
- Derive and implement **Binary Cross-Entropy (Log-Loss)**.
- Derive the gradients and optimize weights and bias using **batch Gradient Descent**.
- Visualize the sigmoid curve, loss functions, training convergence, and fitted classifier.
- Compare the from-scratch implementation with **scikit-learn LogisticRegression**.
- Document three technical talking points and record feedback from peer programming.

## Repository structure

```text
LogisticRegressionClassifier_Workshop/
├── LogisticRegressionClassifier_Workshop.ipynb  # Professor's workshop/reference
├── PairProgrammingReport.ipynb                   # Implementation, results, talking points
├── requirements.txt                              # Pinned Python dependencies
├── .gitignore                                    # Excludes local virtual environment
└── README.md                                     # Project documentation
```

The core assignment is in **[PairProgrammingReport.ipynb](PairProgrammingReport.ipynb)**. The original workshop material is kept separately for reference.

## Getting started

### Prerequisites

- Python (the assignment notebook was run with **Python 3.13.7**)
- Git
- VS Code with the **Python** and **Jupyter** extensions, or another compatible Jupyter environment

### Installation — Windows PowerShell

```powershell
git clone https://github.com/UbaidUllah777/LogisticRegressionClassifier_Workshop.git
cd LogisticRegressionClassifier_Workshop

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open the folder in VS Code, then open **PairProgrammingReport.ipynb**, select your local **.venv** Python kernel, and choose **Run All** to reproduce the outputs.

The 19 study-hours/Pass–Fail examples are embedded directly in the notebook; no external CSV file, database, or API credentials are needed to run this classifier.

## Methodology

### 1. Binary classification and sigmoid

The input feature is **study hours**, and the target is **Fail (0)** or **Pass (1)**. Logistic regression converts a linear score into a probability:

$$
z = Xw + b,\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

A predicted probability of **0.5 or higher** produces class **1 (Pass)**; otherwise, the prediction is class **0 (Fail)**.

### 2. Binary Cross-Entropy

The objective measures how well predicted probabilities match the observed labels:

$$
J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}
\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
$$

The NumPy implementation clips probabilities to avoid undefined logarithms at exactly zero or one.

### 3. Vectorized Gradient Descent

The model uses the gradients

$$
\frac{\partial J}{\partial w} = \frac{1}{m}X^T(p-y),
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(p_i-y_i)
$$

and repeatedly updates both parameters:

$$
w \leftarrow w - \alpha\frac{\partial J}{\partial w},
\qquad
b \leftarrow b - \alpha\frac{\partial J}{\partial b}
$$

The notebook trains the model for **5,000 iterations** with a **learning rate of 0.1**.

### 4. Evaluation and comparison

The trained NumPy model is compared with scikit-learn's logistic regression implementation on the **same 19 training observations**. For a more comparable objective, scikit-learn regularization is disabled.

## Verified results

| Metric | From-scratch NumPy | Scikit-learn |
|:--|--:|--:|
| Training accuracy | **89.47%** | **89.47%** |
| Correct classifications | 17 / 19 | 17 / 19 |
| Binary Cross-Entropy loss | 0.2332 | 0.2321 |
| Learned weight | 2.6941 | 2.9703 |
| Learned bias | -7.7143 | -8.5400 |

**From-scratch decision boundary:** approximately **2.8634 study hours**.

Both models produce the same training confusion matrix (rows: actual Fail/Pass; columns: predicted Fail/Pass):

```text
[[9, 1],
 [1, 8]]
```

That corresponds to **9 true negatives, 8 true positives, 1 false positive, and 1 false negative**.

The saved notebook also includes the sigmoid plot, Binary Cross-Entropy loss curves, Gradient Descent convergence plot, and learned probability curve.

### Interpretation and limitations

The decrease in loss shows that Gradient Descent successfully improved the model's fit to the available observations. The close match to scikit-learn provides an additional implementation check.

However, the dataset contains only **19 examples**, and accuracy was calculated **on the training set**. Performance on unseen students has **not** been established. A larger dataset and separate held-out evaluation would be required before drawing generalization conclusions.

## Pair programming and deliverables

The assignment notebook contains three technical talking points covering:

1. **Sigmoid and classification:** converting scores to probabilities and choosing a threshold.
2. **Binary Cross-Entropy:** measuring the penalty for incorrect probabilities.
3. **Gradient Descent:** minimizing loss by updating model parameters.


### Peer Programming and Code Review

Our team completed a technical code review with
Team A and documented the findings in Section 7
of PairProgrammingReport.ipynb.

**Peer Team Repository:**

https://github.com/kinghighshow/LogisticRegressionClassifierWorkshop.git

The review covered:

- Gradient Descent convergence and learning rates.
- Numerical stability of the sigmoid function.
- Efficiency of loss calculations.
- Object-oriented versus functional implementations.
- Model evaluation and limitations.

We documented the peer team's questions, their
answers to our questions, and our reflections.

Based on their feedback, we improved the sigmoid
function using NumPy clipping to reduce numerical
overflow risks.


## Group members

| Name | Student ID |
|:--|:--|
| Ubaid Ullah | 9110715 |
| Ved Vishal Patel | 9111628 |
| Koushik Balne | 9087547 |
| Sukhchain Singh | 9111541 |

## References and acknowledgment

- **Instructor's workshop:** [ProfEspinosaAIML / LogisticRegressionClassifier_Workshop](https://github.com/ProfEspinosaAIML/LogisticRegressionClassifier_Workshop)
- **This project repository:** [UbaidUllah777 / LogisticRegressionClassifier_Workshop](https://github.com/UbaidUllah777/LogisticRegressionClassifier_Workshop)
- [scikit-learn — LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

This project builds on the professor-provided workshop materials for educational purposes.
