
# Machine learning from scratch: Decision Tree Classifier
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project implements a fully functional **Decision Tree Classifier** from
the ground up in Python, without using high-level machine learning libraries
like Scikit-learn for the core logic.
Developed as part of the *Statistical Methods for Machine Learning* course at
the University of Milan, this project explores the internal mechanics of
non-linear predictors, focusing on algorithmic efficiency, recursive data
structures, and statistical validation.

## 🚀 Key features
- **Native categorical support** &nbsp; Handles categorical attributes using
power set membership tests, eliminating the need for one-hot encoding.
- **Modular splitting criteria** &nbsp; Implements multiple heterogeneity
indices via an object-oriented approach:
  - **Gini index**
  - **Scaled entropy**
  - **Bernoulli standard deviation**
- **Robust stopping criteria** &nbsp; Includes hyperparameters for
regularization:
  - `max_depth` &nbsp; Controls the height of the tree.
  - `min_samples_split` &nbsp; Minimum samples required to explore a split.
  - `min_impurity_decrease` &nbsp; Minimum information gain threshold for a new
  node.
- **Performance optimization** &nbsp; Implements unique value sampling to speed
up training on large datasets without significantly sacrificing accuracy.

## 📊 Dataset: UCI Secondary Mushroom
The model is trained on the
[UCI Secondary Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Secondary+Mushroom+Dataset),
consisting of ~20k instances and 15 features.
- **Preprocessing** &nbsp; Handles missing values (NaN) and performs Principal
Component Analysis (PCA) for data visualization.
- **Performance** &nbsp; The custom implementation achieves a generalization
accuracy of **~97%**, matching industry-standard implementations.

## 🔬 Scientific validation
To ensure the correctness of the implementation, the project includes rigorous
sanity checks:
- **Label swapping test** &nbsp; A methodology where training labels are
randomly shuffled to verify that the tree converges toward expected stochastic
behavior (0.5 loss), proving the absence of data leakage.
- **Complexity analysis** &nbsp; Comparison of training times and accuracy
against Scikit-learn's `DecisionTreeClassifier`.

## 📁 Project structure
- `report.pdf` &nbsp; A detailed LaTeX report covering the theoretical
background and experimental results.

The following are all included in the `src/ml_trees/tree.py` source file.
- `TreePredictor` &nbsp; The core class handling the recursive `fit` and
`predict` logic.
- `Node` &nbsp; A class representing internal nodes (decision criteria) and
leaves (labels).
- `SplittingCriterion` &nbsp; An abstract base class for implementing modular
heterogeneity indices.

## 🛠 Installation & usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/AlessandroDiGioacchino/ml-trees-from-scratch.git
   ```

2. **Install dependencies**
    ```bash
    pip install -e .
    ```

3. **Import `TreePredictor`**
   ```python
   from ml_trees import TreePredictor
   ```

Or:

2. **Install Jupyter dependencies**
    ```bash
    pip install -e '.[jlab]'
    ```

3. **Run the analysis**  
   Open the provided Jupyter Notebook or Python script to train the model and
   generate performance plots.

## 📈 Visualizations
### Principal Component Analysis
<details>
  <summary>
    The dataset is projected into 2D space to understand class separability.
  </summary>

  ![PCA Plot](images/mushroom_scatter_plot.png)
</details>

### Decision Tree Structure
<details>
  <summary>
    Example of a tree built using the Gini Index and specific stopping
    criteria.
  </summary>

  ![Tree](images/gini_tree.png)
</details>

## ✍️ Author
**Alessandro Di Gioacchino**
Master's Degree in Computer Science - University of Milan.  
[LinkedIn](https://www.linkedin.com/in/alessandrodigioacchino) |
[GitHub](https://github.com/AlessandroDiGioacchino)

## 📄 License
This project is licensed under the MIT License - see the `LICENSE` file for
details.
