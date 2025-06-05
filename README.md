# support-vector-machine-classification
# Task 7 – Support Vector Machines (SVM) Classification

## 📁 Internship Project: AI & ML – SVM Implementation
This project focuses on implementing **Support Vector Machines (SVM)** for binary classification using both **linear** and **non-linear (RBF kernel)** models. The Breast Cancer dataset is used for model training, evaluation, and visualization.
---
## 📚 What I Learned
- Concept of Support Vectors and margin maximization
- Linear vs Non-linear classification using SVM
- Kernel trick (RBF)
- Hyperparameter tuning (C and gamma)
- Model evaluation using confusion matrix and classification report
- PCA for visualizing decision boundaries
---
## 🛠 Tools & Libraries Used
- Python
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
---
## 📊 Dataset
- **Name**: Breast Cancer Wisconsin Diagnostic Dataset
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset) / `sklearn.datasets.load_breast_cancer`
- **Target classes**:
  - 0: Malignant
  - 1: Benign
- **Features**: 30 numerical features like mean radius, texture, area, etc.
---
## 🔎 Steps Performed
1. **Loaded Dataset** from scikit-learn
2. **Normalized** features using `StandardScaler`
3. **Split** data into training and test sets
4. **Trained SVM** models with:
   - Linear kernel
   - RBF (Gaussian) kernel
5. **Evaluated** models using:
   - Accuracy
   - Confusion matrix
   - Classification report
6. **Tuned Hyperparameters** using `GridSearchCV`
7. **Visualized** data with PCA for 2D boundary
---
## 📈 Results
| Kernel | Accuracy | Best Params |
|--------|----------|-------------|
| Linear | ~96%     | C = 1       |
| RBF    | ~98%     | C = 10, gamma = 0.01 |

- Best results achieved with RBF kernel.
- Confusion matrix and PCA plots attached in notebook.
---
## 📌 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/svm-task7.git
   cd svm-task7
