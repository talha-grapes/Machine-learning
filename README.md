# E-commerce Dataset ML Project

## Overview
This project involves building machine learning models to classify an e-commerce dataset using three algorithms:
1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. Logistic Regression

The code includes data preprocessing, hyperparameter tuning using GridSearchCV, and evaluation of models based on multiple metrics. Additionally, feature importance and confusion matrices are visualized.

## Code Structure
```plaintext
.
├── Load Dataset
│   ├── Load the `ecommerce_dataset_updated.csv` dataset.
│   ├── Display basic information about the dataset (head, info, describe).
│   └── Handle missing values by imputing with the mean.
├── Preprocessing
│   ├── Encode categorical features using LabelEncoder.
│   ├── Define features (`X`) and target (`y`).
│   ├── Balance the dataset using SMOTE.
│   ├── Split data into training and testing sets (70-30 split, stratified).
│   └── Standardize features using StandardScaler.
├── Model Evaluation
│   ├── Define `evaluate_model_matrix` function:
│   │   ├── Predict class labels and probabilities.
│   │   ├── Compute metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
│   │   └── Generate confusion matrix.
├── Random Forest Classifier
│   ├── Define parameter grid for hyperparameter tuning.
│   ├── Perform GridSearchCV to find the best parameters.
│   ├── Train the model with the best parameters.
│   ├── Evaluate the model using the test set.
│   └── Visualize feature importance.
├── Support Vector Machine (SVM)
│   ├── Define parameter grid for hyperparameter tuning.
│   ├── Perform GridSearchCV to find the best parameters.
│   ├── Train the model with the best parameters.
│   └── Evaluate the model using the test set.
├── Logistic Regression
│   ├── Define parameter grid for hyperparameter tuning.
│   ├── Perform GridSearchCV to find the best parameters.
│   ├── Train the model with the best parameters.
│   ├── Evaluate the model using the test set.
│   └── Visualize feature importance.
├── Model Comparison
│   ├── Compare metrics for all three models (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
│   └── Visualize the comparison using bar plots.
├── Confusion Matrices
│   ├── Generate confusion matrices for each model.
│   └── Visualize confusion matrices using heatmaps.
└── Results
    ├── Display metrics for each model.
    └── Identify strengths and weaknesses based on metrics.
```

## Setup and Execution
1. **Prerequisites**
    - Python 3.x
    - Required libraries:
        - `pandas`
        - `numpy`
        - `scikit-learn`
        - `imblearn`
        - `matplotlib`
        - `seaborn`

    Install dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

2. **Dataset**
    Place the dataset file `ecommerce_dataset_updated.csv` in the project directory.

3. **Run the Code**
    Execute the script using:
    ```bash
    python script_name.py
    ```

4. **Outputs**
    - Model performance metrics.
    - Bar plot comparison of metrics.
    - Confusion matrix heatmaps.
    - Feature importance visualizations.

## Key Metrics
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of true positive predictions.
- **Recall**: Proportion of actual positives captured by the model.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Measures the ability to distinguish between classes.

## Visualizations
- **Model Comparison**: Bar plots comparing metrics.
- **Confusion Matrices**: Heatmaps for each model.
- **Feature Importance**: Bar plots for Random Forest and Logistic Regression.

## Notes
- Ensure the dataset is clean and complete before running the script.
- Adjust hyperparameters in the code as needed for experimentation.
- Use the visualizations to interpret model performance and feature impact.

## Acknowledgments
This project utilizes SMOTE for handling imbalanced datasets and scikit-learn for model building and evaluation.
