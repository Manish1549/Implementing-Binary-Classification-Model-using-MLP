# Implementing-Binary-Classification-Model-using-MLP
# Banknote Authentication with a Multi-Layer Perceptron (MLP) Model

This repository contains a trained Multi-Layer Perceptron (MLP) model for binary classification of banknotes as either authentic or counterfeit. The model was trained and evaluated using the Banknote Authentication Data Set from the UCI Machine Learning Repository (id=267). The goal of this project was to create an effective classifier, optimizing for accuracy and generalizability.

## Dataset

The Banknote Authentication Data Set (UCI ML Repo id=267) comprises image features extracted from genuine and forged banknotes. The features are:

*   Variance of Wavelet Transformed image
*   Skewness of Wavelet Transformed image
*   Curtosis of Wavelet Transformed image
*   Entropy of image

The dataset was preprocessed using `StandardScaler` to normalize the features, ensuring that each feature contributes equally to the model's training process. The target variable is a binary label: `0` for authentic banknotes and `1` for counterfeit ones.

## Model Architecture

*   **Type:** Multi-Layer Perceptron (MLP)
*   **Hidden Layers:** Two hidden layers:
    *   Layer 1: 10 neurons
    *   Layer 2: 5 neurons
*   **Activation Function:** Hyperbolic Tangent (`tanh`) - chosen for its non-linearity and zero-centered output.
*   **Solver:** `lbfgs` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) - selected for its efficiency in handling relatively small datasets and its quasi-Newton optimization approach.
*   **Regularization:** Implicit L2 regularization through the `lbfgs` solver.
*   **Training/Testing Split:** 80% training, 20% testing, stratified to maintain class balance, with a `random_state` of 42 for reproducibility.
*   **Max Iterations**: 500
* **random_state**: 42
* **Libraries**: `scikit-learn`
* **Target**: binary classification
* **Data pre-processing**: data was normalized using `StandardScaler`

## Model Performance

The model's performance was evaluated on the held-out test set, yielding the following metrics:

*   **Test Accuracy:** 0.9964
*   **Train Accuracy:** 0.9982
*   **Classification Report:**
*     precision    recall  f1-score   support

       0       1.00      0.99      1.00       153
       1       0.99      1.00      1.00       122

accuracy                           1.00       275
*   **Confusion Matrix:**
*   [[152 1] [ 0 122]]
*   These metrics demonstrate the model's high performance in accurately classifying banknotes. The high precision and recall indicate the model's effectiveness at minimizing both false positives and false negatives.

## Model File

*   `trained_mlp_banknote_model.joblib`: This is the serialized, trained model file, ready to be loaded and used for inference.

## Usage

1.  **Load the Model:**
2.  **Make Predictions:**

 ## Further Development

The current architecture offers good performance, but potential improvements could include:
* **Hyperparameter Optimization:** Conduct a more thorough search for optimal hyperparameters.
* **Alternative Architectures**: Consider other architectures or more layers.
* **Feature Engineering**: Try creating new features or transforming existing ones.

## Additional Notes

*   The model was trained using the `scikit-learn` library.
*   The `joblib` library is used for efficient model serialization and deserialization.
* This model was trained on the data normalized with the `StandardScaler`.
* The data and training code used to generate this model are also available in this repository.
