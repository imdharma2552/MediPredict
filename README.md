# MediPredict

## Disease Prediction

This project implements a disease prediction system using a Stacking Classifier model. The system takes a set of symptoms as input and predicts the possible disease.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing and Feature Selection](#preprocessing-and-feature-selection)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Gradio Interface](#gradio-interface)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview

The goal of this project is to build a machine learning model that can predict a disease based on a given set of symptoms. The project uses a stacking ensemble method, combining multiple base classifiers (Random Forest, Decision Tree, Naive Bayes, SVM, and XGBoost) with a Logistic Regression meta-classifier to improve prediction accuracy.

## Dataset

The project uses a dataset (`dataset_shuffled.csv`) containing patient symptom data and their corresponding diagnoses. Each row represents a patient case, and the columns represent different symptoms (binary: 0 or 1 indicating absence or presence) and the `prognosis` (the diagnosed disease).

## Preprocessing and Feature Selection

1.  **Loading the Data**: The dataset is loaded into a pandas DataFrame.
2.  **Encoding the Target Variable**: The `prognosis` column (disease names) is encoded into numerical labels.
3.  **Handling Missing Values**: A `SimpleImputer` with a 'mean' strategy is used to handle any potential missing values in the symptom data.
4.  **Feature Selection**: `SelectKBest` with the `f_classif` score function is used to select the top 100 features (symptoms) that are most relevant to the target variable (`prognosis`).
5.  **Scaling the Data**: The selected features are scaled using `StandardScaler` to ensure that all features contribute equally to the model training.

## Model Training

A Stacking Classifier is used, which combines the predictions of several base models:

-   Random Forest Classifier
-   Decision Tree Classifier
-   Gaussian Naive Bayes
-   Support Vector Machine (SVM)
-   XGBoost Classifier

The base models are trained on the training data, and their predictions are then used as input for a final meta-classifier, a Logistic Regression model, which makes the final prediction. Stratified K-Fold cross-validation is used during the stacking process to ensure robust training.

## Model Evaluation

The trained stacking model is evaluated on a separate test set using the following metrics:

-   **Accuracy**: The proportion of correctly predicted cases.
-   **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
-   **Precision**: The ability of the model to correctly identify positive cases among all predicted positive cases.
-   **Recall**: The ability of the model to find all positive cases.

## Gradio Interface

A user interface is created using Gradio to allow users to interact with the trained model. The interface takes five symptoms as input from dropdown menus (populated with the selected features) and outputs the predicted disease.
## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/imdharma2552/MediPredict.git
    cd MediPredict
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**
    ```bash
    python app.py
    ```

## Usage

1.  Ensure you have the `dataset_shuffled.csv` file and the saved model files (`trained_model.sav` and `selected_features.pkl`) in the correct paths as specified in the code.
2.  Run the Python script or Jupyter Notebook containing the Gradio interface code.
3.  Access the Gradio web interface through the provided local or public URL.
4.  Select five symptoms from the dropdown menus.
5.  The predicted disease will be displayed.

## Project Structure

-   `dataset_shuffled.csv`: The dataset containing symptom data and diagnoses.
-   `disease_prediction_notebook.ipynb`: Jupyter Notebook or Python script with the code for data loading, preprocessing, model training, evaluation, and the Gradio interface.
-   `trained_model.sav`: Saved trained Stacking Classifier model.
-   `selected_features.pkl`: Saved list of selected features.
-   `README.md`: This file.

## Technologies Used

- Python 3.x
- Machine Learning Libraries: scikit-learn, pandas, numpy
- Data Visualization: matplotlib, seaborn
- Web Framework: (e.g., Flask, Streamlit) *(Update based on your tech stack)*

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

1. Fork it
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

