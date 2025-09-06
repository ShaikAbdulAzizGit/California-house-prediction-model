

# California House Price Prediction Model

This repository contains a machine learning pipeline for predicting California housing prices using the [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html). The model is implemented in Python using `scikit-learn` and employs a Random Forest Regressor. The workflow includes data preprocessing, model training, and inference.

## Features

- **Data Preprocessing**: Handles missing values, scales numerical features, and encodes categorical variables.
- **Stratified Sampling**: Splits the dataset based on income categories to ensure representative train/test sets.
- **Model Training**: Trains a Random Forest Regressor on the processed data.
- **Inference Pipeline**: Predicts house values for new data and saves the results to a CSV file.
- **Persistence**: Saves the trained model and preprocessing pipeline for future use.

## Files

- `main.py` – Main script for pipeline construction, training, and inference.
- `housing.csv` – Input dataset (not included for copyright reasons).
- `input_data.csv` – Test data generated from the split.
- `output.csv` – Prediction results after inference.
- `model.pkl` – Saved trained model.
- `pipeline.pkl` – Saved preprocessing pipeline.

## How It Works

1. **First Run (Training):**
   - The script checks for the existence of `model.pkl`.
   - If not found, it loads `housing.csv`, splits the data, preprocesses it, trains the model, and saves the artifacts.

2. **Subsequent Runs (Inference):**
   - If `model.pkl` exists, the script loads the saved pipeline and model, runs inference on `input_data.csv`, and writes predictions to `output.csv`.

## Usage

1. **Install Dependencies**

    ```bash
    pip install numpy pandas scikit-learn joblib
    ```

2. **Prepare Data**
   
   - Place `housing.csv` in the project directory. You can obtain the dataset from [here](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).

3. **Run the Script**

    ```bash
    python main.py
    ```

   - On the first run, the model will be trained and artifacts will be created.
   - On the second run, predictions will be made on the test split (`input_data.csv`).

## Customization

- You can modify the model (currently Random Forest) or the preprocessing pipeline in `main.py` as needed.
- To retrain the model, delete `model.pkl` and `pipeline.pkl` and rerun the script.

## License

This project is provided for educational purposes. Refer to the dataset's license for usage restrictions.

## Author

[Abdul Aziz Shaik](https://github.com/ShaikAbdulAzizGit)
