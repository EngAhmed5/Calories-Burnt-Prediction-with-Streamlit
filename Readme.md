# Calories Burnt Prediction

A machine learning regression project to predict calories burnt during physical activities using personal and physiological features. The project covers data preprocessing, model training, evaluation, and deployment with a Streamlit app.

---

## Problem Statement

Accurately estimating calories burned is vital for fitness tracking and health monitoring. This project predicts calories burnt based on features like gender, age, height, weight, exercise duration, heart rate, and body temperature.

## Project Structure

```bash
Calories Burnt Prediction/
├── Data/
│   └── data.csv                        # Raw dataset

├── EDA/
│   └── EDA.ipynb                       # Exploratory Data Analysis notebook

├── Model/
│   ├── data_processing.py              # Data loading, splitting, scaling, encoding
│   ├── modeling.py                     # Model selection, training, saving, loading
│   ├── evaluate.py                     # Evaluation metrics
│   ├── main.py                         # Full pipeline script
│   ├── app.py                          # Streamlit app for interactive prediction

├── Saved Models/
│   ├── Linear Regression.pkl
│   ├── Ridge.pkl
│   ├── Random Forest Regressor.pkl
│   ├── SVR.pkl
│   └── XGB Regressor.pkl               # Saved models with scalers & encoders

├── Results/
│   └── [Evaluation output images]

├── requirements.txt
└── README.md
```

---

### Evaluation Metrics

- Mean Squared Error (MSE)

- Root Mean Squared Error (RMSE)

- R² Score (Coefficient of Determination)

---


## Models & Performance

| Model                    | Train RMSE | Validation RMSE | Train R²  | Validation R² |
|--------------------------|------------|-----------------|-----------|---------------|
| Linear Regression        | 11.27      | 11.49           | 96.72%    | 96.73%        |
| Ridge Regression         | 11.27      | 11.49           | 96.72%    | 96.73%        |
| Random Forest Regressor  | 1.05       | 2.62            | 99.97%    | 99.83%        |
| Support Vector Regressor | 1.12       | 1.01            | 99.97%    | 99.97%        |
| XGBoost Regressor        | 1.32       | 1.66            | 99.96%    | 99.93%        |

---

## How to Run

1. Clone the repo or navigate to the project directory.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt

## Run training and evaluation

- python Model/main.py

## Run the interactive app

- streamlit run Streamlit_App/app.py

---

### Technologies Used

- Python

- Pandas / NumPy

- Scikit-learn

- XGBoost

- Streamlit

- Joblib



### Author

Ahmed Mohamed Hussain
