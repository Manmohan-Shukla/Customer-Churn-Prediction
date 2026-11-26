# Customer Churn Prediction & Retention System

**End-to-End Machine Learning Project**

Predicting customer churn for a telecommunications company and turning insights into actionable retention strategies.

---

## Project Overview

Customer acquisition is expensive. Retaining existing customers is significantly more cost-effective.

This project builds a complete, production-oriented machine learning system that:

1. Predicts which customers are likely to churn.
2. Explains *why* they are likely to leave using SHAP.
3. Prioritizes high-risk customers.
4. Provides actionable retention recommendations.
5. Serves predictions through an API and interactive dashboard.

The system follows the complete data science lifecycle, from raw data processing to model deployment.

## Business Problem

- Average telecom churn rate: **15–25% annually**
- Acquiring a new customer can cost **5–25× more** than retaining an existing one.
- Goal: Identify customers at high risk of churning so businesses can intervene early with targeted retention strategies.

### Success Metrics

- **High Recall:** Identify as many potential churners as possible.
- **Good Precision:** Avoid unnecessary retention costs.
- **ROC-AUC and PR-AUC:** Evaluate model discrimination and performance on imbalanced data.
- **Explainability:** Provide actionable insights for business teams.

## Dataset

**IBM Telco Customer Churn Dataset**

- Total customers: 7,043
- Features: 21
- Target variable: `Churn` (Yes / No)
- Churn rate: Approximately 26.5%

### Dataset Sources

- [IBM Sample Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- [Kaggle Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/                  # Original immutable dataset
│   ├── processed/            # Cleaned and feature-engineered data
│   └── external/             # Optional external data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb     # Model training and evaluation
│
├── src/
│   ├── data/                 # Data loading and cleaning
│   ├── features/             # Feature engineering
│   ├── models/               # Training, prediction and evaluation
│   └── visualization/        # Visualization utilities
│
├── models/                   # Saved model artifacts (.pkl / .joblib)
│
├── reports/
│   ├── figures/              # EDA and model performance plots
│   └── metrics/              # Evaluation reports
│
├── app/
│   ├── streamlit_app.py      # Interactive dashboard
│   └── api.py                # FastAPI prediction service
│
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── mlruns/                   # MLflow experiment tracking
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## End-to-End Pipeline

| Stage | Description | Key Tools |
|---|---|---|
| 1. Data Ingestion | Load raw CSV and store structured data | pandas, SQLite |
| 2. Data Cleaning | Handle missing values, type conversion and outliers | pandas, scikit-learn |
| 3. Feature Engineering | Create features and preprocessing pipelines | scikit-learn Pipeline |
| 4. Exploratory Analysis | Understand patterns and churn drivers | seaborn, plotly, ydata-profiling |
| 5. Modeling | Train multiple algorithms and tune hyperparameters | XGBoost, LightGBM, Optuna |
| 6. Experiment Tracking | Log metrics, parameters and models | MLflow |
| 7. Explainability | Global and local feature importance | SHAP |
| 8. Evaluation | Business-oriented metrics and error analysis | scikit-learn |
| 9. Deployment | Serve predictions through an API and dashboard | FastAPI, Streamlit, Docker |
| 10. Monitoring (Optional) | Detect data and model drift | Evidently |

## Tech Stack

### Core Data Science

- **Python 3.10+**
- **pandas / Polars** – Data manipulation
- **NumPy** – Numerical computing
- **scikit-learn** – Preprocessing, pipelines and classical ML models
- **XGBoost / LightGBM / CatBoost** – Gradient boosting algorithms
- **imbalanced-learn** – Handling class imbalance
- **Optuna** – Hyperparameter optimization
- **SHAP** – Model explainability

### Experimentation and Tracking

- **MLflow** – Experiment tracking and model registry
- **DVC (Optional)** – Data and model versioning
- **Jupyter Lab** – Interactive development

### Visualization

- **Matplotlib / Seaborn** – Static visualizations
- **Plotly** – Interactive visualizations
- **Streamlit** – Interactive dashboard

### Deployment and Engineering

- **FastAPI** – Prediction API
- **Pydantic** – Data validation
- **Docker** – Containerization
- **Git & GitHub** – Version control
- **GitHub Actions (Optional)** – CI/CD automation

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git

cd customer-churn-prediction
```

### 2. Create and Activate Virtual Environment

**Windows:**

```bash
python -m venv .venv

.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the IBM Telco Customer Churn dataset and place the CSV file inside:

```text
data/raw/
```

### 5. Run Notebooks

Execute the notebooks in the following order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_modeling.ipynb`

### 6. Launch the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### 7. Launch the FastAPI Service

```bash
uvicorn app.api:app --reload
```

API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

### 8. Run Using Docker (Optional)

```bash
docker-compose up --build
```

## Key Features

- **End-to-End ML Pipeline:** Reproducible preprocessing and model training using scikit-learn pipelines.
- **Data Leakage Prevention:** Consistent preprocessing fitted only on training data.
- **Class Imbalance Handling:** Techniques to improve churner identification.
- **Experiment Tracking:** Track model parameters, metrics and artifacts using MLflow.
- **Model Explainability:** Understand individual predictions and global churn drivers using SHAP.
- **Interactive Dashboard:** Explore customer churn predictions and insights using Streamlit.
- **Prediction API:** Integrate the trained model with other applications using FastAPI.
- **Containerization:** Package the application using Docker.

## Model Performance

The following table contains illustrative results. Actual performance will depend on preprocessing, feature engineering, model selection and hyperparameter tuning.

| Model | ROC-AUC | PR-AUC | F1-Score | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.84 | 0.63 | 0.61 | 0.68 |
| Random Forest | 0.86 | 0.67 | 0.64 | 0.71 |
| **LightGBM** | **0.88** | **0.71** | **0.67** | **0.74** |

> **Note:** These are example values, not verified experimental results. Replace them with metrics obtained from your actual test set.

## Future Improvements

- Real-time churn prediction using streaming data.
- Automated model retraining pipeline.
- Customer Lifetime Value (CLV) integration.
- Cloud deployment using AWS, GCP or Azure.
- A/B testing framework to evaluate retention campaigns.
- Model and data drift monitoring.

## Author

**Your Name**

Aspiring Data Scientist / Machine Learning Engineer

- [LinkedIn](https://linkedin.com/in/yourprofile)
- [GitHub](https://github.com/yourusername)

## License

This project is intended for educational and portfolio purposes.
