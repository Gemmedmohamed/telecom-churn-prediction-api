# Customer Churn Prediction API 

An end-to-end production-ready Machine Learning API built with **FastAPI**, **Scikit-Learn**, and **Docker**. This service predicts customer churn probabilities based on demographic and usage data, utilizing an optimized Random Forest classifier tailored for imbalanced data.

---

##  Project Overview

Customer churn is a critical business metric. This project delivers a complete pipeline—from trained model artifacts to an API container—designed to evaluate customer churn risk in real-time.

### Key Features
- **Optimized Model:** Random Forest Classifier trained with `class_weight='balanced_subsample'` and a manual optimal threshold of `0.50` for balanced Precision-Recall performance.
- **Dynamic Feature Engineering:** Computes real-time derived features (`Is_New_Customer`, `Charges_Per_Age`, `Risk_Score`) directly inside the API payload handler.
- **Production-Ready FastAPI:** Async-ready backend with automatic interactive swagger documentation.
- **Containerized Deployment:** Fully Dockerized environment ensuring seamless execution across different environments without dependency conflicts.
- **Automated Testing:** Covered with `pytest` and `TestClient` for unit testing API endpoints and input validation.

---

## 📁 Project Structure

```text
churn-prediction-api/
├── main.py                         # FastAPI application and endpoint logic
├── random_forest_churn_model.pkl   # Serialized model and feature names
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition file
├── pytest.ini                      # Test configurations
├── README.md                       # Documentation
└── tests/
    └── test_main.py                # Automated unit tests using pytest
```

---

##  Tech Stack & Tools

- **Language:** Python 3.10 / 3.11
- **Machine Learning:** Scikit-Learn, Pandas, Joblib
- **API Framework:** FastAPI, Pydantic, Uvicorn
- **Testing:** Pytest, HTTPX
- **Containerization:** Docker, WSL 2

---

##  Quick Start Guide

### Option 1: Running with Docker (Recommended)

1. **Build the Docker Image:**
   ```bash
   docker build -t churn-prediction-api .
   ```

2. **Run the Container:**
   ```bash
   docker run -d -p 8000:8000 --name churn_api_container churn-prediction-api
   ```

3. **Access the API Documentation:**
   Open your browser and navigate to:
   - **Swagger UI:** `http://localhost:8000/docs`
   - **ReDoc:** `http://localhost:8000/redoc`

---

### Option 2: Local Python Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gemmedmohamed/telecom-churn-prediction-api
   cd telecom-churn-prediction-api
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

---

## 🧪 Running Unit Tests

Run automated tests to ensure endpoint stability and validation logic:

```bash
python -m pytest
```

---

##  Sample Request & Response

### Endpoint: `POST /predict`

#### Sample Payload:
```json
{
  "TenureMonths": 5,
  "MonthlyCharges": 1000,
  "Age": 18,
  "SignUp_Year": 2026,
  "TotalCharges": 5000,
  "SignUp_Month": 4,
  "Gender_Male": 0,
  "PaymentMethod_Electronic_check": 0,
  "PaymentMethod_Mailed_check": 0,
  "PaymentMethod_Credit_card": 1,
  "Contract_One_year": 0,
  "Contract_Two_year": 0
}
```

#### Sample Response:
```json
{
  "prediction": 1,
  "churn_label": "Churn",
  "churn_probability": 0.824,
  "applied_threshold": 0.50
}
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
