from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_churn_success():
    # اختبار إرسال بيانات صحيحة
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_label" in data
    assert "churn_probability" in data
    assert data["applied_threshold"] == 0.50

def test_predict_churn_invalid_data():
    # اختبار إرسال بيانات ناقصة أو خاطئة
    payload = {
        "TenureMonths": "invalid_number" # نص بدلاً من رقم
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity