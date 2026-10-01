from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API للتوقع بمدى احتمالية مغادرة العميل بناءً على نموذج Random Forest",
    version="1.0.0"
)

# 1. تحميل النموذج والـ Payload
try:
    payload = joblib.load('random_forest_churn_model.pkl')
    model = payload['model']
    feature_names = payload['feature_names']
    print("تم تحميل النموذج وقائمة الميزات بنجاح!")
except Exception as e:
    raise RuntimeError(f"فشل تحميل ملف النموذج: {str(e)}")

# الـ Threshold المحدد أثناء التدريب
OPTIMAL_THRESHOLD = 0.50

# 2. Schema البيانات المباشرة
class CustomerData(BaseModel):
    TenureMonths: float
    MonthlyCharges: float
    Age: int
    SignUp_Year: int
    TotalCharges: float
    SignUp_Month: int
    Gender_Male: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0
    PaymentMethod_Credit_card: int = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0


@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # تحويل البيانات إلى DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        
        # 3. حساب الـ 3 Features المستحدثة بالمعادلات والأدوار المداخيلية الصحيحة
        df['Is_New_Customer'] = (df['TenureMonths'] <= 12).astype(int)
        df['Charges_Per_Age'] = df['MonthlyCharges'] / (df['Age'] + 1)
        df['Risk_Score'] = df['MonthlyCharges'] / (df['TenureMonths'] + 1)
        
        # 4. إعادة ترتيب الميزات طبقاً لـ feature_names
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # 5. التوقع بناءً على الـ Threshold المعتمد (0.26)
        probability = float(model.predict_proba(df)[0][1])
        prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
        
        return {
            "prediction": prediction,
            "churn_label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability, 4),
            "applied_threshold": OPTIMAL_THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"حدث خطأ أثناء المعالجة: {str(e)}")