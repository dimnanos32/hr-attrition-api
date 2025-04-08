from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import numpy as np

# Load model and features
model = joblib.load("final_lr_pipeline.pkl")
feature_names = joblib.load("feature_columns.pkl")

app = FastAPI(title="Employee Attrition Predictor")

class EmployeeData(BaseModel):
    OverTime: Literal[0, 1] = Field(..., description="1 = Overtime, 0 = No Overtime")
    MaritalStatus_Single: Literal[0, 1] = Field(..., description="1 = Single, 0 = Married/Divorced")
    TotalWorkingYears: float = Field(..., ge=0, description="Total years of experience")
    JobLevel: Literal[1, 2, 3, 4, 5] = Field(..., description="Job level (1 to 5)")
    YearsInCurrentRole: float = Field(..., ge=0, description="Years in current role")
    MonthlyIncome: float = Field(..., ge=1000, le=20000, description="Monthly salary in euros (1000 to 20000)")
    Age: float = Field(..., ge=18, le=65, description="Age of the employee")
    JobRole_Sales_Representative: Literal[0, 1] = Field(..., description="1 = Yes, 0 = No")
    YearsWithCurrManager: float = Field(..., ge=0)
    StockOptionLevel: Literal[0, 1, 2, 3] = Field(..., description="Stock option level (0 to 3)")
    YearsAtCompany: float = Field(..., ge=0)
    JobInvolvement: Literal[1, 2, 3, 4] = Field(..., description="1 (Low) to 4 (High)")
    BusinessTravel: Literal[0, 1, 2] = Field(..., description="2 = Travel Frequently, 1 = Travels often, 0 = Non-Travel")
    JobSatisfaction: Literal[1, 2, 3, 4] = Field(..., description="1 (Low) to 4 (High)")
    EnvironmentSatisfaction: Literal[1, 2, 3, 4] = Field(..., description="1 (Low) to 4 (High)")
    
@app.post("/predict")
def predict(data: EmployeeData):
    try:
        print("Incoming data:", data)

        # Adjust column names to match incoming keys
        normalized_features = [col.replace(" ", "_") for col in feature_names]

        # Create input array from JSON data
        input_array = np.array([[getattr(data, col) for col in normalized_features]])
        print("Feature matrix:", input_array)

        # Make prediction
        prob = model.predict_proba(input_array)[0][1]
        pred = int(prob >= 0.5)

        print("Prediction done:", prob, pred)

        return {
            "attrition_probability": round(prob, 3),
            "prediction": pred
        }

    except Exception as e:
        print("🔥 ERROR during prediction:", e)
        raise e



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
