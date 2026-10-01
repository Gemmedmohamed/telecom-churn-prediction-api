import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 10000

# 1. Base Features
customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
gender = np.random.choice(
    ["Male", "Female", "M", "F", "female", "  Male  "], size=n_samples
)  # Messy categorical
age = np.random.normal(loc=40, scale=12, size=n_samples).round()  # Will add outliers
contract_type = np.random.choice(
    ["Month-to-month", "One year", "Two year", "month-to-month", "1 Year"],
    size=n_samples,
)  # Messy categories
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    size=n_samples,
)

# 2. Financials & Engagement
tenure_months = np.random.randint(1, 72, size=n_samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples).round(2)
total_charges = monthly_charges * tenure_months  # Mixed types later

# 3. Dates with Inconsistent Formats
dates_raw = pd.date_range(start="2021-01-01", periods=n_samples, freq="D")
signup_dates = []
for d in dates_raw:
    fmt = np.random.choice(
        ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "YYYY/MM/DD"]
    )  # Includes invalid text
    if fmt == "YYYY/MM/DD":
        signup_dates.append("INVALID_DATE")
    else:
        signup_dates.append(d.strftime(fmt))

# 4. Target Variable (Churn: 0 or 1)
# Higher monthly charges and lower tenure increase probability of churn
churn_prob = 1 / (
    1 + np.exp(-(-2.0 + 0.03 * monthly_charges - 0.05 * tenure_months))
)
churn = np.random.binomial(1, churn_prob)
tenure_months = tenure_months.astype(object)  # Convert to object for later type issues
# Construct Initial DataFrame
df = pd.DataFrame(
    {
        "CustomerID": customer_ids,
        "Gender": gender,
        "Age": age,
        "SignUpDate": signup_dates,
        "Contract": contract_type,
        "PaymentMethod": payment_method,
        "TenureMonths": tenure_months,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    }
)

# -------------------------------------------------------------
# Injecting Real-World Data Quality Problems
# -------------------------------------------------------------

# Problem A: Missing Values (NaNs and blank spaces)
df.loc[df.sample(frac=0.08).index, "Age"] = np.nan
df.loc[df.sample(frac=0.10).index, "TotalCharges"] = np.nan
df.loc[df.sample(frac=0.05).index, "PaymentMethod"] = " "  # Empty string missingness

# Problem B: Incorrect / Mixed Data Types
# Convert TotalCharges to string and inject non-numeric artifacts
df["TotalCharges"] = df["TotalCharges"].astype(str)
df.loc[df.sample(n=15).index, "TotalCharges"] = "Unknown"
df.loc[df.sample(n=15).index, "TotalCharges"] = "$1200.50"

# Problem C: Outliers
df.loc[df.sample(n=5).index, "Age"] = np.random.choice([150, -5, 200])
df.loc[df.sample(n=5).index, "MonthlyCharges"] = np.random.choice(
    [9999.0, -50.0]
)

# Problem D: Duplicates
duplicate_rows = df.sample(n=20)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Save to CSV
df.to_csv("messy_churn_data.csv", index=False)
print("Dataset successfully generated: 'messy_churn_data.csv'")