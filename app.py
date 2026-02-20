import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ----------------------
# Title
# ----------------------
st.title("Customer Churn Prediction App")

# ----------------------
# Load Dataset
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"H:\My Drive\Python\Presentation\Pandas datasource\customer_churn.csv")
    return df

df = load_data()

# ----------------------
# Data Cleaning
# ----------------------
df.fillna(method="ffill", inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ----------------------
# Encoding categorical columns
# ----------------------
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    if col != "customerID":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ----------------------
# Feature & Target
# ----------------------
X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

# ----------------------
# Train Model
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ----------------------
# Sidebar Input
# ----------------------
st.sidebar.header("Enter Customer Details")

input_data = {}

for col in X.columns:
    if col in label_encoders:
        categories = label_encoders[col].classes_
        value = st.sidebar.selectbox(col, categories)
        value = label_encoders[col].transform([value])[0]
    else:
        value = st.sidebar.number_input(col, float(df[col].min()), float(df[col].max()))
    input_data[col] = value

# Convert to dataframe
input_df = pd.DataFrame([input_data])

# Scale input
input_scaled = scaler.transform(input_df)

# ----------------------
# Prediction
# ----------------------
if st.button("Predict Churn"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"Customer is likely to CHURN ⚠️")
    else:
        st.success("Customer is NOT likely to churn 😊")

    st.write(f"Churn Probability: {probability:.2f}")