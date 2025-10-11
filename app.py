import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("💸 Loan Approval Prediction App")
st.markdown("This app allows you to predict whether a loan will be approved based on applicant details.")

# Define default CSV path
csv_path = r"C:\\Users\\rakes\\Downloads\\loan Approval predi\\loan_approval_dataset.csv"

# Load CSV automatically if available
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success(f"Loaded dataset from {csv_path}")
else:
    st.error(f"❌ CSV file not found at {csv_path}. Please check the path.")
    st.stop()

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Train Model
X = df.drop(columns=[' loan_status'], errors='ignore')
y = df[' loan_status'] if ' loan_status' in df.columns else None

if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.success(f"✅ Model trained successfully with accuracy: {acc:.2f}")
else:
    st.warning("⚠️ Target column 'loan_status' not found in dataset.")
    st.stop()

# Prediction UI
st.subheader("🧮 Predict Loan Approval Status")
input_data = {}

for col in X.columns:
    if X[col].dtype == 'object':
        val = st.selectbox(f"Select {col}", df[col].unique())
        input_data[col] = label_encoders[col].transform([val])[0]
    else:
        input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

if st.button("Predict Loan Status"):
    pred = model.predict(pd.DataFrame([input_data]))[0]
    if pred == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Rejected!")