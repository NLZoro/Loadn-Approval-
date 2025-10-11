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
st.markdown("This app allows you to explore a preloaded loan dataset, visualize correlations, and train a Random Forest model to predict loan approval.")

# Define default CSV path
csv_path = r"C:\\Users\\rakes\\Downloads\\loan Approval predi\\loan_approval_dataset.csv"

# Load CSV automatically if available
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.success(f"Loaded dataset from {csv_path}")
else:
    st.error(f"❌ CSV file not found at {csv_path}. Please check the path.")
    st.stop()

# Data Overview
st.subheader("📊 Data Overview")
st.write("Shape:", df.shape)
st.dataframe(df.head())

# Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

st.write("After cleaning, shape:", df.shape)

# Visualizations
st.subheader("📈 Data Visualizations")
numeric_df = df.select_dtypes(include=['int64', 'float64'])

if not numeric_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.write("Pairplot (sampled for performance):")
sampled_df = df.sample(min(100, len(df)))
pair_fig = sns.pairplot(sampled_df)
st.pyplot(pair_fig)

# Model Training
st.subheader("🤖 Model Training")
target_col = st.selectbox("Select Target Column", options=df.columns)
feature_cols = st.multiselect("Select Feature Columns", options=[c for c in df.columns if c != target_col])

if st.button("Train Model"):
    X = df[feature_cols]
    y = df[target_col]

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    st.success(f"✅ Model trained successfully with accuracy: {acc:.2f}")

    st.subheader("🧮 Make a Prediction")
    input_data = {}
    for col in feature_cols:
        if X[col].dtype == 'object':
            val = st.selectbox(f"Select {col}", df[col].unique())
            input_data[col] = label_encoders[col].transform([val])[0]
        else:
            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

    if st.button("Predict Loan Status"):
        pred = model.predict(pd.DataFrame([input_data]))[0]
        st.success(f"Prediction: {pred}")