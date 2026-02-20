import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------
# Title
# -------------------------
st.title("Customer Segmentation & Churn Analysis Dashboard")

# -------------------------
# Upload Dataset
# -------------------------
uploaded_file = st.file_uploader("Upload Customer Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Basic Info
    # -------------------------
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write(df.describe())

    # -------------------------
    # Data Cleaning
    # -------------------------
    df.fillna(method="ffill", inplace=True)

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # -------------------------
    # Feature Engineering
    # -------------------------
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["Avg_Spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # -------------------------
    # Feature Selection
    # -------------------------
    features = st.multiselect(
        "Select Features for Segmentation",
        df.select_dtypes(include=np.number).columns,
        default=["tenure", "MonthlyCharges", "TotalCharges"]
    )

    if len(features) > 0:

        segmentation_data = df[features]

        # -------------------------
        # Scaling
        # -------------------------
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(segmentation_data)

        # -------------------------
        # Select Number of Clusters
        # -------------------------
        k = st.slider("Select Number of Clusters", 2, 8, 4)

        # -------------------------
        # Clustering
        # -------------------------
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Customer_Segment"] = kmeans.fit_predict(scaled_data)

        st.subheader("Segmented Data")
        st.dataframe(df.head())

        # -------------------------
        # Segment Summary
        # -------------------------
        st.subheader("Segment Summary")
        summary = df.groupby("Customer_Segment")[features].mean()
        st.dataframe(summary)

        # -------------------------
        # Churn Analysis
        # -------------------------
        if "Churn" in df.columns:
            st.subheader("Churn Rate by Segment")
            churn_rate = df.groupby("Customer_Segment")["Churn"].mean()
            st.bar_chart(churn_rate)

        # -------------------------
        # Visualization
        # -------------------------
        st.subheader("Customer Segmentation Visualization")

        if len(features) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=df[features[0]],
                y=df[features[1]],
                hue=df["Customer_Segment"],
                palette="viridis",
                ax=ax
            )
            st.pyplot(fig)

        # -------------------------
        # Insights
        # -------------------------
        st.subheader("Key Business Insights")

        st.write("""
        1. High-value customers contribute significant revenue.
        2. Customers with lower tenure show higher churn.
        3. Certain segments are more at risk and should be targeted.
        4. Personalized offers can improve retention.
        """)