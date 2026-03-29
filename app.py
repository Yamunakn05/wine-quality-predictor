#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# PAGE SETTINGS
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("Wine Quality Predictor")
st.write("Predict whether a wine is of **Good Quality** or **Bad Quality** using Machine Learning.")

# LOAD DATA
df = pd.read_csv("wine.csv")

# SHOW DATA
st.subheader("Dataset Preview")
st.dataframe(df.head())

# DATA VISUALIZATION
st.subheader("Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(df["alcohol"], kde=True, ax=ax1)
    ax1.set_title("Alcohol Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.countplot(x="quality", data=df, ax=ax2)
    ax2.set_title("Quality Count")
    st.pyplot(fig2)

# PREPROCESSING
X = df.drop("quality", axis=1)
y = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAIN MODELS
lr = LogisticRegression()
lr.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# MODEL PERFORMANCE
st.subheader("Model Performance")

y_pred_lr = lr.predict(X_test)
y_pred_knn = knn.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_knn = accuracy_score(y_test, y_pred_knn)

st.write(f"Logistic Regression Accuracy: {acc_lr:.2f}")
st.write(f"KNN Accuracy: {acc_knn:.2f}")

# Accuracy comparison chart
fig3, ax3 = plt.subplots()
ax3.bar(["Logistic Regression", "KNN"], [acc_lr, acc_knn])
ax3.set_ylabel("Accuracy")
ax3.set_title("Model Comparison")
st.pyplot(fig3)

# USER INPUT
st.subheader("Predict Wine Quality")

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", float(df["fixed acidity"].min()), float(df["fixed acidity"].max()))
    volatile_acidity = st.slider("Volatile Acidity", float(df["volatile acidity"].min()), float(df["volatile acidity"].max()))

with col2:
    citric_acid = st.slider("Citric Acid", float(df["citric acid"].min()), float(df["citric acid"].max()))
    residual_sugar = st.slider("Residual Sugar", float(df["residual sugar"].min()), float(df["residual sugar"].max()))

with col3:
    chlorides = st.slider("Chlorides", float(df["chlorides"].min()), float(df["chlorides"].max()))
    alcohol = st.slider("Alcohol", float(df["alcohol"].min()), float(df["alcohol"].max()))
    free_sulfur_dioxide = st.slider(
    "Free Sulfur Dioxide",
    float(df["free sulfur dioxide"].min()),
    float(df["free sulfur dioxide"].max())
)

total_sulfur_dioxide = st.slider(
    "Total Sulfur Dioxide",
    float(df["total sulfur dioxide"].min()),
    float(df["total sulfur dioxide"].max())
)

density = st.slider(
    "Density",
    float(df["density"].min()),
    float(df["density"].max())
)

pH = st.slider(
    "pH",
    float(df["pH"].min()),
    float(df["pH"].max())
)

sulphates = st.slider(
    "Sulphates",
    float(df["sulphates"].min()),
    float(df["sulphates"].max())
)

# Create input
input_data = np.array([[fixed_acidity,
                        volatile_acidity,
                        citric_acid,
                        residual_sugar,
                        chlorides,
                        free_sulfur_dioxide,
                        total_sulfur_dioxide,
                        density,
                        pH,
                        sulphates,
                        alcohol]])

input_scaled = scaler.transform(input_data)

# PREDICTION
if st.button("Predict Quality"):
    result = lr.predict(input_scaled)

    if result[0] == 1:
        st.success("Good Quality Wine!! ")
    else:
        st.error("Bad Quality Wine!!")

# CONFUSION MATRIX
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_lr)
fig4, ax4 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title("Confusion Matrix (Logistic Regression)")
st.pyplot(fig4)

# FOOTER
st.write("---")
st.write("Made using Machine Learning and Streamlit")
