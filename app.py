# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Wine Quality Predictor 🍷", layout="wide")
st.title("🍷 Wine Quality Prediction System")
st.markdown("### Machine Learning + Web App Project")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("wine.csv")

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "KNN", "Random Forest"]
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# =========================
# DATA OVERVIEW
# =========================
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

if st.checkbox("Show Dataset Info"):
    st.write(df.describe())

# =========================
# VISUALIZATION
# =========================
st.subheader("📈 Data Visualization")

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

# =========================
# PREPROCESSING
# =========================
X = df.drop(["quality", "Id"], axis=1, errors="ignore")
y = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL SELECTION
# =========================
if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
else:
    model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("🤖 Model Performance")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Accuracy: {accuracy:.2f}")

# Classification report
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
st.pyplot(fig3)

# =========================
# FEATURE IMPORTANCE (RF only)
# =========================
if model_choice == "Random Forest":
    st.subheader("⭐ Feature Importance")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig4, ax4 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax4)
    st.pyplot(fig4)

# =========================
# USER INPUT
# =========================
st.subheader("🧪 Predict Wine Quality")

input_data = {}

for col in X.columns:
    input_data[col] = st.slider(
        col,
        float(df[col].min()),
        float(df[col].max())
    )

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Good Quality Wine ({prob*100:.2f}% confidence)")
    else:
        st.error(f"❌ Bad Quality Wine ({(1-prob)*100:.2f}% confidence)")

# =========================
# SAVE MODEL
# =========================
if st.button("💾 Save Model"):
    with open("wine_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model Saved Successfully!")

# =========================
# FOOTER
# =========================
st.write("---")
st.markdown("💡 Built with Streamlit | Machine Learning Project")
