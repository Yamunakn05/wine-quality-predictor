# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Unbiased Wine Predictor 🍷", layout="wide")
st.title("🍷 Unbiased Wine Quality Predictor")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("wine.csv")

# =========================
# IMPORTANT FEATURES
# =========================
features = ["alcohol", "volatile acidity", "sulphates", "density", "citric acid"]

X = df[features]
y = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# APPLY SMOTE (🔥 KEY FIX)
# =========================
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL SELECTION
# =========================
st.sidebar.title("Model Settings")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "KNN", "Random Forest"]
)

threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.4)

if model_option == "Logistic Regression":
    model = LogisticRegression(class_weight='balanced')
elif model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=7)
else:
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced')

model.fit(X_train, y_train)

# =========================
# PERFORMANCE
# =========================
st.subheader("Model Evaluation")

y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob > threshold).astype(int)

st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
st.text(classification_report(y_test, y_pred))

# =========================
# INPUT
# =========================
st.subheader("Enter Wine Details")

input_data = {}
for col in features:
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
if st.button("Predict"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Prediction Probability: {prob:.2f}")

    if prob > threshold:
        st.success("✅ Good Quality Wine")
    else:
        st.error("❌ Bad Quality Wine")

# =========================
# VISUALIZATION
# =========================
st.subheader("Feature Correlation")

fig, ax = plt.subplots()
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.write("---")
st.write("Built with SMOTE + Threshold Tuning for Unbiased Predictions 🚀")
