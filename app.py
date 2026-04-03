# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Wine Quality Predictor 🍷", layout="wide")
st.title("🍷 Wine Quality Predictor (Optimized Model)")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("wine.csv")

# =========================
# USE ONLY IMPORTANT FEATURES
# =========================
features = ["alcohol", "volatile acidity", "sulphates", "density", "citric acid"]

X = df[features]
y = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

# =========================
# BALANCE DATASET
# =========================
df_combined = pd.concat([X, y], axis=1)

df_majority = df_combined[df_combined["quality"] == 0]
df_minority = df_combined[df_combined["quality"] == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced[features]
y = df_balanced["quality"]

# =========================
# SPLIT + SCALE
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL SELECTION
# =========================
st.sidebar.title("Choose Model")

model_option = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "KNN", "Random Forest"]
)

if model_option == "Logistic Regression":
    model = LogisticRegression()
elif model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
else:
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced')

# Train
model.fit(X_train, y_train)

# =========================
# PERFORMANCE
# =========================
st.subheader("Model Accuracy")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Accuracy: {acc:.2f}")

# =========================
# INPUT SLIDERS (ONLY 5 NOW)
# =========================
st.subheader("Enter Wine Details")

alcohol = st.slider("Alcohol", float(df["alcohol"].min()), float(df["alcohol"].max()))
volatile_acidity = st.slider("Volatile Acidity", float(df["volatile acidity"].min()), float(df["volatile acidity"].max()))
sulphates = st.slider("Sulphates", float(df["sulphates"].min()), float(df["sulphates"].max()))
density = st.slider("Density", float(df["density"].min()), float(df["density"].max()))
citric_acid = st.slider("Citric Acid", float(df["citric acid"].min()), float(df["citric acid"].max()))

# Create input
input_df = pd.DataFrame([[
    alcohol,
    volatile_acidity,
    sulphates,
    density,
    citric_acid
]], columns=features)

input_scaled = scaler.transform(input_df)

# =========================
# PREDICTION
# =========================
if st.button("Predict Quality"):
    prob = model.predict_proba(input_scaled)[0][1]

    if prob > 0.4:
        st.success(f"✅ Good Quality Wine ({prob*100:.2f}%)")
    else:
        st.error(f"❌ Bad Quality Wine ({(1-prob)*100:.2f}%)")

# =========================
# SIMPLE VISUALIZATION
# =========================
st.subheader("Feature Impact")

fig, ax = plt.subplots()
sns.barplot(x=features, y=df[features].mean(), ax=ax)
plt.xticks(rotation=30)
st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.write("---")
st.write("Optimized ML Model using Important Features 🍷")
