# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(page_title="Wine Quality Predictor 🍷", layout="wide")
st.title("🍷 Wine Quality Prediction System")
st.write("Predict whether a wine is **Good** or **Bad** using Machine Learning")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("wine.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# SHOW CLASS DISTRIBUTION
# =========================
st.subheader("Class Distribution (Before Balancing)")
st.write(df["quality"].value_counts())

# =========================
# PREPROCESSING + BALANCING
# =========================
X = df.drop(["quality", "Id"], axis=1, errors="ignore")
y = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

df_combined = pd.concat([X, y], axis=1)

df_majority = df_combined[df_combined["quality"] == 0]
df_minority = df_combined[df_combined["quality"] == 1]

# Upsample minority
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Show balanced distribution
st.subheader("Class Distribution (After Balancing)")
st.write(df_balanced["quality"].value_counts())

# Split
X = df_balanced.drop("quality", axis=1)
y = df_balanced["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL TRAINING
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("Model Performance")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Feature Importance")

importance = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax2)
st.pyplot(fig2)

# =========================
# USER INPUT
# =========================
st.subheader("Predict Wine Quality")

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
if st.button("Predict Quality"):
    prob = model.predict_proba(input_scaled)[0][1]

    # Adjusted threshold
    if prob > 0.35:
        st.success(f"✅ Good Quality Wine ({prob*100:.2f}% confidence)")
    else:
        st.error(f"❌ Bad Quality Wine ({(1-prob)*100:.2f}% confidence)")

# =========================
# FOOTER
# =========================
st.write("---")
st.write("Built with ❤️ using Streamlit & Machine Learning")
