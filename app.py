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

st.set_page_config(page_title="Wine Intelligence System 🍷", layout="wide")

st.title("🍷 Intelligent Wine Quality Prediction System")

df = pd.read_csv("wine.csv")

st.subheader("📂 Data Overview")
st.dataframe(df.head())

features = ["alcohol", "volatile acidity", "sulphates", "density", "citric acid"]

X = df[features]
y = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

st.subheader("⚖️ Quality Distribution Analysis")
st.write(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

st.sidebar.title("⚙️ Model Control Panel")

model_option = st.sidebar.selectbox(
    "Select Algorithm",
    ["Logistic Regression", "KNN", "Random Forest"]
)

threshold = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.4)

if model_option == "Logistic Regression":
    model = LogisticRegression(class_weight='balanced')
elif model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=7)
else:
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced')

model.fit(X_train, y_train)

st.subheader("🧠 Model Performance Analysis")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > threshold).astype(int)

st.subheader("📊 Evaluation Metrics")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
st.text(classification_report(y_test, y_pred))

st.subheader("🧪 Input Wine Characteristics")

input_data = {}
for col in features:
    input_data[col] = st.slider(
        col,
        float(df[col].min()),
        float(df[col].max())
    )

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

st.subheader("🔮 Prediction Outcome")

if st.button("Predict Quality"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Prediction Confidence: {prob:.2f}")

    if prob > threshold:
        st.success("✅ High Quality Wine Detected")
    else:
        st.error("❌ Low Quality Wine Detected")

st.subheader("🔗 Feature Correlation Insights")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size": 8},
    ax=ax
)
st.pyplot(fig)

st.write("---")
st.markdown("🚀 Developed using Machine Learning & Streamlit")
