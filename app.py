import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Bank Marketing ML App", layout="wide")
st.title("Bank Marketing Classification System")

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div { cursor: pointer !important; }
    .cm-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        width: fit-content;
        margin: auto;
    }
    button[title="View fullscreen"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a dataset to begin")
    st.stop()

success = st.empty()
success.success("Dataset uploaded successfully")
time.sleep(1)
success.empty()

df = pd.read_csv(uploaded_file)

# Target encoding
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

X = df.drop("deposit", axis=1)
y = df["deposit"]

categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# Model Selection
st.subheader("🔍 Select Model")

model_name = st.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Initialize Model
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)

elif model_name == "kNN":
    model = KNeighborsClassifier(n_neighbors=7, weights="distance")

elif model_name == "Naive Bayes":
    model = GaussianNB()

elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)

elif model_name == "XGBoost":
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

# Train & Predict
with st.spinner("Training model..."):

    if model_name in ["Naive Bayes", "Random Forest", "XGBoost"]:
        X_train_use = X_train_p.toarray()
        X_test_use = X_test_p.toarray()
    else:
        X_train_use = X_train_p
        X_test_use = X_test_p

    model.fit(X_train_use, y_train)
    y_pred = model.predict(X_test_use)
    y_prob = model.predict_proba(X_test_use)[:, 1]

# Save Artifacts
os.makedirs("model", exist_ok=True)

model_path = f"model/{model_name.replace(' ', '_')}.pkl"
joblib.dump(model, model_path)
joblib.dump(preprocessor, "model/preprocessor.pkl")

st.success(f"{model_name} trained successfully")
st.caption(f"Saved at: {model_path}")


# Evaluation Metrics
st.subheader(f"Model Evaluation Metrics – {model_name}")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
    "Value": [
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        matthews_corrcoef(y_test, y_pred)
    ]
})

st.table(metrics_df.style.format({"Value": "{:.4f}"}))

# Confusion Matrix
st.subheader(f"Confusion Matrix – {model_name}")

cm = confusion_matrix(y_test, y_pred)

st.markdown('<div class="cm-card">', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"],
    cbar=False,
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig, use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Probability Distribution
st.subheader(f"Prediction Probability Distribution – {model_name}")

results_df = X_test.copy()
results_df["Actual_Deposit"] = y_test.values
results_df["Predicted_Deposit"] = y_pred
results_df["Probability_Deposit_Yes"] = y_prob

fig = px.histogram(
    results_df,
    x="Probability_Deposit_Yes",
    nbins=30,
    title=f"Prediction Probability Distribution for {model_name}",
    labels={
        "Probability_Deposit_Yes": "Probability of Deposit = Yes"
    },
    opacity=0.8
)

fig.update_layout(
    xaxis_title="Predicted Probability",
    yaxis_title="Number of Customers",
    bargap=0.1
)

st.plotly_chart(fig, use_container_width=True)
