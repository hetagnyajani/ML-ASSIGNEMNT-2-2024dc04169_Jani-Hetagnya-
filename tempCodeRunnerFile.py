import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

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
st.title("🏦 Bank Marketing Classification System")

# -----------------------------
# Cursor fix (NO editable text)
# -----------------------------
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    .cm-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        width: fit-content;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# 1️⃣ Upload Dataset
# =============================
uploaded_file = st.file_uploader(
    "Upload Dataset (CSV with target column 'deposit')",
    type=["csv"]
)

if uploaded_file:

    success_placeholder = st.empty()
    success_placeholder.success("Dataset uploaded successfully")
    time.sleep(2)
    success_placeholder.empty()

    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # Prepare Data
    # -----------------------------
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

    X = df.drop("deposit", axis=1)
    y = df["deposit"]

    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # =============================
    # 2️⃣ Model Selection (NO re-select placeholder)
    # =============================
    st.subheader("🔍 Select Model")

    if "model_selected" not in st.session_state:
        st.session_state.model_selected = None

    model_options = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]

    if st.session_state.model_selected is None:
        model_name = st.selectbox(
            "Choose a Classification Model",
            ["Select a model"] + model_options
        )
        if model_name != "Select a model":
            st.session_state.model_selected = model_name
    else:
        model_name = st.selectbox(
            "Choose a Classification Model",
            model_options,
            index=model_options.index(st.session_state.model_selected)
        )

    if st.session_state.model_selected is None:
        st.info("👆 Please select a model to view evaluation results")
        st.stop()

    model_name = st.session_state.model_selected

    # -----------------------------
    # Initialize Model
    # -----------------------------
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "kNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

    # =============================
    # Train Model
    # =============================
    with st.spinner("Training model and evaluating..."):

        if model_name == "Naive Bayes":
            X_train_nb = X_train_p.toarray()
            X_test_nb = X_test_p.toarray()
            model.fit(X_train_nb, y_train)
            y_pred = model.predict(X_test_nb)
            y_prob = model.predict_proba(X_test_nb)[:, 1]
        else:
            model.fit(X_train_p, y_train)
            y_pred = model.predict(X_test_p)
            y_prob = model.predict_proba(X_test_p)[:, 1]

    # =============================
    # 3️⃣ Evaluation Metrics
    # =============================
    st.subheader(f"📊 Model Evaluation Metrics – {model_name}")

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

    # =============================
    # 4️⃣ Confusion Matrix (UI improved)
    # =============================
    st.subheader(f"📉 Confusion Matrix – {model_name}")

    cm = confusion_matrix(y_test, y_pred)

    with st.container():
        st.markdown('<div class="cm-card">', unsafe_allow_html=True)

        fig_cm, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            cbar=False,
            annot_kws={"size": 11},
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(labelsize=10)

        st.pyplot(fig_cm, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("⬆️ Please upload a dataset to begin")
