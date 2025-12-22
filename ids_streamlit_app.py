# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL CENTER-ALIGNED VERSION
# Login + Reports + Pair Plots + Prediction
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# =========================================================
# 🔐 LOGIN SYSTEM
# =========================================================
USERS = {
    "admin": hashlib.sha256("Akash123".encode()).hexdigest(),
    "user": hashlib.sha256("Akash123".encode()).hexdigest()
}

def login():
    st.markdown("<h2 style='text-align:center;'>🔐 IDS Dashboard Login</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and hashlib.sha256(password.encode()).hexdigest() == USERS[username]:
            st.session_state["auth"] = True
            st.session_state["user"] = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.clear()
    st.rerun()

if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login()
    st.stop()

# =========================================================
# PAGE CONFIG & STYLE
# =========================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}
.center-text {
    text-align: center;
}
.centered {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.success(f"Logged in as: {st.session_state['user']}")
if st.sidebar.button("🚪 Logout"):
    logout()

# =========================================================
# 🔷 CENTER-ALIGNED TITLE & INTRO TEXT
# =========================================================
st.markdown(
    "<h1 class='center-text'>🛡️ Intrusion Detection System </h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 class='center-text'>Linear SVM vs XGBoost</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='center-text'>"
    "This dashboard performs intrusion detection using machine learning models, "
    "visualizes attack behavior, compares model performance, and supports real-time prediction."
    "</p>",
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# FILE UPLOAD
# =========================================================
st.markdown("<h3 class='center-text'>📂 Upload Dataset</h3>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv", "xlsx"])
if uploaded is None:
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)
df = df.dropna().drop_duplicates()

st.markdown(
    f"<p class='center-text'>Dataset Loaded: <b>{df.shape[0]}</b> rows × <b>{df.shape[1]}</b> columns</p>",
    unsafe_allow_html=True
)

# =========================================================
# DATA PREP
# =========================================================
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# =========================================================
# MODELS
# =========================================================
svm = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

svm.fit(X_train_svm, y_train)
xgb.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test_svm)
y_pred_xgb = xgb.predict(X_test)

# =========================================================
# METRICS (CENTERED TITLE)
# =========================================================
st.markdown("<h3 class='center-text'>📊 Model Performance Summary</h3>", unsafe_allow_html=True)

metrics_df = pd.DataFrame({
    "Model": ["SVM", "XGBoost"],
    "Accuracy": [accuracy_score(y_test,y_pred_svm), accuracy_score(y_test,y_pred_xgb)],
    "Precision": [precision_score(y_test,y_pred_svm), precision_score(y_test,y_pred_xgb)],
    "Recall": [recall_score(y_test,y_pred_svm), recall_score(y_test,y_pred_xgb)],
    "F1-Score": [f1_score(y_test,y_pred_svm), f1_score(y_test,y_pred_xgb)]
}).round(4)

st.markdown('<div class="centered">', unsafe_allow_html=True)
st.dataframe(metrics_df, use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# REPORTS (CENTERED TEXT)
# =========================================================
st.markdown("<h3 class='center-text'>📄 Classification Reports</h3>", unsafe_allow_html=True)

rep_svm = pd.DataFrame(classification_report(y_test,y_pred_svm,output_dict=True)).T.round(4)
rep_xgb = pd.DataFrame(classification_report(y_test,y_pred_xgb,output_dict=True)).T.round(4)

c1, c2 = st.columns(2)
c1.markdown("<h4 class='center-text'>SVM Report</h4>", unsafe_allow_html=True)
c1.dataframe(rep_svm)

c2.markdown("<h4 class='center-text'>XGBoost Report</h4>", unsafe_allow_html=True)
c2.dataframe(rep_xgb)

# =========================================================
# PAIR PLOT
# =========================================================
st.markdown("<h3 class='center-text'>🔗 Pair Plot (Top Features)</h3>", unsafe_allow_html=True)

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False)

top_feats = imp_df["Feature"].head(4).tolist()
df_pair = df[top_feats + [target]].sample(n=min(3000,len(df)), random_state=42)

fig = sns.pairplot(df_pair, hue=target, corner=True, plot_kws={"alpha":0.6})
st.pyplot(fig)

# =========================================================
# REAL-TIME PREDICTION
# =========================================================
st.sidebar.header("🧪 Real-Time Prediction")

user_input = {}
for col in X.columns[:8]:
    user_input[col] = st.sidebar.slider(
        col,
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])

if st.sidebar.button("🔍 Predict"):
    pred = xgb.predict(input_df)[0]
    prob = xgb.predict_proba(input_df)[0][1]

    st.sidebar.success("ATTACK 🚨" if pred==1 else "NORMAL ✅")
    st.sidebar.info(f"Attack Probability: {prob:.2%}")

# =========================================================
# END
# =========================================================
st.markdown(
    "<h4 class='center-text'>✅ IDS Dashboard Loaded Successfully</h4>",
    unsafe_allow_html=True
)

