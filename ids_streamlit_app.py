# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL FULL VERSION (FAST)
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
    accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# =========================================================
# 🔐 LOGIN
# =========================================================
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

USERS = {"akash": hash_password("ids@2025")}

if "auth" not in st.session_state:
    st.session_state.auth = False

def login():
    st.markdown("<h2 style='text-align:center;'>🔐 IDS Dashboard Login</h2>", unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in USERS and hash_password(p) == USERS[u]:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Invalid username or password")

if not st.session_state.auth:
    login()
    st.stop()

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white;}
.center-text {text-align:center;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE & REPORT
# =========================================================
st.markdown("<h1 class='center-text'> Intrusion Detection System </h1>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:15px;">
<b>Purpose:</b> Detect cyber attacks using machine learning<br>
<b>Outcome:</b> XGBoost performs better than SVM<br>
<b>Usage:</b> Network traffic monitoring & early attack detection
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# DATA LOADING (FAST + SAFE)
# =========================================================
@st.cache_data
def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl").dropna().drop_duplicates()
    return pd.read_csv(file).dropna().drop_duplicates()

uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv","xlsx"])
if uploaded is None:
    st.stop()

df = load_data(uploaded)

target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

# =========================================================
# MODEL TRAINING (CACHED)
# =========================================================
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_svm = scaler.fit_transform(X_train)
    X_test_svm = scaler.transform(X_test)

    svm = SVC(kernel="linear", probability=True, class_weight="balanced")
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
    )

    svm.fit(X_train_svm, y_train)
    xgb.fit(X_train, y_train)

    return svm, xgb, X_test, X_test_svm, y_test

svm, xgb, X_test, X_test_svm, y_test = train_models(X, y)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

acc_svm = accuracy_score(y_test, svm.predict(X_test_svm))
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# =========================================================
# FAST SAMPLE FOR VISUALS
# =========================================================
VIS_SAMPLE = min(3000, len(df))
df_vis = df.sample(VIS_SAMPLE, random_state=42)
y_vis = df_vis[target]

# =========================================================
# 🔟 ALL 10 VISUALIZATIONS
# =========================================================
st.markdown("## 📊 Visual Analysis (All Required Visualizations)")

# 1️⃣ Class Distribution
st.subheader("1️⃣ Class Distribution")
fig = plt.figure(figsize=(4,3))
sns.countplot(x=y_vis)
plt.title("Normal vs Attack Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
st.pyplot(fig); plt.close()

# 2️⃣ Accuracy Comparison
st.subheader("2️⃣ Model Accuracy Comparison")
fig = plt.figure(figsize=(4,3))
sns.barplot(x=["SVM","XGBoost"], y=[acc_svm, acc_xgb])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
st.pyplot(fig); plt.close()

# 3️⃣ Confusion Matrix
st.subheader("3️⃣ Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_xgb)
fig = plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig); plt.close()

# 4️⃣ ROC Curve
st.subheader("4️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
fig = plt.figure(figsize=(4,3))
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
st.pyplot(fig); plt.close()

# 5️⃣ Precision–Recall Curve
st.subheader("5️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)
fig = plt.figure(figsize=(4,3))
plt.plot(recall, precision)
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
st.pyplot(fig); plt.close()

# 6️⃣ Prediction Confidence
st.subheader("6️⃣ Prediction Confidence Distribution")
fig = plt.figure(figsize=(4,3))
plt.hist(y_prob_xgb, bins=25)
plt.title("Attack Probability Distribution")
plt.xlabel("Attack Probability")
plt.ylabel("Frequency")
st.pyplot(fig); plt.close()

# 7️⃣ Error Breakdown
st.subheader("7️⃣ Error Breakdown")
error_df = pd.DataFrame({
    "Type":["TN","FP","FN","TP"],
    "Count":[cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})
fig = plt.figure(figsize=(4,3))
sns.barplot(data=error_df, x="Type", y="Count")
plt.title("Prediction Error Breakdown")
st.pyplot(fig); plt.close()

# 8️⃣ Feature Importance
st.subheader("8️⃣ Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(8)

fig = plt.figure(figsize=(5,3))
sns.barplot(data=imp_df, x="Importance", y="Feature")
plt.title("Top Important Features")
st.pyplot(fig); plt.close()

# 9️⃣ Feature vs Class
st.subheader("9️⃣ Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]
fig = plt.figure(figsize=(4,3))
sns.boxplot(x=y_vis, y=df_vis[top_feat])
plt.title(f"{top_feat} Distribution by Class")
plt.xlabel("Class")
plt.ylabel(top_feat)
st.pyplot(fig); plt.close()

# 🔟 Feature Interaction
st.subheader("🔟 Feature Interaction")
f1, f2 = imp_df.iloc[0]["Feature"], imp_df.iloc[1]["Feature"]
fig = plt.figure(figsize=(4,3))
sns.scatterplot(x=df_vis[f1], y=df_vis[f2], hue=y_vis, alpha=0.6)
plt.title("Feature Interaction Scatter Plot")
plt.xlabel(f1)
plt.ylabel(f2)
st.pyplot(fig); plt.close()

# =========================================================
# END
# =========================================================
st.markdown("<h4 class='center-text'>✅ All 10 Visualizations Loaded Successfully</h4>", unsafe_allow_html=True)
