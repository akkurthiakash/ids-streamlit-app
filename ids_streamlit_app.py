# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL DEPLOY VERSION
# Linear SVM vs XGBoost | Fast | Styled
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intrusion Detection System",
    layout="wide"
)

# ---------------- BEAUTIFUL BACKGROUND (DEPLOY SAFE) ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141e30, #243b55);
    }

    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.4);
        color: white;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        padding: 0.5em 1.2em;
    }

    div[data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px;
    }

    h1, h2, h3, h4 {
        color: #e0f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.title("🛡️ Intrusion Detection System Dashboard")
st.caption("Linear SVM vs XGBoost — Fast & Deploy Ready")

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload IDS Dataset (CSV / XLSX)",
    ["csv", "xlsx"]
)

if uploaded is None:
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)
df = df.dropna().drop_duplicates()

st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------- FEATURES & TARGET ----------------
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# ---------------- SCALE (SVM ONLY) ----------------
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# ---------------- MODELS ----------------
svm = SVC(
    kernel="linear",
    C=1.0,
    probability=True,
    class_weight="balanced",
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

# ---------------- TRAIN ----------------
svm.fit(X_train_svm, y_train)
xgb.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred_svm = svm.predict(X_test_svm)
y_prob_svm = svm.predict_proba(X_test_svm)[:, 1]

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

# ---------------- METRICS ----------------
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

prec_svm = precision_score(y_test, y_pred_svm)
rec_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

# =========================================================
# 📊 ACCURACY
# =========================================================
st.subheader("📊 Accuracy Comparison")
c1, c2 = st.columns(2)
c1.metric("SVM Accuracy", f"{acc_svm:.4f}")
c2.metric("XGBoost Accuracy", f"{acc_xgb:.4f}")

# =========================================================
# 📐 PRECISION / RECALL / F1
# =========================================================
st.subheader("📐 Precision · Recall · F1-Score")

m1, m2 = st.columns(2)
with m1:
    st.markdown("### 🔹 SVM")
    st.metric("Precision", f"{prec_svm:.4f}")
    st.metric("Recall", f"{rec_svm:.4f}")
    st.metric("F1-Score", f"{f1_svm:.4f}")

with m2:
    st.markdown("### 🔹 XGBoost")
    st.metric("Precision", f"{prec_xgb:.4f}")
    st.metric("Recall", f"{rec_xgb:.4f}")
    st.metric("F1-Score", f"{f1_xgb:.4f}")

# =========================================================
# 🔁 CROSS-VALIDATION
# =========================================================
st.subheader("🔁 5-Fold Cross-Validation")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_cv = cross_val_score(svm, X_train_svm, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

cv1, cv2 = st.columns(2)
cv1.metric("SVM CV Accuracy", f"{svm_cv.mean():.4f} ± {svm_cv.std():.4f}")
cv2.metric("XGBoost CV Accuracy", f"{xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

# =========================================================
# 🧩 CONFUSION MATRICES
# =========================================================
st.subheader("🧩 Confusion Matrices")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_svm),
            annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("SVM")

sns.heatmap(confusion_matrix(y_test, y_pred_xgb),
            annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("XGBoost")

st.pyplot(fig)

# =========================================================
# 📈 ROC–AUC
# =========================================================
st.subheader("📈 ROC–AUC Curve")

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

fig = plt.figure(figsize=(6, 5))
plt.plot(fpr_svm, tpr_svm, label=f"SVM AUC = {auc(fpr_svm, tpr_svm):.3f}")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost AUC = {auc(fpr_xgb, tpr_xgb):.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.legend()
st.pyplot(fig)

# =========================================================
# 🌟 FEATURE IMPORTANCE
# =========================================================
st.subheader("🌟 XGBoost Feature Importance")

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig = plt.figure(figsize=(7, 4))
sns.barplot(data=imp_df.head(15), x="Importance", y="Feature")
st.pyplot(fig)

# =========================================================
# 📦 FEATURE DISTRIBUTION BY CLASS
# =========================================================
st.subheader("📦 Feature Distribution by Class")

top_feat = imp_df["Feature"].iloc[0]

fig = plt.figure(figsize=(6, 4))
sns.boxplot(x=y, y=df[top_feat])
plt.xlabel("Class (0 = Normal, 1 = Attack)")
plt.ylabel(top_feat)
st.pyplot(fig)

# =========================================================
# 🔥 CORRELATION HEATMAP
# =========================================================
st.subheader("🔥 Correlation Heatmap (Top 15 Features)")

top_corr_feats = imp_df["Feature"].head(15).values
fig = plt.figure(figsize=(9, 7))
sns.heatmap(df[top_corr_feats].corr(), cmap="coolwarm", annot=False)
st.pyplot(fig)

# ---------------- END ----------------
st.success("✅ Final IDS Evaluation Complete — Deploy Ready")
