
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
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ---------------- BACKGROUND ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);color:white;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#141e30,#243b55);}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius:14px;padding:16px;
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;border-radius:10px;font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛡️ Intrusion Detection System Dashboard")
st.caption("Linear SVM vs XGBoost — Reports • Visuals • Export")

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv", "xlsx"])
if uploaded is None:
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)
df = df.dropna().drop_duplicates()

st.success(f"Dataset Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------- FEATURES ----------------
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ---------------- SCALE (SVM) ----------------
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# ---------------- MODELS ----------------
svm = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", random_state=42
)

# ---------------- TRAIN ----------------
svm.fit(X_train_svm, y_train)
xgb.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred_svm = svm.predict(X_test_svm)
y_prob_svm = svm.predict_proba(X_test_svm)[:,1]

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

# ---------------- METRICS ----------------
metrics_df = pd.DataFrame({
    "Model": ["SVM", "XGBoost"],
    "Accuracy": [accuracy_score(y_test,y_pred_svm), accuracy_score(y_test,y_pred_xgb)],
    "Precision": [precision_score(y_test,y_pred_svm), precision_score(y_test,y_pred_xgb)],
    "Recall": [recall_score(y_test,y_pred_svm), recall_score(y_test,y_pred_xgb)],
    "F1-Score": [f1_score(y_test,y_pred_svm), f1_score(y_test,y_pred_xgb)]
})

# =========================================================
# 📊 METRICS TABLE
# =========================================================
st.subheader("📊 Model Performance Summary")
st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

# =========================================================
# 📄 CLASSIFICATION REPORTS
# =========================================================
st.subheader("📄 Classification Reports")

rep_svm = pd.DataFrame(classification_report(y_test,y_pred_svm,output_dict=True)).T
rep_xgb = pd.DataFrame(classification_report(y_test,y_pred_xgb,output_dict=True)).T

c1, c2 = st.columns(2)
c1.markdown("### 🔹 SVM Report")
c1.dataframe(rep_svm.round(4))

c2.markdown("### 🔹 XGBoost Report")
c2.dataframe(rep_xgb.round(4))

# =========================================================
# 🔥 REPORT HEATMAP
# =========================================================
st.subheader("🔥 F1-Score Heatmap")
fig = plt.figure(figsize=(6,4))
sns.heatmap(
    metrics_df.set_index("Model")[["Precision","Recall","F1-Score"]],
    annot=True, cmap="coolwarm"
)
st.pyplot(fig)

# =========================================================
# 📈 PROBABILITY DISTRIBUTION
# =========================================================
st.subheader("📈 Prediction Probability Distribution")

fig = plt.figure(figsize=(7,4))
sns.kdeplot(y_prob_svm, label="SVM", fill=True)
sns.kdeplot(y_prob_xgb, label="XGBoost", fill=True)
plt.legend()
st.pyplot(fig)

# =========================================================
# ❌ ERROR ANALYSIS
# =========================================================
st.subheader("❌ Error Analysis")

err_df = pd.DataFrame({
    "Model": ["SVM","XGBoost"],
    "False Positives": [
        confusion_matrix(y_test,y_pred_svm)[0,1],
        confusion_matrix(y_test,y_pred_xgb)[0,1]
    ],
    "False Negatives": [
        confusion_matrix(y_test,y_pred_svm)[1,0],
        confusion_matrix(y_test,y_pred_xgb)[1,0]
    ]
})

fig = plt.figure(figsize=(6,4))
sns.barplot(data=err_df.melt(id_vars="Model"), x="Model", y="value", hue="variable")
st.pyplot(fig)

# =========================================================
# 🌟 FEATURE IMPORTANCE
# =========================================================
st.subheader("🌟 XGBoost Feature Importance")

imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False)

fig = plt.figure(figsize=(7,4))
sns.barplot(data=imp_df.head(15), x="Importance", y="Feature")
st.pyplot(fig)

# =========================================================
# 📥 EXPORT SECTION
# =========================================================
st.subheader("📥 Download Reports")

st.download_button(
    "⬇️ Download Metrics Summary",
    metrics_df.to_csv(index=False),
    "ids_metrics_summary.csv",
    "text/csv"
)

st.download_button(
    "⬇️ Download SVM Report",
    rep_svm.to_csv(),
    "svm_classification_report.csv",
    "text/csv"
)

st.download_button(
    "⬇️ Download XGBoost Report",
    rep_xgb.to_csv(),
    "xgb_classification_report.csv",
    "text/csv"
)

st.download_button(
    "⬇️ Download Feature Importance",
    imp_df.to_csv(index=False),
    "xgb_feature_importance.csv",
    "text/csv"
)

# ---------------- END ----------------
st.success("✅ IDS Analysis Complete — Reports Generated & Ready for Deployment")
