# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL VERSION
# Reports Shown in Dashboard | Centered | PDF Export
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#141e30,#243b55);
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px;
}
.centered {
    display: flex;
    justify-content: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🛡️ Intrusion Detection System Dashboard")
st.caption("Linear SVM vs XGBoost — Full Reports Shown")

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

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# ---------------- SCALING (SVM) ----------------
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# ---------------- MODELS ----------------
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

# ---------------- TRAIN ----------------
svm.fit(X_train_svm, y_train)
xgb.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred_svm = svm.predict(X_test_svm)
y_pred_xgb = xgb.predict(X_test)

# ---------------- METRICS ----------------
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

prec_svm = precision_score(y_test, y_pred_svm)
rec_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

metrics_df = pd.DataFrame({
    "Model": ["SVM", "XGBoost"],
    "Accuracy": [acc_svm, acc_xgb],
    "Precision": [prec_svm, prec_xgb],
    "Recall": [rec_svm, rec_xgb],
    "F1-Score": [f1_svm, f1_xgb]
}).round(4)

# =========================================================
# 📊 METRICS SUMMARY (CENTERED)
# =========================================================
st.subheader("📊 Model Performance Summary")
st.markdown('<div class="centered">', unsafe_allow_html=True)
st.dataframe(metrics_df, use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 📄 CLASSIFICATION REPORTS (SHOWN IN DASHBOARD)
# =========================================================
st.subheader("📄 Model Evaluation Reports")

rep_svm = pd.DataFrame(
    classification_report(y_test, y_pred_svm, output_dict=True)
).T.round(4)

rep_xgb = pd.DataFrame(
    classification_report(y_test, y_pred_xgb, output_dict=True)
).T.round(4)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🔹 SVM Classification Report")
    st.dataframe(rep_svm, use_container_width=True)

with c2:
    st.markdown("### 🔹 XGBoost Classification Report")
    st.dataframe(rep_xgb, use_container_width=True)

# =========================================================
# 🧩 CONFUSION MATRIX REPORT
# =========================================================
st.subheader("🧩 Confusion Matrix Report")

cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("SVM Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("XGBoost Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

st.pyplot(fig)

# =========================================================
# 📝 TEXT REPORT SUMMARY
# =========================================================
st.subheader("📝 Report Summary")

st.markdown(f"""
**SVM Model**
- Accuracy: `{acc_svm:.4f}`
- Precision: `{prec_svm:.4f}`
- Recall: `{rec_svm:.4f}`
- F1-Score: `{f1_svm:.4f}`

**XGBoost Model**
- Accuracy: `{acc_xgb:.4f}`
- Precision: `{prec_xgb:.4f}`
- Recall: `{rec_xgb:.4f}`
- F1-Score: `{f1_xgb:.4f}`
""")

# =========================================================
# 📄 PDF REPORT
# =========================================================
def generate_pdf(metrics_df, svm_report, xgb_report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Intrusion Detection System Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Model Performance Summary", styles["Heading2"]))
    elements.append(Table([metrics_df.columns.tolist()] + metrics_df.values.tolist()))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("SVM Classification Report", styles["Heading2"]))
    elements.append(Table([svm_report.reset_index().columns.tolist()] +
                          svm_report.reset_index().values.tolist()))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("XGBoost Classification Report", styles["Heading2"]))
    elements.append(Table([xgb_report.reset_index().columns.tolist()] +
                          xgb_report.reset_index().values.tolist()))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =========================================================
# 📥 DOWNLOAD SECTION
# =========================================================
st.subheader("📥 Download Reports")

pdf_file = generate_pdf(metrics_df, rep_svm, rep_xgb)

st.download_button(
    "⬇️ Download Full Report (PDF)",
    pdf_file,
    "IDS_Report.pdf",
    "application/pdf"
)

st.download_button(
    "⬇️ Download Metrics (CSV)",
    metrics_df.to_csv(index=False),
    "metrics.csv",
    "text/csv"
)

# ---------------- END ----------------
st.success("✅ IDS Dashboard Complete — Reports Displayed Successfully")
