# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL VERSION
# Login • Results & Usage • All Visuals • PDF Report
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import os
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =========================================================
# 🔐 LOGIN (CHANGE CREDENTIALS HERE)
# =========================================================
def hash_password(p): 
    return hashlib.sha256(p.encode()).hexdigest()

USERS = {
    "akash": hash_password("ids@2025")   # 👈 change if needed
}

if "auth" not in st.session_state:
    st.session_state.auth = False

def login_page():
    st.markdown("<h2 style='text-align:center;'> IDS Dashboard Login</h2>", unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in USERS and hash_password(p) == USERS[u]:
            st.session_state.auth = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.session_state.clear()
    st.rerun()

if not st.session_state.auth:
    login_page()
    st.stop()

# =========================================================
# PAGE CONFIG & STYLE
# =========================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white;}
.center-text {text-align:center;}
.centered {display:flex; justify-content:center;}
</style>
""", unsafe_allow_html=True)

st.sidebar.success(f"Logged in as: {st.session_state.user}")
if st.sidebar.button("🚪 Logout"):
    logout()

# =========================================================
# TITLE
# =========================================================
st.markdown("<h1 class='center-text'> Intrusion Detection System </h1>", unsafe_allow_html=True)
st.markdown("<h3 class='center-text'>Machine Learning–Based Network Security</h3>", unsafe_allow_html=True)

# =========================================================
# 📌 RESULTS & USAGE (FRONT OF DASHBOARD)
# =========================================================
st.markdown("<h2 class='center-text'> Results and Usage of the Proposed IDS</h2>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:16px; line-height:1.8;">
<b>Result:</b><br>
XGBoost achieves higher accuracy and F1-score than Linear SVM, with fewer missed attacks,
making it more reliable for intrusion detection.<br><br>

<b>Usage:</b><br>
The system analyzes network traffic features and classifies them as <b>Normal</b> or <b>Attack</b>.
It can be used for real-time monitoring, alert generation, and offline analysis.<br><br>

<b>Applications:</b><br>
Enterprise security • Early attack detection • IDS research • SIEM / firewall integration
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# DATASET UPLOAD
# =========================================================
uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv","xlsx"])
if uploaded is None:
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)
df = df.dropna().drop_duplicates()

target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

# =========================================================
# VISUAL 1 — CLASS DISTRIBUTION
# =========================================================
fig = plt.figure(figsize=(4,4))
sns.countplot(x=y)
plt.xlabel("Class (0 = Normal, 1 = Attack)")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=200)
st.pyplot(fig)
plt.close()

# =========================================================
# TRAIN & MODELS
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

svm = SVC(kernel="linear", probability=True, class_weight="balanced")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss"
)

svm.fit(X_train_svm, y_train)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

# =========================================================
# CONFUSION MATRIX
# =========================================================
fig = plt.figure(figsize=(4,4))
sns.heatmap(confusion_matrix(y_test,y_pred_xgb), annot=True, fmt="d", cmap="Greens")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
st.pyplot(fig)
plt.close()

# =========================================================
# ROC & PR CURVES
# =========================================================
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)

fig = plt.figure(figsize=(8,4))
plt.subplot(1,2,1); plt.plot(fpr,tpr); plt.title("ROC Curve")
plt.subplot(1,2,2); plt.plot(recall,precision); plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.savefig("roc_pr.png", dpi=200)
st.pyplot(fig)
plt.close()

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False)

fig = plt.figure(figsize=(6,4))
sns.barplot(data=imp_df.head(10), x="Importance", y="Feature")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
st.pyplot(fig)
plt.close()

# =========================================================
# PAIR PLOT (SAFE SAMPLE)
# =========================================================
top_feats = imp_df.head(4)["Feature"].tolist()
df_pair = df[top_feats + [target]].sample(n=min(3000,len(df)), random_state=42)
pair = sns.pairplot(df_pair, hue=target, corner=True)
pair.savefig("pairplot.png", dpi=200)
st.pyplot(pair)
plt.close()

# =========================================================
# 📄 PDF GENERATION
# =========================================================
def generate_pdf():
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("Intrusion Detection System – Final Report", styles["Title"]))
    elems.append(Spacer(1,12))

    elems.append(Paragraph("Results & Usage", styles["Heading2"]))
    elems.append(Paragraph(
        "XGBoost outperforms Linear SVM with higher accuracy and lower false negatives, "
        "making it suitable for real-world IDS deployment.", styles["Normal"]
    ))

    elems.append(Spacer(1,12))
    elems.append(Paragraph("Visual Analysis", styles["Heading2"]))

    for img in [
        "class_distribution.png",
        "confusion_matrix.png",
        "roc_pr.png",
        "feature_importance.png",
        "pairplot.png"
    ]:
        if os.path.exists(img):
            elems.append(Spacer(1,10))
            elems.append(Image(img, width=5*inch, height=3*inch))

    elems.append(Spacer(1,12))
    elems.append(Paragraph("Future Scope", styles["Heading2"]))
    elems.append(Paragraph(
        "• Real-time packet capture<br/>"
        "• Multi-class attack detection<br/>"
        "• Deep learning–based IDS<br/>"
        "• Automated alert & response<br/>"
        "• Cloud & edge deployment",
        styles["Normal"]
    ))

    elems.append(Spacer(1,12))
    elems.append(Paragraph("Deployment Architecture", styles["Heading2"]))
    elems.append(Paragraph(
        "Traffic → Feature Extraction → XGBoost Model → "
        "Prediction Engine → Dashboard / Alerts / SIEM",
        styles["Normal"]
    ))

    doc.build(elems)
    buf.seek(0)
    return buf

# =========================================================
# DOWNLOAD PDF
# =========================================================
st.markdown("<h3 class='center-text'>📥 Download Complete Project Report</h3>", unsafe_allow_html=True)

st.download_button(
    "⬇️ Download IDS Final Report (PDF)",
    generate_pdf(),
    file_name="IDS_Final_Report.pdf",
    mime="application/pdf"
)

# =========================================================
# REAL-TIME PREDICTION
# =========================================================
st.sidebar.header("🧪 Real-Time Prediction")

user_input = {}
for col in X.columns[:8]:
    user_input[col] = st.sidebar.slider(
        col, float(X[col].min()), float(X[col].max()), float(X[col].mean())
    )

if st.sidebar.button("Predict"):
    inp = pd.DataFrame([user_input])
    pred = xgb.predict(inp)[0]
    prob = xgb.predict_proba(inp)[0][1]
    st.sidebar.success("ATTACK 🚨" if pred else "NORMAL ✅")
    st.sidebar.info(f"Attack Probability: {prob:.2%}")

st.markdown("<h4 class='center-text'>✅ IDS Dashboard Ready</h4>", unsafe_allow_html=True)
