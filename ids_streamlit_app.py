# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL STABLE VERSION
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
    accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intrusion Detection System",
    layout="wide"
)

# ---------------- GLOBAL STYLE (TIMES NEW ROMAN) ----------------
st.markdown(
    """
    <style>
    div[data-testid="stMarkdownContainer"] * {
        font-family: "Times New Roman", Times, serif !important;
    }

    h2 { font-size: 24px !important; font-weight: 700 !important; }
    h3 { font-size: 24px !important; font-weight: 600 !important; }

    p, li { font-size: 20px !important; line-height: 1.6 !important; }

    thead tr th {
        font-size: 20px !important;
        font-weight: 800 !important;
        text-align: center !important;
    }

    tbody tr td {
        font-size: 20px !important;
        text-align: center !important;
    }

    button {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 20px !important;
    }
    
    .ids-title {
        font-size: 44px !important;
        font-weight: 900 !important;
        text-align: center !important;
        margin-top: 40px !important;
        margin-bottom: 40px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- PLOT SIZE CONTROL ----------------
MEDIUM_FIGSIZE = (5.2, 3.2)
plt.rcParams["figure.dpi"] = 110

def show_plot(fig):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# =========================================================
# TITLE
# =========================================================
st.markdown('<div class="ids-title">Intrusion Detection System</div>', unsafe_allow_html=True)

# =========================================================
# PROJECT OVERVIEW
# =========================================================
st.markdown("## Project Overview")

st.markdown(
    "This project implements a **Machine Learning–based Intrusion Detection System (IDS)** "
    "to classify network traffic as **Normal** or **Attack**."
)

st.markdown("### Aim")
st.markdown("To detect malicious network activity using supervised machine learning models.")

st.markdown("### Dataset")
st.markdown("A labeled network traffic dataset containing normal and attack records.")

st.markdown("### Algorithms Used")
st.markdown("""
- **Linear SVM** – baseline classification model  
- **XGBoost** – advanced ensemble model with higher accuracy
""")

st.markdown("### Outcomes")
st.markdown("""
- XGBoost outperforms Linear SVM  
- Reduced missed attack detection  
- Improved intrusion detection reliability
""")

st.markdown("---")

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv", "xlsx"])
if not uploaded:
    st.stop()

@st.cache_data
def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl").dropna().drop_duplicates()
    return pd.read_csv(file).dropna().drop_duplicates()

df = load_data(uploaded)
target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

# =========================================================
# MODEL TRAINING
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
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist"
    )

    svm.fit(X_train_svm, y_train)
    xgb.fit(X_train, y_train)

    return svm, xgb, X_test, X_test_svm, y_test

svm, xgb, X_test, X_test_svm, y_test = train_models(X, y)

# =========================================================
# METRICS
# =========================================================
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]

acc_svm = accuracy_score(y_test, svm.predict(X_test_svm))
acc_xgb = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# =========================================================
# ACCURACY SUMMARY
# =========================================================
st.markdown("## Accuracy Summary")
acc_df = pd.DataFrame({
    "Model": ["Linear SVM", "XGBoost"],
    "Accuracy (%)": [f"{acc_svm*100:.2f}", f"{acc_xgb*100:.2f}"]
})
st.table(acc_df)

# =========================================================
# VISUALIZATIONS (11)
# =========================================================

# 1. Class Distribution
st.subheader("1. Class Distribution")
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.countplot(x=y, ax=ax)
ax.set_ylabel("Count")
show_plot(fig)

# 2. Accuracy Comparison
st.subheader("2. Accuracy Comparison")
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.bar(acc_df["Model"], acc_df["Accuracy (%)"].astype(float))
ax.set_ylabel("Accuracy (%)")
show_plot(fig)

# 3. Confusion Matrix
st.subheader("3. Confusion Matrix")
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
show_plot(fig)

# 4. ROC Curve
st.subheader("4. ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],'--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
show_plot(fig)

# 5. Precision–Recall Curve
st.subheader("5. Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.plot(recall, precision)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
show_plot(fig)

# 6. Prediction Confidence
st.subheader("6. Prediction Confidence")
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.hist(y_prob, bins=20)
ax.set_xlabel("Attack Probability")
show_plot(fig)

# 7. Error Breakdown
st.subheader("7. Error Breakdown")
error_df = pd.DataFrame({
    "Type": ["TN","FP","FN","TP"],
    "Count": [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})
st.table(error_df)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
show_plot(fig)

# 8. Feature Importance
st.subheader("8. Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(6)
st.table(imp_df)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
show_plot(fig)

# 9. Feature vs Class
st.subheader("9. Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.boxplot(x=y, y=df[top_feat], ax=ax)
show_plot(fig)

# 10. Scatter Plot
st.subheader("10. Scatter Plot")
f1, f2 = X.columns[:2]
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.scatterplot(x=df[f1], y=df[f2], hue=y, ax=ax)
show_plot(fig)

# 11. Pair Plot
st.subheader("11. Pair Plot")
pair_fig = sns.pairplot(df[[f1,f2,target]].sample(300), hue=target)
pair_fig.fig.set_size_inches(4.8,4.8)
st.pyplot(pair_fig.fig, use_container_width=False)
plt.close("all")

st.success("Dashboard rendered successfully ✔")

def generate_ids_pdf_medium_plots():
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Times-Roman"
    styles["Heading2"].fontName = "Times-Roman"
    styles["Normal"].fontName = "Times-Roman"

    elements = []

    # ---------- TITLE ----------
    elements.append(Paragraph(
        "<b>Intrusion Detection System – Final Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 12))

    # ---------- PROJECT OVERVIEW ----------
    elements.append(Paragraph("<b>Project Overview</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "This project implements a Machine Learning–based Intrusion Detection System (IDS) "
        "to classify network traffic as Normal or Attack using supervised learning models.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # ---------- ACCURACY TABLE ----------
    acc_table = Table([
        ["Model", "Accuracy (%)"],
        ["Linear SVM", f"{acc_svm*100:.2f}"],
        ["XGBoost", f"{acc_xgb*100:.2f}"]
    ])
    acc_table.setStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Times-Bold")
    ])
    elements.append(acc_table)
    elements.append(PageBreak())

    # ---------- HELPER FUNCTION ----------
    def add_plot(fig, title):
        elements.append(Paragraph(title, styles["Heading2"]))
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        # Medium size that fits A4 perfectly
        elements.append(
            Image(buf, width=5.0*inch, height=3.1*inch)
        )
        elements.append(PageBreak())

    # ---------- 11 PLOTS ----------

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.countplot(x=y, ax=ax)
    add_plot(fig, "1. Class Distribution")

    # 2. Accuracy Comparison
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.bar(["SVM", "XGBoost"], [acc_svm*100, acc_xgb*100])
    ax.set_ylabel("Accuracy (%)")
    add_plot(fig, "2. Accuracy Comparison")

    # 3. Confusion Matrix
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    add_plot(fig, "3. Confusion Matrix")

    # 4. ROC Curve
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    add_plot(fig, "4. ROC Curve")

    # 5. Precision–Recall Curve
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    add_plot(fig, "5. Precision–Recall Curve")

    # 6. Prediction Confidence
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.hist(y_prob, bins=20)
    ax.set_xlabel("Attack Probability")
    add_plot(fig, "6. Prediction Confidence")

    # 7. Error Breakdown
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.bar(["TN","FP","FN","TP"], [cm[0,0],cm[0,1],cm[1,0],cm[1,1]])
    add_plot(fig, "7. Error Breakdown")

    # 8. Feature Importance
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.barplot(data=imp_df.head(6), x="Importance", y="Feature", ax=ax)
    add_plot(fig, "8. Feature Importance")

    # 9. Feature vs Class
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.boxplot(x=y, y=df[top_feat], ax=ax)
    add_plot(fig, "9. Feature vs Class")

    # 10. Scatter Plot
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.scatterplot(x=df[f1], y=df[f2], hue=y, ax=ax)
    add_plot(fig, "10. Scatter Plot")

    # 11. Pair Plot
    pair_fig = sns.pairplot(df[[f1,f2,target]].sample(300), hue=target)
    pair_fig.fig.set_size_inches(4.8, 4.8)
    buf = BytesIO()
    pair_fig.fig.savefig(buf, format="png", dpi=110)
    plt.close("all")
    buf.seek(0)
    elements.append(Paragraph("11. Pair Plot", styles["Heading2"]))
    elements.append(Image(buf, width=4.8*inch, height=4.8*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Full IDS Report")

st.download_button(
    label="Download IDS Report (Medium-Sized Plots)",
    data=generate_ids_pdf_medium_plots(),
    file_name="IDS_Final_Report_Medium_Plots.pdf",
    mime="application/pdf"
)




