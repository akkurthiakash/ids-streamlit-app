# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- GLOBAL PLOT SIZE (MEDIUM) ----------
MEDIUM_FIGSIZE = (3.5, 2.5)   # width, height in inches
plt.rcParams["figure.dpi"] = 100

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO

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
st.markdown(
    """
    <style>
    /* ===== FORCE TIMES NEW ROMAN EVERYWHERE ===== */

    /* Streamlit text blocks */
    div[data-testid="stMarkdownContainer"] * {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Headings */
    div[data-testid="stMarkdownContainer"] h2 {
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMarkdownContainer"] h3 {
        font-size: 24px !important;
        font-weight: 600 !important;
    }

    /* Body text */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        font-size: 24px !important;
        line-height: 1.6 !important;
    }

    /* Tables */
    thead tr th {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        text-align: center !important;
    }

    tbody tr td {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 22px !important;
        text-align: center !important;
    }

    /* Buttons */
    button {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 20px !important;
    }

    /* ===== TITLE (BIGGER) ===== */
    .ids-title {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 56px !important;
        font-weight: 900 !important;
        text-align: center !important;
        margin-top: 40px !important;
        margin-bottom: 40px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MAIN TITLE
# =========================================================
st.markdown(
    '<div class="ids-title">Intrusion Detection System</div>',
    unsafe_allow_html=True
)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

# =========================================================
# PROJECT OVERVIEW
# =========================================================
st.markdown("## Project Overview")

st.markdown(
    "This project implements a **Machine Learning–based Intrusion Detection System (IDS)** "
    "to classify network traffic as **Normal** or **Attack**."
)

st.markdown("### Aim")
st.markdown(
    "To detect malicious network activity using supervised machine learning models."
)

st.markdown("### Dataset")
st.markdown(
    "A labeled network traffic dataset containing normal and attack records."
)

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


# =========================================================
# FILE UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload IDS Dataset (CSV or XLSX)",
    ["csv", "xlsx"]
)

if not uploaded:
    st.info("Please upload a dataset to continue.")
    st.stop()

st.success("Dataset uploaded successfully ✔")

# =========================================================
# LOAD DATA
# =========================================================
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
st.markdown("---")

# =========================================================
# VISUALIZATIONS (11)
# =========================================================
def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

# 1. Class Distribution
st.subheader("1. Class Distribution")
st.table(y.value_counts().rename_axis("Class").reset_index(name="Count"))
fig, ax = plt.subplots()
sns.countplot(x=y, ax=ax)
show_plot(fig)

# 2. Accuracy Comparison
st.subheader("2. Accuracy Comparison")
fig, ax = plt.subplots()
sns.barplot(x=acc_df["Model"], y=acc_df["Accuracy (%)"].astype(float), ax=ax)
show_plot(fig)

# 3. Confusion Matrix
st.subheader("3. Confusion Matrix")
st.table(pd.DataFrame(cm, index=["Actual Normal","Actual Attack"],
         columns=["Pred Normal","Pred Attack"]))
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
show_plot(fig)

# 4. ROC Curve
st.subheader("4. ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],'--')
show_plot(fig)

# 5. Precision–Recall Curve
st.subheader("5. Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(recall, precision)
show_plot(fig)

# 6. Prediction Confidence
st.subheader("6. Prediction Confidence")
fig, ax = plt.subplots()
ax.hist(y_prob, bins=20)
show_plot(fig)

# 7. Error Breakdown
st.subheader("7. Error Breakdown")
error_df = pd.DataFrame({
    "Type": ["TN","FP","FN","TP"],
    "Count": [cm[0,0],cm[0,1],cm[1,0],cm[1,1]]
})
st.table(error_df)
fig, ax = plt.subplots()
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
show_plot(fig)

# 8. Feature Importance
st.subheader("8. Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(6)
st.table(imp_df)
fig, ax = plt.subplots()
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
show_plot(fig)

# 9. Feature vs Class
st.subheader("9. Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]
fig, ax = plt.subplots()
sns.boxplot(x=y, y=df[top_feat], ax=ax)
show_plot(fig)

# 10. Scatter Plot
st.subheader("10. Scatter Plot")
f1, f2 = X.columns[:2]
fig, ax = plt.subplots()
sns.scatterplot(x=df[f1], y=df[f2], hue=y, ax=ax)
show_plot(fig)

# 11. Pair Plot
st.subheader("11. Pair Plot")
pair_fig = sns.pairplot(df[[f1,f2,target]].sample(300), hue=target)
st.pyplot(pair_fig.fig)
plt.close("all")

st.success("Dashboard execution completed successfully ✔")

def generate_ids_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    # ---- Force Times New Roman in PDF ----
    styles["Title"].fontName = "Times-Roman"
    styles["Heading2"].fontName = "Times-Roman"
    styles["Normal"].fontName = "Times-Roman"

    elements = []

    # -------- TITLE --------
    elements.append(Paragraph(
        "<b>Intrusion Detection System – Final Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 12))

    # -------- PROJECT OVERVIEW --------
    elements.append(Paragraph("<b>Project Overview</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "This project implements a Machine Learning–based Intrusion Detection System (IDS) "
        "to classify network traffic as Normal or Attack using supervised learning models.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # -------- ACCURACY TABLE --------
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

    # -------- HELPER TO ADD PLOTS --------
    def add_plot(fig, title):
        elements.append(Paragraph(title, styles["Heading2"]))
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        elements.append(
            Image(buf, width=5.2*inch, height=3.3*inch)
        )
        elements.append(PageBreak())

    # -------- ADD ALL 11 PLOTS --------

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    sns.countplot(x=y, ax=ax)
    add_plot(fig, "1. Class Distribution")

    # 2. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    ax.bar(["SVM","XGBoost"], [acc_svm*100, acc_xgb*100])
    ax.set_ylabel("Accuracy (%)")
    add_plot(fig, "2. Accuracy Comparison")

    # 3. Confusion Matrix
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    add_plot(fig, "3. Confusion Matrix")

    # 4. ROC Curve
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],'--')
    add_plot(fig, "4. ROC Curve")

    # 5. Precision–Recall Curve
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    ax.plot(recall, precision)
    add_plot(fig, "5. Precision–Recall Curve")

    # 6. Prediction Confidence
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    ax.hist(y_prob, bins=20)
    add_plot(fig, "6. Prediction Confidence")

    # 7. Error Breakdown
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    ax.bar(["TN","FP","FN","TP"], cm.flatten())
    add_plot(fig, "7. Error Breakdown")

    # 8. Feature Importance
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    sns.barplot(data=imp_df.head(6), x="Importance", y="Feature", ax=ax)
    add_plot(fig, "8. Feature Importance")

    # 9. Feature vs Class
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    sns.boxplot(x=y, y=df[top_feat], ax=ax)
    add_plot(fig, "9. Feature vs Class")

    # 10. Scatter Plot
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    sns.scatterplot(x=df[f1], y=df[f2], hue=y, ax=ax)
    add_plot(fig, "10. Scatter Plot")

    # 11. Pair Plot
    pair_fig = sns.pairplot(df[[f1,f2,target]].sample(300), hue=target)
    buf = BytesIO()
    pair_fig.fig.savefig(buf, format="png", dpi=120)
    plt.close("all")
    buf.seek(0)
    elements.append(Paragraph("11. Pair Plot", styles["Heading2"]))
    elements.append(Image(buf, width=5.2*inch, height=5.2*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Full IDS Report")

st.download_button(
    label="Download Complete IDS Report (PDF)",
    data=generate_ids_pdf(),
    file_name="IDS_Final_Report.pdf",
    mime="application/pdf"
)







