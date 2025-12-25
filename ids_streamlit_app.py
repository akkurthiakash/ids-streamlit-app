# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL STREAMLIT-SAFE VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intrusion Detection System",
    layout="wide"
)

# ---------------- GLOBAL STYLE (SAFE) ----------------
st.markdown(
    """
    <style>
    /* Use Streamlit default background */
    .stApp {
        background-color: transparent;
        color: inherit;
    }

    /* Times New Roman everywhere */
    html, body, [class*="css"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* Table styling */
    table {
        border-collapse: collapse !important;
    }
    thead tr th {
        font-size: 17px !important;
        font-weight: bold !important;
    }
    tbody tr td {
        font-size: 16px !important;
    }

    /* Plot size control */
    canvas, img {
        max-width: 750px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TITLE
# =========================================================
st.markdown(
    "<h1 style='text-align:center;'>Intrusion Detection System </h1>",
    unsafe_allow_html=True
)

# =========================================================
# PROJECT OVERVIEW (NORMAL TEXT — NO BACKGROUND)
# =========================================================
st.markdown("## Project Overview")

st.markdown("""
This project develops a Machine Learning–based Intrusion Detection System (IDS)
to identify malicious network activity in network traffic.

### Aim
To classify network traffic accurately as Normal or Attack.

### Dataset
A labeled network traffic dataset containing normal and attack records.

### Algorithms Used
-  SVM – baseline machine learning model  
- XGBoost – advanced model with improved detection accuracy  

### Evaluation
The system performance is evaluated using accuracy, confusion matrix,
ROC curve, precision–recall analysis, and other visual analytics.

### Outcomes
- XGBoost achieves higher accuracy than SVM  
- Reduced missed attack detection  
- Improved intrusion detection reliability  

### Importance in Data Science
This project demonstrates the practical application of data science in
cybersecurity through preprocessing, machine learning modeling,
evaluation, and visualization.
""")

st.markdown("---")

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl").dropna().drop_duplicates()
    return pd.read_csv(file).dropna().drop_duplicates()

uploaded = st.file_uploader(
    "Upload IDS Dataset (CSV or XLSX)",
    ["csv", "xlsx"]
)

if uploaded is None:
    st.stop()

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
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_svm = scaler.fit_transform(X_train)
    X_test_svm = scaler.transform(X_test)

    svm = SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    )

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
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

acc_svm = accuracy_score(y_test, svm.predict(X_test_svm))
acc_xgb = accuracy_score(y_test, y_pred_xgb)
cm = confusion_matrix(y_test, y_pred_xgb)

df_vis = df.sample(min(1500, len(df)), random_state=42)

# =========================================================
# ACCURACY TABLE (TOP)
# =========================================================
st.markdown("## Accuracy Summary")

acc_df = pd.DataFrame({
    "Model": ["Linear SVM", "XGBoost"],
    "Accuracy (%)": [f"{acc_svm*100:.2f}", f"{acc_xgb*100:.2f}"]
})

st.table(acc_df)

st.markdown("---")

# =========================================================
# HELPER FUNCTION
# =========================================================
def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

# =========================================================
# 11 VISUALIZATIONS (ORDERED)
# =========================================================

# 1. Class Distribution
st.subheader("1. Class Distribution")
st.table(y.value_counts().rename_axis("Class").reset_index(name="Count"))
fig, ax = plt.subplots()
sns.countplot(x=y, ax=ax)
show_plot(fig)

# 2. Accuracy Comparison
st.subheader("2. Accuracy Comparison")
st.table(acc_df)
fig, ax = plt.subplots()
sns.barplot(x=acc_df["Model"], y=acc_df["Accuracy (%)"].astype(float), ax=ax)
show_plot(fig)

# 3. Confusion Matrix
st.subheader("3. Confusion Matrix")
cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Attack"],
    columns=["Pred Normal", "Pred Attack"]
)
st.table(cm_df)
fig, ax = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
show_plot(fig)

# 4. ROC Curve
st.subheader("4. ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
st.table(pd.DataFrame({"FPR": fpr, "TPR": tpr}).iloc[::max(1, len(fpr)//6)])
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0,1], [0,1], "--")
show_plot(fig)

# 5. Precision–Recall Curve
st.subheader("5. Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)
st.table(pd.DataFrame({"Recall": recall, "Precision": precision}).iloc[::max(1, len(recall)//6)])
fig, ax = plt.subplots()
ax.plot(recall, precision)
show_plot(fig)

# 6. Prediction Confidence
st.subheader("6. Prediction Confidence")
st.table(pd.DataFrame({
    "Sample": range(1, 9),
    "Attack Probability": np.round(y_prob_xgb[:8], 4)
}))
fig, ax = plt.subplots()
ax.hist(y_prob_xgb, bins=20)
show_plot(fig)

# 7. Error Breakdown
st.subheader("7. Error Breakdown")
error_df = pd.DataFrame({
    "Type": ["TN", "FP", "FN", "TP"],
    "Count": [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
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
st.table(df_vis.groupby(target)[top_feat].mean().reset_index())
fig, ax = plt.subplots()
sns.boxplot(x=df_vis[target], y=df_vis[top_feat], ax=ax)
show_plot(fig)

# 10. Scatter Plot
st.subheader("10. Scatter Plot")
f1, f2 = X.columns[:2]
st.table(df_vis[[f1, f2, target]].sample(8, random_state=1))
fig, ax = plt.subplots()
sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
show_plot(fig)

# 11. Pair Plot
st.subheader("11. Pair Plot")
pair_df = df_vis[[f1, f2, target]].sample(min(300, len(df_vis)), random_state=1)
pair_fig = sns.pairplot(pair_df, hue=target)
st.pyplot(pair_fig.fig)
plt.close("all")

st.markdown(
    "<h4 style='text-align:center;'>Dashboard Execution Completed Successfully</h4>",
    unsafe_allow_html=True
)

def generate_full_pdf(
    acc_svm, acc_xgb, y, cm, fpr, tpr,
    precision, recall, y_prob_xgb, imp_df, df_vis, X, target
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ---------------- TITLE ----------------
    elements.append(Paragraph("Intrusion Detection System – Final Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # ---------------- PROJECT OVERVIEW ----------------
    elements.append(Paragraph("<b>Project Overview</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    overview_text = """
    This project develops a Machine Learning–based Intrusion Detection System (IDS)
    to classify network traffic as Normal or Attack.<br/><br/>
    <b>Algorithms Used:</b> Linear SVM and XGBoost.<br/>
    <b>Outcome:</b> XGBoost achieves higher accuracy and reduces missed attacks.
    """
    elements.append(Paragraph(overview_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # ---------------- ACCURACY TABLE ----------------
    elements.append(Paragraph("<b>Accuracy Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    acc_table = [
        ["Model", "Accuracy (%)"],
        ["Linear SVM", f"{acc_svm*100:.2f}"],
        ["XGBoost", f"{acc_xgb*100:.2f}"]
    ]
    elements.append(Table(acc_table))
    elements.append(PageBreak())

    # ---------------- HELPER: ADD FIGURE ----------------
    def add_plot(fig, title):
        elements.append(Paragraph(title, styles["Heading3"]))
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        elements.append(Image(buf, width=5.5*inch, height=3.5*inch))
        elements.append(PageBreak())

    # ---------------- 1. CLASS DISTRIBUTION ----------------
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    add_plot(fig, "1. Class Distribution")

    # ---------------- 2. ACCURACY COMPARISON ----------------
    fig, ax = plt.subplots()
    ax.bar(["SVM", "XGBoost"], [acc_svm*100, acc_xgb*100])
    ax.set_ylabel("Accuracy (%)")
    add_plot(fig, "2. Accuracy Comparison")

    # ---------------- 3. CONFUSION MATRIX ----------------
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    add_plot(fig, "3. Confusion Matrix")

    # ---------------- 4. ROC CURVE ----------------
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],'--')
    add_plot(fig, "4. ROC Curve")

    # ---------------- 5. PRECISION–RECALL ----------------
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    add_plot(fig, "5. Precision–Recall Curve")

    # ---------------- 6. PREDICTION CONFIDENCE ----------------
    fig, ax = plt.subplots()
    ax.hist(y_prob_xgb, bins=20)
    add_plot(fig, "6. Prediction Confidence")

    # ---------------- 7. ERROR BREAKDOWN ----------------
    fig, ax = plt.subplots()
    ax.bar(["TN","FP","FN","TP"], [cm[0,0], cm[0,1], cm[1,0], cm[1,1]])
    add_plot(fig, "7. Error Breakdown")

    # ---------------- 8. FEATURE IMPORTANCE ----------------
    fig, ax = plt.subplots()
    sns.barplot(data=imp_df.head(6), x="Importance", y="Feature", ax=ax)
    add_plot(fig, "8. Feature Importance")

    # ---------------- 9. FEATURE VS CLASS ----------------
    top_feat = imp_df.iloc[0]["Feature"]
    fig, ax = plt.subplots()
    sns.boxplot(x=df_vis[target], y=df_vis[top_feat], ax=ax)
    add_plot(fig, "9. Feature vs Class")

    # ---------------- 10. SCATTER PLOT ----------------
    f1, f2 = X.columns[:2]
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
    add_plot(fig, "10. Scatter Plot")

    # ---------------- 11. PAIR PLOT ----------------
    pair_df = df_vis[[f1, f2, target]].sample(min(300, len(df_vis)), random_state=1)
    pair_fig = sns.pairplot(pair_df, hue=target)
    buf = BytesIO()
    pair_fig.fig.savefig(buf, format="png", dpi=120)
    plt.close("all")
    buf.seek(0)
    elements.append(Paragraph("11. Pair Plot", styles["Heading3"]))
    elements.append(Image(buf, width=5.5*inch, height=5.5*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## Download Full Project Report")

st.download_button(
    label="Download IDS Complete Report (PDF)",
    data=generate_full_pdf(
        acc_svm, acc_xgb, y, cm,
        fpr, tpr, precision, recall,
        y_prob_xgb, imp_df, df_vis, X, target
    ),
    file_name="IDS_Full_Project_Report.pdf",
    mime="application/pdf"
)
