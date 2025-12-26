# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL COMPLETE VERSION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ---------------- GLOBAL STYLE ----------------
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] * {
    font-family: "Times New Roman", Times, serif !important;
}
h2 { font-size: 28px !important; font-weight: 700 !important; }
h3 { font-size: 22px !important; font-weight: 600 !important; }
p, li { font-size: 17px !important; line-height: 1.6 !important; }

table, th, td {
    border: 2px solid white !important;
    border-collapse: collapse !important;
}
thead tr th {
    font-weight: 800 !important;
    text-align: center !important;
}
tbody tr td {
    text-align: center !important;
}

.ids-title {
    font-size: 56px !important;
    font-weight: 900 !important;
    text-align: center !important;
    margin: 40px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PLOT SIZE ----------------
MEDIUM_FIGSIZE = (5.2, 3.2)
plt.rcParams["figure.dpi"] = 110

def show_plot(fig):
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# =========================================================
# TITLE
# =========================================================
st.markdown('<div class="ids-title">Intrusion Detection System</div>', unsafe_allow_html=True)

# =========================================================
# PROJECT PREVIEW
# =========================================================
st.markdown("## Project Preview")

st.markdown("""
- This project focuses on building a **Machine Learning–based Intrusion Detection System (IDS)**.
- The system classifies network traffic into **Normal** and **Attack** categories.
- A labeled network traffic dataset is used for training and testing.
- ** SVM** and **XGBoost** algorithms are implemented and compared.
- Performance is evaluated using accuracy, ROC curve, confusion matrix, and other metrics.
- Results show that **XGBoost outperforms  SVM** in intrusion detection.
- The project demonstrates the application of **data science in cybersecurity**.
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
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", tree_method="hist"
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
# VISUALIZATIONS (11) WITH TABLE + REPORT
# =========================================================

# 1. Class Distribution
st.subheader("1. Class Distribution")
st.markdown("Shows the distribution of Normal and Attack samples.")
st.table(y.value_counts().rename_axis("Class").reset_index(name="Count"))
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.countplot(x=y, ax=ax)
show_plot(fig)

# 2. Accuracy Comparison
st.subheader("2. Accuracy Comparison")
st.markdown("Compares accuracy of Linear SVM and XGBoost.")
st.table(acc_df)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.bar(acc_df["Model"], acc_df["Accuracy (%)"].astype(float))
show_plot(fig)

# 3. Confusion Matrix
st.subheader("3. Confusion Matrix")
st.markdown("Displays correct and incorrect predictions.")
cm_df = pd.DataFrame(cm, index=["Actual Normal","Actual Attack"],
                     columns=["Pred Normal","Pred Attack"])
st.table(cm_df)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
show_plot(fig)

# 4. ROC Curve
st.subheader("4. ROC Curve")
st.markdown("Shows trade-off between true positive and false positive rates.")
fpr, tpr, _ = roc_curve(y_test, y_prob)
st.table(pd.DataFrame({"FPR": fpr, "TPR": tpr}).iloc[::max(1,len(fpr)//8)])
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
show_plot(fig)

# 5. Precision–Recall Curve
st.subheader("5. Precision–Recall Curve")
st.markdown("Useful for evaluating performance on imbalanced data.")
precision, recall, _ = precision_recall_curve(y_test, y_prob)
st.table(pd.DataFrame({"Recall": recall, "Precision": precision}).iloc[::max(1,len(recall)//8)])
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.plot(recall, precision)
show_plot(fig)

# 6. Prediction Confidence
st.subheader("6. Prediction Confidence")
st.markdown("Shows model confidence for predicting attacks.")
st.table(pd.DataFrame({"Attack Probability": np.round(y_prob[:8],4)}))
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
ax.hist(y_prob, bins=20)
show_plot(fig)

# 7. Error Breakdown
st.subheader("7. Error Breakdown")
st.markdown("Breakdown of true and false predictions.")
error_df = pd.DataFrame({"Type":["TN","FP","FN","TP"],
                         "Count":[cm[0,0],cm[0,1],cm[1,0],cm[1,1]]})
st.table(error_df)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
show_plot(fig)

# 8. Feature Importance
st.subheader("8. Feature Importance")
st.markdown("Identifies most influential features.")
imp_df = pd.DataFrame({"Feature":X.columns,
                       "Importance":xgb.feature_importances_}).sort_values("Importance",ascending=False).head(6)
st.table(imp_df)
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
show_plot(fig)

# 9. Feature vs Class
st.subheader("9. Feature vs Class")
st.markdown("Shows feature variation across classes.")
top_feat = imp_df.iloc[0]["Feature"]
st.table(df.groupby(target)[top_feat].mean().reset_index())
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.boxplot(x=y, y=df[top_feat], ax=ax)
show_plot(fig)

# 10. Scatter Plot
st.subheader("10. Scatter Plot")
st.markdown("Visualizes relationship between two features.")
f1, f2 = X.columns[:2]
st.table(df[[f1,f2,target]].sample(8))
fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
sns.scatterplot(x=df[f1], y=df[f2], hue=y, ax=ax)
show_plot(fig)

# 11. Pair Plot
st.subheader("11. Pair Plot")
st.markdown("Shows multivariate relationships.")
st.table(df[[f1,f2,target]].groupby(target).mean().reset_index())
pair_fig = sns.pairplot(df[[f1,f2,target]].sample(300), hue=target)
pair_fig.fig.set_size_inches(4.8,4.8)
st.pyplot(pair_fig.fig, use_container_width=False)
plt.close("all")

st.success("IDS Dashboard executed successfully ✔")

def generate_ids_pdf_with_tables_reports():
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Force Times New Roman
    for k in styles.byName:
        styles[k].fontName = "Times-Roman"

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12
    )

    normal = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontSize=11,
        spaceAfter=6
    )

    elements = []

    # ================= TITLE =================
    elements.append(Paragraph(
        "<b>Intrusion Detection System – Final Report</b>",
        styles["Title"]
    ))
    elements.append(Spacer(1, 12))

    # ================= PROJECT OVERVIEW =================
    elements.append(Paragraph("<b>Project Overview</b>", heading))
    elements.append(Paragraph(
        "This project implements a Machine Learning–based Intrusion Detection System (IDS) "
        "to classify network traffic as Normal or Attack. Linear SVM and XGBoost models are "
        "used, where XGBoost achieves superior performance.",
        normal
    ))

    # ================= ACCURACY TABLE =================
    acc_table = Table([
        ["Model", "Accuracy (%)"],
        ["Linear SVM", f"{acc_svm*100:.2f}"],
        ["XGBoost", f"{acc_xgb*100:.2f}"]
    ])
    acc_table.setStyle([
        ("GRID", (0,0), (-1,-1), 1.5, colors.black),
        ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER")
    ])
    elements.append(acc_table)
    elements.append(PageBreak())

    # ================= HELPER =================
    def add_section(title, report_text, table_df, fig):
        elements.append(Paragraph(title, heading))
        elements.append(Paragraph(report_text, normal))

        # ---- TABLE ----
        if table_df is not None:
            table_data = [table_df.columns.tolist()] + table_df.values.tolist()
            t = Table(table_data)
            t.setStyle([
                ("GRID", (0,0), (-1,-1), 1.2, colors.black),
                ("FONTNAME", (0,0), (-1,0), "Times-Bold"),
                ("ALIGN", (0,0), (-1,-1), "CENTER")
            ])
            elements.append(t)
            elements.append(Spacer(1, 10))

        # ---- IMAGE ----
        if fig is not None:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            elements.append(
                Image(buf, width=5.0*inch, height=3.1*inch)
            )

        elements.append(PageBreak())

    # ================= VISUALIZATIONS =================

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.countplot(x=y, ax=ax)
    add_section(
        "1. Class Distribution",
        "Shows the number of Normal and Attack samples in the dataset.",
        y.value_counts().rename_axis("Class").reset_index(name="Count"),
        fig
    )

    # 2. Accuracy Comparison
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.bar(acc_df["Model"], acc_df["Accuracy (%)"].astype(float))
    add_section(
        "2. Accuracy Comparison",
        "Compares accuracy of Linear SVM and XGBoost models.",
        acc_df,
        fig
    )

    # 3. Confusion Matrix
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    cm_df = pd.DataFrame(cm, index=["Actual Normal","Actual Attack"],
                         columns=["Pred Normal","Pred Attack"])
    add_section(
        "3. Confusion Matrix",
        "Shows correct and incorrect predictions made by the model.",
        cm_df,
        fig
    )

    # 4. ROC Curve
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr}).iloc[::max(1,len(fpr)//8)]
    add_section(
        "4. ROC Curve",
        "Illustrates the trade-off between true positive and false positive rates.",
        roc_df,
        fig
    )

    # 5. Precision–Recall Curve
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.plot(recall, precision)
    pr_df = pd.DataFrame({"Recall": recall, "Precision": precision}).iloc[::max(1,len(recall)//8)]
    add_section(
        "5. Precision–Recall Curve",
        "Shows precision vs recall relationship for attack detection.",
        pr_df,
        fig
    )

    # 6. Prediction Confidence
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.hist(y_prob, bins=20)
    conf_df = pd.DataFrame({"Attack Probability": np.round(y_prob[:10],4)})
    add_section(
        "6. Prediction Confidence",
        "Displays model confidence for predicting attacks.",
        conf_df,
        fig
    )

    # 7. Error Breakdown
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax.bar(["TN","FP","FN","TP"], [cm[0,0],cm[0,1],cm[1,0],cm[1,1]])
    error_df = pd.DataFrame({
        "Type":["TN","FP","FN","TP"],
        "Count":[cm[0,0],cm[0,1],cm[1,0],cm[1,1]]
    })
    add_section(
        "7. Error Breakdown",
        "Summarizes correct and incorrect classifications.",
        error_df,
        fig
    )

    # 8. Feature Importance
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
    add_section(
        "8. Feature Importance",
        "Shows most influential features for attack detection.",
        imp_df,
        fig
    )

    # 9. Feature vs Class
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.boxplot(x=y, y=df[top_feat], ax=ax)
    feat_df = df.groupby(target)[top_feat].mean().reset_index()
    add_section(
        "9. Feature vs Class",
        "Compares feature behavior across traffic classes.",
        feat_df,
        fig
    )

    # 10. Scatter Plot
    fig, ax = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.scatterplot(x=df[f1], y=df[f2], hue=y, ax=ax)
    scatter_df = df[[f1,f2,target]].sample(10)
    add_section(
        "10. Scatter Plot",
        "Visualizes relationship between two features.",
        scatter_df,
        fig
    )

    # 11. Pair Plot
    pair_fig = sns.pairplot(df[[f1,f2,target]].sample(300), hue=target)
    pair_fig.fig.set_size_inches(4.8,4.8)
    buf = BytesIO()
    pair_fig.fig.savefig(buf, format="png", dpi=110)
    plt.close("all")
    buf.seek(0)

    elements.append(Paragraph("11. Pair Plot", heading))
    elements.append(Paragraph(
        "Shows multivariate relationships among selected features.",
        normal
    ))
    elements.append(Image(buf, width=4.8*inch, height=4.8*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Full IDS Report")

st.download_button(
    label="Download IDS Report (Tables + Reports + Plots)",
    data=generate_ids_pdf_with_tables_reports(),
    file_name="IDS_Final_Report_Full.pdf",
    mime="application/pdf"
)

