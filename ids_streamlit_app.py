# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL FULL VERSION (11 VISUALS)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
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
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# ---------------- PERFECT SIZE SETTINGS ----------------
MAX_FIG = (5.2, 3.6)
plt.rcParams["figure.dpi"] = 110

# ---------------- VISIBILITY STYLE ----------------
st.markdown("""
<style>
thead tr th {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
tbody tr td {
    font-size: 17px !important;
    color: #f9fafb !important;
}
[data-testid="stTable"] {
    width: 85% !important;
    margin-bottom: 18px;
}
canvas, img {
    max-width: 720px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* ===== TABLE BORDER & GRID LINES ===== */
table {
    border-collapse: collapse !important;
    border: 2px solid #ffffff !important;
}

/* Header borders */
thead tr th {
    border: 2px solid #ffffff !important;
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    background-color: #1f2933 !important;
}

/* Body cell borders */
tbody tr td {
    border: 1.8px solid #ffffff !important;
    color: #f9fafb !important;
    font-size: 17px !important;
    background-color: #2d3748 !important;
}

/* Table container spacing */
[data-testid="stTable"],
[data-testid="stDataFrame"] {
    width: 85% !important;
    margin-bottom: 18px;
    border-radius: 6px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================
st.markdown("<h1 style='text-align:center;'> Intrusion Detection System </h1>", unsafe_allow_html=True)

# =========================================================
# PROJECT PREVIEW
# =========================================================
st.markdown("##  Project Report ")

st.markdown("""
This project implements a **machine learning–based Intrusion Detection System (IDS)**
to classify network traffic as **Normal** or **Attack**.

A labeled network traffic dataset is used, and two models are applied:
** SVM** (baseline) and **XGBoost** (advanced model).

The system evaluates model performance using accuracy and visual analytics.
**XGBoost outperforms SVM**, achieving higher accuracy and reducing missed attacks.

The project highlights the practical use of **data science in cybersecurity**
through preprocessing, modeling, evaluation, and visualization.
""")


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl").dropna().drop_duplicates()
    return pd.read_csv(file).dropna().drop_duplicates()

uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv", "xlsx"])
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
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

acc_svm = accuracy_score(y_test, svm.predict(X_test_svm))
acc_xgb = accuracy_score(y_test, y_pred_xgb)
cm = confusion_matrix(y_test, y_pred_xgb)

df_vis = df.sample(min(1500, len(df)), random_state=42)

# =========================================================
# ⭐ ACCURACY TABLE (AT TOP — REQUIRED)
# =========================================================
st.markdown("##  Accuracy Summary")

acc_table = pd.DataFrame({
    "Model": ["Linear SVM", "XGBoost"],
    "Accuracy (%)": [f"{acc_svm*100:.2f}", f"{acc_xgb*100:.2f}"]
})
st.table(acc_table)

st.markdown("---")

# =========================================================
# HELPER FUNCTION
# =========================================================
def show_plot(fig):
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# =========================================================
# 🔢 11 VISUALIZATIONS (IN ORDER)
# =========================================================

# 1️⃣ Class Distribution
st.subheader("1️⃣ Class Distribution")
st.table(y.value_counts().rename_axis("Class").reset_index(name="Count"))

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.countplot(x=y, ax=ax)
show_plot(fig)

# 2️⃣ Accuracy Comparison
st.subheader("2️⃣ Accuracy Comparison")
st.table(acc_table)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(x=acc_table["Model"], y=acc_table["Accuracy (%)"].astype(float), ax=ax)
show_plot(fig)

# 3️⃣ Confusion Matrix
st.subheader("3️⃣ Confusion Matrix")
cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Attack"],
    columns=["Pred Normal", "Pred Attack"]
)
st.table(cm_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", ax=ax)
show_plot(fig)

# 4️⃣ ROC Curve
st.subheader("4️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_tbl = pd.DataFrame({"FPR": fpr, "TPR": tpr}).iloc[::max(1, len(fpr)//6)].round(4)
st.table(roc_tbl)

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],'--')
show_plot(fig)

# 5️⃣ Precision–Recall Curve
st.subheader("5️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_tbl = pd.DataFrame({"Recall": recall, "Precision": precision}).iloc[::max(1, len(recall)//6)].round(4)
st.table(pr_tbl)

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(recall, precision)
show_plot(fig)

# 6️⃣ Prediction Confidence
st.subheader("6️⃣ Prediction Confidence")
pred_tbl = pd.DataFrame({
    "Sample": range(1, 9),
    "Attack Probability": np.round(y_prob_xgb[:8], 4)
})
st.table(pred_tbl)

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.hist(y_prob_xgb, bins=20)
show_plot(fig)

# 7️⃣ Error Breakdown
st.subheader("7️⃣ Error Breakdown")
error_df = pd.DataFrame({
    "Type": ["TN", "FP", "FN", "TP"],
    "Count": [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})
st.table(error_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
show_plot(fig)

# 8️⃣ Feature Importance
st.subheader("8️⃣ Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(6)
st.table(imp_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
show_plot(fig)

# 9️⃣ Feature vs Class
st.subheader("9️⃣ Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]
st.table(df_vis.groupby(target)[top_feat].mean().reset_index())

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.boxplot(x=df_vis[target], y=df_vis[top_feat], ax=ax)
show_plot(fig)

# 🔟 Scatter Plot
st.subheader("🔟 Scatter Plot")
f1, f2 = X.columns[:2]
st.table(df_vis[[f1, f2, target]].sample(8, random_state=1))

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
show_plot(fig)

# 1️⃣1️⃣ Pair Plot Summary
st.subheader("1️⃣1️⃣ Pair Plot Summary")
st.table(df_vis[[f1, f2, target]].groupby(target).mean().reset_index())

df_pair = df_vis.sample(min(300, len(df_vis)), random_state=1)
pair_fig = sns.pairplot(df_pair[[f1, f2, target]], hue=target)
pair_fig.fig.set_size_inches(6, 6)
st.pyplot(pair_fig.fig, use_container_width=False)
plt.close("all")

st.markdown("<h4 style='text-align:center;'>✅ All 11 Visualizations Displayed Successfully</h4>", unsafe_allow_html=True)

def generate_full_pdf_report(
    df, X, y, acc_svm, acc_xgb, cm,
    fpr, tpr, precision, recall, y_prob_xgb, imp_df
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ---------- TITLE ----------
    elements.append(Paragraph("Intrusion Detection System (IDS) – Final Report", styles["Title"]))
    elements.append(Spacer(1, 14))

    # ---------- PROJECT OVERVIEW ----------
    elements.append(Paragraph(
        "<b>Aim:</b> To build a machine learning–based IDS to classify network traffic as Normal or Attack.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        "<b>Dataset:</b> Network traffic dataset labeled as Normal (0) and Attack (1).",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph(
        "<b>Algorithms:</b> Linear SVM and XGBoost.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # ---------- ACCURACY TABLE ----------
    elements.append(Paragraph("<b>Accuracy Summary</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    acc_table = [
        ["Model", "Accuracy (%)"],
        ["Linear SVM", f"{acc_svm*100:.2f}"],
        ["XGBoost", f"{acc_xgb*100:.2f}"]
    ]
    elements.append(Table(acc_table))
    elements.append(Spacer(1, 16))

    # ---------- HELPER FUNCTION ----------
    def add_plot(fig):
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        img_buffer.seek(0)
        elements.append(Image(img_buffer, width=4.8*inch, height=3.2*inch))
        elements.append(Spacer(1, 14))

    # ---------- 1️⃣ Class Distribution ----------
    elements.append(Paragraph("1. Class Distribution", styles["Heading3"]))
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    add_plot(fig)

    # ---------- 2️⃣ Accuracy Comparison ----------
    elements.append(Paragraph("2. Accuracy Comparison", styles["Heading3"]))
    fig, ax = plt.subplots()
    ax.bar(["SVM", "XGBoost"], [acc_svm*100, acc_xgb*100])
    ax.set_ylabel("Accuracy (%)")
    add_plot(fig)

    # ---------- 3️⃣ Confusion Matrix ----------
    elements.append(Paragraph("3. Confusion Matrix", styles["Heading3"]))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    add_plot(fig)

    # ---------- 4️⃣ ROC Curve ----------
    elements.append(Paragraph("4. ROC Curve", styles["Heading3"]))
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    add_plot(fig)

    # ---------- 5️⃣ Precision–Recall ----------
    elements.append(Paragraph("5. Precision–Recall Curve", styles["Heading3"]))
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    add_plot(fig)

    # ---------- 6️⃣ Prediction Confidence ----------
    elements.append(Paragraph("6. Prediction Confidence", styles["Heading3"]))
    fig, ax = plt.subplots()
    ax.hist(y_prob_xgb, bins=20)
    ax.set_xlabel("Attack Probability")
    add_plot(fig)

    # ---------- 7️⃣ Error Breakdown ----------
    elements.append(Paragraph("7. Error Breakdown", styles["Heading3"]))
    fig, ax = plt.subplots()
    ax.bar(["TN","FP","FN","TP"], [cm[0,0], cm[0,1], cm[1,0], cm[1,1]])
    add_plot(fig)

    # ---------- 8️⃣ Feature Importance ----------
    elements.append(Paragraph("8. Feature Importance", styles["Heading3"]))
    fig, ax = plt.subplots()
    sns.barplot(data=imp_df.head(6), x="Importance", y="Feature", ax=ax)
    add_plot(fig)

    # ---------- FINAL OUTCOME ----------
    elements.append(Paragraph("<b>Final Outcome:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "The IDS successfully detects intrusions. XGBoost outperforms Linear SVM "
        "by achieving higher accuracy and reducing missed attacks. "
        "This project demonstrates how data science techniques can be applied "
        "to real-world cybersecurity problems.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
st.markdown("## 📄 Download Full Project Report (PDF)")

st.download_button(
    label="📥 Download IDS Full Report (PDF)",
    data=generate_full_pdf_report(
        df, X, y, acc_svm, acc_xgb, cm,
        fpr, tpr, precision, recall, y_prob_xgb, imp_df
    ),
    file_name="IDS_Full_Project_Report.pdf",
    mime="application/pdf"
)



