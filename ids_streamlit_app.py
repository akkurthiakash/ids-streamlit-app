# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL VERSION
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
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
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
    /* ===== GLOBAL FONT: TIMES NEW ROMAN ===== */
    html, body, [class*="css"], [class*="st-"] {
        font-family: "Times New Roman", Times, serif !important;
    }

    /* ===== MAIN TITLE ===== */
    .ids-title {
        font-family: "Times New Roman", Times, serif !important;
        font-size: 46px !important;
        font-weight: 900 !important;
        text-align: center !important;
        margin-top: 10px !important;
        margin-bottom: 25px !important;
    }

    /* ===== SIDE / SECTION HEADINGS ===== */
    h2 {
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 22px !important;
        font-weight: 600 !important;
    }

    /* ===== BODY TEXT (MEDIUM SIZE) ===== */
    p, li, div, label, span {
        font-size: 17px !important;
        line-height: 1.6 !important;
    }

    /* ===== TABLE HEADERS ===== */
    thead tr th {
        font-size: 17px !important;
        font-weight: 800 !important;
        text-align: center !important;
    }

    /* ===== TABLE BODY ===== */
    tbody tr td {
        font-size: 16px !important;
        text-align: center !important;
    }

    /* ===== BUTTONS ===== */
    button {
        font-size: 16px !important;
        font-family: "Times New Roman", Times, serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE UPLOAD
# =========================================================
uploaded = st.file_uploader(
    "Upload IDS Dataset (CSV or XLSX)",
    ["csv", "xlsx"]
)

if uploaded:
    st.success("Dataset uploaded successfully ✔")
else:
    st.info("Please upload a dataset to continue.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl").dropna().drop_duplicates()
    return pd.read_csv(file).dropna().drop_duplicates()

df = load_data(uploaded)
st.write("Dataset shape:", df.shape)

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
# HELPER FUNCTION
# =========================================================
def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

# =========================================================
# VISUALIZATIONS (1–11)
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
ax.plot([0,1],[0,1],'--')
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
    "Sample": range(1,9),
    "Attack Probability": np.round(y_prob_xgb[:8],4)
}))
fig, ax = plt.subplots()
ax.hist(y_prob_xgb, bins=20)
show_plot(fig)

# 7. Error Breakdown
st.subheader("7. Error Breakdown")
error_df = pd.DataFrame({
    "Type":["TN","FP","FN","TP"],
    "Count":[cm[0,0],cm[0,1],cm[1,0],cm[1,1]]
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
st.table(df_vis[[f1,f2,target]].sample(8, random_state=1))
fig, ax = plt.subplots()
sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
show_plot(fig)

# 11. Pair Plot
st.subheader("11. Pair Plot")
pair_df = df_vis[[f1,f2,target]].sample(min(300,len(df_vis)), random_state=1)
pair_fig = sns.pairplot(pair_df, hue=target)
st.pyplot(pair_fig.fig)
plt.close("all")

st.markdown("---")

# =========================================================
# PDF DOWNLOAD (ALL 11 PLOTS)
# =========================================================
def generate_full_pdf_all_plots():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Intrusion Detection System – Final Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        "XGBoost outperforms Linear SVM and improves intrusion detection accuracy.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    acc_table = Table([
        ["Model", "Accuracy (%)"],
        ["Linear SVM", f"{acc_svm*100:.2f}"],
        ["XGBoost", f"{acc_xgb*100:.2f}"]
    ])
    acc_table.setStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER")
    ])
    elements.append(acc_table)
    elements.append(PageBreak())

    def add_plot(fig, title):
        elements.append(Paragraph(title, styles["Heading2"]))
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        elements.append(Image(buf, width=5.5*inch, height=3.5*inch))
        elements.append(PageBreak())

    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    add_plot(fig, "1. Class Distribution")

    fig, ax = plt.subplots()
    ax.bar(["SVM","XGBoost"], [acc_svm*100, acc_xgb*100])
    add_plot(fig, "2. Accuracy Comparison")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    add_plot(fig, "3. Confusion Matrix")

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
    add_plot(fig, "4. ROC Curve")

    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    add_plot(fig, "5. Precision–Recall Curve")

    fig, ax = plt.subplots()
    ax.hist(y_prob_xgb, bins=20)
    add_plot(fig, "6. Prediction Confidence")

    fig, ax = plt.subplots()
    ax.bar(["TN","FP","FN","TP"], [cm[0,0],cm[0,1],cm[1,0],cm[1,1]])
    add_plot(fig, "7. Error Breakdown")

    fig, ax = plt.subplots()
    sns.barplot(data=imp_df.head(6), x="Importance", y="Feature", ax=ax)
    add_plot(fig, "8. Feature Importance")

    fig, ax = plt.subplots()
    sns.boxplot(x=df_vis[target], y=df_vis[top_feat], ax=ax)
    add_plot(fig, "9. Feature vs Class")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
    add_plot(fig, "10. Scatter Plot")

    pair_fig = sns.pairplot(pair_df, hue=target)
    buf = BytesIO()
    pair_fig.fig.savefig(buf, format="png", dpi=120)
    plt.close("all")
    buf.seek(0)
    elements.append(Paragraph("11. Pair Plot", styles["Heading2"]))
    elements.append(Image(buf, width=5.5*inch, height=5.5*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## Download Complete IDS Report")

st.download_button(
    "Download Full IDS Report with All 11 Plots",
    data=generate_full_pdf_all_plots(),
    file_name="IDS_Full_Report_All_Visualizations.pdf",
    mime="application/pdf"
)

