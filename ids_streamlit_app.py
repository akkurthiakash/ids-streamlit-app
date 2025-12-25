# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL VERSION
# Beautiful UI | 11 Visualizations | PDF Report
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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Intrusion Detection System",
    layout="wide"
)

# ---------------- GLOBAL SIZE ----------------
MAX_FIG = (5.2, 3.6)
plt.rcParams["figure.dpi"] = 110

# ---------------- BEAUTIFUL BACKGROUND + FONT ----------------
st.markdown("""
<style>

/* ===== GLOBAL FONT ===== */
html, body, [class*="css"] {
    font-family: "Times New Roman", Times, serif !important;
}

/* ===== APP BACKGROUND ===== */
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
    </style>
    """,
    unsafe_allow_html=True
)


/* ===== HEADINGS ===== */
h1, h2, h3, h4 {
    color: #ffffff !important;
    text-align: center;
}

/* ===== TABLE STYLE ===== */
table {
    border-collapse: collapse !important;
    background-color: #262730 !important;
    color: #ffffff !important;
    border: 1.8px solid #ffffff !important;
}

thead tr th {
    background-color: #1f2933 !important;
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border: 1.8px solid #ffffff !important;
    padding: 10px 14px !important;
    text-align: center !important;
}

tbody tr td {
    background-color: #262730 !important;
    color: #f9fafb !important;
    font-size: 17px !important;
    border: 1.5px solid #ffffff !important;
    padding: 9px 14px !important;
    text-align: center !important;
}

[data-testid="stTable"],
[data-testid="stDataFrame"] {
    width: 85% !important;
    margin: auto;
    margin-bottom: 20px;
}

/* ===== IMAGE SIZE ===== */
canvas, img {
    max-width: 720px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.markdown(
    """
    <h1 style="
        text-align: center;
        font-family: 'Times New Roman', Times, serif;
        color: white;
    ">
         Intrusion Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

# =========================================================
# BRIEF PROJECT REPORT
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
- Linear SVM – baseline machine learning model  
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
cybersecurity through data preprocessing, machine learning modeling,
evaluation, and visualization.
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
# ACCURACY TABLE (TOP)
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
# 11 VISUALIZATIONS (TABLE → IMAGE)
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
cm_df = pd.DataFrame(cm,
    index=["Actual Normal","Actual Attack"],
    columns=["Pred Normal","Pred Attack"]
)
st.table(cm_df)
fig, ax = plt.subplots(figsize=MAX_FIG)
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", ax=ax)
show_plot(fig)

# 4️⃣ ROC Curve
st.subheader("4️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_tbl = pd.DataFrame({"FPR": fpr, "TPR": tpr}).iloc[::max(1,len(fpr)//6)].round(4)
st.table(roc_tbl)
fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
show_plot(fig)

# 5️⃣ Precision–Recall Curve
st.subheader("5️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_tbl = pd.DataFrame({"Recall": recall, "Precision": precision}).iloc[::max(1,len(recall)//6)].round(4)
st.table(pr_tbl)
fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(recall, precision)
show_plot(fig)

# 6️⃣ Prediction Confidence
st.subheader("6️⃣ Prediction Confidence")
pred_tbl = pd.DataFrame({
    "Sample": range(1,9),
    "Attack Probability": np.round(y_prob_xgb[:8],4)
})
st.table(pred_tbl)
fig, ax = plt.subplots(figsize=MAX_FIG)
ax.hist(y_prob_xgb, bins=20)
show_plot(fig)

# 7️⃣ Error Breakdown
st.subheader("7️⃣ Error Breakdown")
error_df = pd.DataFrame({
    "Type":["TN","FP","FN","TP"],
    "Count":[cm[0,0],cm[0,1],cm[1,0],cm[1,1]]
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
st.table(df_vis[[f1,f2,target]].sample(8, random_state=1))
fig, ax = plt.subplots(figsize=MAX_FIG)
sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
show_plot(fig)

# 1️⃣1️⃣ Pair Plot Summary
st.subheader("1️⃣1️⃣ Pair Plot Summary")
st.table(df_vis[[f1,f2,target]].groupby(target).mean().reset_index())
df_pair = df_vis.sample(min(300,len(df_vis)), random_state=1)
pair_fig = sns.pairplot(df_pair[[f1,f2,target]], hue=target)
pair_fig.fig.set_size_inches(6,6)
st.pyplot(pair_fig.fig, use_container_width=False)
plt.close("all")

# =========================================================
# PDF DOWNLOAD
# =========================================================
def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Intrusion Detection System (IDS) – Final Report", styles["Title"]))
    elements.append(Spacer(1,12))
    elements.append(Paragraph(
        "XGBoost achieves higher accuracy than Linear SVM and improves intrusion detection reliability.",
        styles["Normal"]
    ))
    elements.append(Spacer(1,12))
    elements.append(Table([
        ["Model","Accuracy (%)"],
        ["Linear SVM",f"{acc_svm*100:.2f}"],
        ["XGBoost",f"{acc_xgb*100:.2f}"]
    ]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Project Report")
st.download_button(
    "📥 Download IDS Report (PDF)",
    generate_pdf(),
    file_name="IDS_Project_Report.pdf",
    mime="application/pdf"
)

st.markdown("<h4 style='text-align:center;'>✅ Final Dashboard Ready</h4>", unsafe_allow_html=True)




