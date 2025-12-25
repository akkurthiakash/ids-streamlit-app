# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL STREAMLIT-SAFE VERSION
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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# ---------------- FORCE SMALL IMAGE SIZE ----------------
MAX_FIG = (2.8, 2.0)
plt.rcParams["figure.dpi"] = 100

# ---------------- LIMIT IMAGE WIDTH (CRITICAL) ----------------
st.markdown("""
<style>
img, canvas {
    max-width: 420px !important;
}
thead tr th, tbody tr td {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE & BRIEF PROJECT DESCRIPTION
# =========================================================
st.markdown("<h1 style='text-align:center;'> Intrusion Detection System</h1>", unsafe_allow_html=True)

st.markdown("##  Project Preview")

st.markdown("""
### Intrusion Detection System (IDS) – Project Overview

**Aim:**  
The aim of this project is to develop a machine learning–based Intrusion Detection System (IDS) that classifies network traffic as Normal or Attack.

**Dataset:**  
A network traffic dataset containing labeled records:
- 0 – Normal traffic  
- 1 – Attack traffic  
The dataset is cleaned by removing missing values and duplicates.

**Algorithms Used:**  
- Linear Support Vector Machine (SVM) – baseline model  
- XGBoost – advanced model with higher accuracy  

**Purpose:**  
- Automate intrusion detection  
- Compare machine learning models  
- Analyze results using a dashboard  

**Outcomes:**  
- Accurate classification of network traffic  
- XGBoost performs better than SVM  
- Reduced missed attacks  

**Usefulness in Data Science:**  
Demonstrates data preprocessing, model evaluation, visualization, and real-world application of machine learning.
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

df_vis = df.sample(min(1200, len(df)), random_state=42)

# =========================================================
# ACCURACY TABLE
# =========================================================
st.markdown("## 🎯 Accuracy Summary")

acc_df = pd.DataFrame({
    "Model": ["Linear SVM", "XGBoost"],
    "Accuracy (%)": [acc_svm*100, acc_xgb*100]
}).round(2)

st.dataframe(acc_df)
st.markdown("---")

# =========================================================
# HELPER FUNCTION (SMALL IMAGE)
# =========================================================
def show_plot(fig):
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# =========================================================
# VISUALIZATIONS (TABLE FIRST → IMAGE BELOW)
# =========================================================

# 1️⃣ Class Distribution
st.subheader("1️⃣ Class Distribution")
st.dataframe(y.value_counts().rename_axis("Class").reset_index(name="Count"))

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.countplot(x=y, ax=ax)
ax.set_xlabel("Class")
ax.set_ylabel("Count")
show_plot(fig)

# 2️⃣ Accuracy Comparison
st.subheader("2️⃣ Accuracy Comparison")
st.dataframe(acc_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(x=acc_df["Model"], y=acc_df["Accuracy (%)"], ax=ax)
ax.set_ylabel("Accuracy (%)")
show_plot(fig)

# 3️⃣ Confusion Matrix
st.subheader("3️⃣ Confusion Matrix")
cm_df = pd.DataFrame(cm,
    index=["Actual Normal","Actual Attack"],
    columns=["Pred Normal","Pred Attack"]
)
st.dataframe(cm_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", ax=ax)
show_plot(fig)

# 4️⃣ ROC Curve
st.subheader("4️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)

roc_tbl = pd.DataFrame({"FPR": fpr, "TPR": tpr}).iloc[::max(1, len(fpr)//8)]
st.dataframe(roc_tbl)

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],'--')
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
show_plot(fig)

# 5️⃣ Precision–Recall Curve
st.subheader("5️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)

pr_tbl = pd.DataFrame({"Recall": recall, "Precision": precision}).iloc[::max(1, len(recall)//8)]
st.dataframe(pr_tbl)

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(recall, precision)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
show_plot(fig)

# 6️⃣ Prediction Confidence
st.subheader("6️⃣ Prediction Confidence")
st.dataframe(pd.DataFrame({"Attack Probability": y_prob_xgb[:8]}))

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.hist(y_prob_xgb, bins=20)
ax.set_xlabel("Probability")
ax.set_ylabel("Frequency")
show_plot(fig)

# 7️⃣ Error Breakdown
st.subheader("7️⃣ Error Breakdown")
error_df = pd.DataFrame({
    "Type":["TN","FP","FN","TP"],
    "Count":[cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})
st.dataframe(error_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
show_plot(fig)

# 8️⃣ Feature Importance
st.subheader("8️⃣ Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(6)
st.dataframe(imp_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
show_plot(fig)

# 9️⃣ Feature vs Class
st.subheader("9️⃣ Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]
st.dataframe(df_vis.groupby(target)[top_feat].mean().reset_index())

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.boxplot(x=df_vis[target], y=df_vis[top_feat], ax=ax)
show_plot(fig)

# 🔟 Scatter Plot
st.subheader("🔟 Scatter Plot")
f1, f2 = X.columns[:2]
st.dataframe(df_vis[[f1, f2, target]].sample(8, random_state=1))

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
show_plot(fig)

# 1️⃣1️⃣ Pair Plot (SAFE)
st.subheader("1️⃣1️⃣ Pair Plot Summary")
st.dataframe(df_vis[[f1, f2, target]].groupby(target).mean().reset_index())

df_pair = df_vis.sample(min(250, len(df_vis)), random_state=1)
pair_fig = sns.pairplot(df_pair[[f1, f2, target]], hue=target, plot_kws={"s":6, "alpha":0.5})
pair_fig.fig.set_size_inches(4, 4)
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

    elements.append(Paragraph("IDS Final Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(
        "The IDS effectively detects intrusions. XGBoost outperforms Linear SVM with higher accuracy.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))
    elements.append(Table([
        ["Model", "Accuracy (%)"],
        ["Linear SVM", f"{acc_svm*100:.2f}"],
        ["XGBoost", f"{acc_xgb*100:.2f}"]
    ]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Report")
st.download_button(
    "📥 Download IDS Report (PDF)",
    generate_pdf(),
    file_name="IDS_Final_Report.pdf",
    mime="application/pdf"
)

st.markdown("<h4 style='text-align:center;'>✅ Dashboard Ready & Streamlit-Safe</h4>", unsafe_allow_html=True)

