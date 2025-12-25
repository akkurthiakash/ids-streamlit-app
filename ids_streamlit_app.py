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
    .ids-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 56px;          /* BIG */
        font-weight: 900;         /* THICK */
        text-align: center;
        margin-top: 40px;         /* PUSH DOWN */
        margin-bottom: 40px;      /* SPACE BELOW */
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

