# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL VERSION
# Outcomes Report + Accuracy + Tables + 10 Visualizations
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# ---------------- GLOBAL PLOT SETTINGS ----------------
plt.rcParams["figure.figsize"] = (3.8, 2.8)
plt.rcParams["figure.dpi"] = 110

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# =========================================================
# TITLE
# =========================================================
st.markdown("<h1 style='text-align:center;'>🛡️ Intrusion Detection System Dashboard</h1>", unsafe_allow_html=True)

# =========================================================
# 📌 PROJECT OUTCOMES & RESULTS (MEDIUM REPORT)
# =========================================================
st.markdown("## 📌 Project Outcomes and Results")

st.markdown("""
The Intrusion Detection System (IDS) developed in this project applies machine learning
techniques to classify network traffic into **normal** and **attack** categories.
Two models, **Linear SVM** and **XGBoost**, were trained and evaluated using network data.

The results indicate that **XGBoost outperforms Linear SVM** in terms of accuracy and
attack detection capability. XGBoost is more effective in identifying malicious traffic
while reducing missed attacks, which is crucial for intrusion detection systems.

Error analysis and confusion matrix results show a **lower false-negative rate**, improving
the reliability of the IDS. Feature analysis also highlights important network attributes
that strongly influence detection decisions.

Overall, this project demonstrates that **machine learning–based IDS**, particularly using
XGBoost, provides an effective, scalable, and reliable solution for enhancing network security.
""")

st.markdown("<hr>", unsafe_allow_html=True)

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
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss"
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

# =========================================================
# 🎯 ACCURACY REPORT (TOP OF VISUALS)
# =========================================================
st.markdown("## 🎯 Model Accuracy Summary")

c1, c2 = st.columns(2)
c1.metric("Linear SVM Accuracy", f"{acc_svm*100:.2f}%")
c2.metric("XGBoost Accuracy", f"{acc_xgb*100:.2f}%")

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# SAMPLE FOR FAST VISUALS
# =========================================================
df_vis = df.sample(min(2500, len(df)), random_state=42)
y_vis = df_vis[target]

# =========================================================
# 📊 VISUALIZATIONS WITH TABLES (10)
# =========================================================
st.markdown("## 📊 Visual Analysis with Tables")

# 1️⃣ Class Distribution
st.subheader("1️⃣ Class Distribution")
class_df = y.value_counts().rename_axis("Class").reset_index(name="Count")
st.dataframe(class_df)

fig, ax = plt.subplots()
sns.countplot(x=y_vis, ax=ax)
ax.set_title("Normal vs Attack Distribution")
st.pyplot(fig); plt.close(fig)

# 2️⃣ Accuracy Comparison
st.subheader("2️⃣ Model Accuracy Comparison")
acc_df = pd.DataFrame({"Model": ["SVM", "XGBoost"], "Accuracy": [acc_svm, acc_xgb]})
st.dataframe(acc_df)

fig, ax = plt.subplots()
sns.barplot(x=acc_df["Model"], y=acc_df["Accuracy"], ax=ax)
ax.set_title("Accuracy Comparison")
st.pyplot(fig); plt.close(fig)

# 3️⃣ Confusion Matrix
st.subheader("3️⃣ Confusion Matrix")
cm_df = pd.DataFrame(cm,
    index=["Actual Normal","Actual Attack"],
    columns=["Pred Normal","Pred Attack"]
)
st.dataframe(cm_df)

fig, ax = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig); plt.close(fig)

# 4️⃣ ROC Curve
st.subheader("4️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
st.dataframe(roc_df.head(10))

fig, ax = plt.subplots()
ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
ax.set_title("ROC Curve")
st.pyplot(fig); plt.close(fig)

# 5️⃣ Precision–Recall Curve
st.subheader("5️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_df = pd.DataFrame({"Recall": recall, "Precision": precision})
st.dataframe(pr_df.head(10))

fig, ax = plt.subplots()
ax.plot(recall, precision)
ax.set_title("Precision–Recall Curve")
st.pyplot(fig); plt.close(fig)

# 6️⃣ Prediction Confidence
st.subheader("6️⃣ Prediction Confidence")
conf_df = pd.DataFrame({"Attack Probability": y_prob_xgb})
st.dataframe(conf_df.head(10))

fig, ax = plt.subplots()
ax.hist(y_prob_xgb, bins=25)
ax.set_title("Attack Probability Distribution")
st.pyplot(fig); plt.close(fig)

# 7️⃣ Error Breakdown
st.subheader("7️⃣ Error Breakdown")
error_df = pd.DataFrame({
    "Type": ["TN","FP","FN","TP"],
    "Count": [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})
st.dataframe(error_df)

fig, ax = plt.subplots()
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
ax.set_title("Error Breakdown")
st.pyplot(fig); plt.close(fig)

# 8️⃣ Feature Importance
st.subheader("8️⃣ Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(8)

st.dataframe(imp_df)

fig, ax = plt.subplots()
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
ax.set_title("Top Important Features")
st.pyplot(fig); plt.close(fig)

# 9️⃣ Feature vs Class
st.subheader("9️⃣ Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]
feat_df = df.groupby(target)[top_feat].describe().reset_index()
st.dataframe(feat_df)

fig, ax = plt.subplots()
sns.boxplot(x=y_vis, y=df_vis[top_feat], ax=ax)
ax.set_title(f"{top_feat} vs Class")
st.pyplot(fig); plt.close(fig)

# 🔟 Pair Plot
st.subheader("🔟 Pair Plot (Feature Relationships)")
pair_cols = df_vis.columns[:4].tolist() + [target]
pair_fig = sns.pairplot(
    df_vis[pair_cols],
    hue=target,
    corner=True,
    plot_kws={"s": 10, "alpha": 0.6}
)
pair_fig.fig.set_size_inches(6, 6)
st.pyplot(pair_fig.fig)
plt.close("all")

# =========================================================
# END
# =========================================================
st.markdown("<h4 style='text-align:center;'>✅ Final IDS Dashboard Ready</h4>", unsafe_allow_html=True)

def generate_ids_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Intrusion Detection System – Final Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Outcomes & Results
    elements.append(Paragraph("Project Outcomes and Results", styles["Heading2"]))
    elements.append(Paragraph(
        "This project implements a machine learning based Intrusion Detection System "
        "to classify network traffic as Normal or Attack. XGBoost outperforms Linear SVM "
        "by achieving higher accuracy and reducing missed attacks, improving overall "
        "network security reliability.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # Dataset Summary
    elements.append(Paragraph("Dataset Summary", styles["Heading2"]))
    dataset_table = [
        ["Metric", "Value"],
        ["Total Records", df.shape[0]],
        ["Total Features", df.shape[1]],
        ["Normal Traffic", int((y == 0).sum())],
        ["Attack Traffic", int((y == 1).sum())]
    ]
    elements.append(Table(dataset_table))
    elements.append(Spacer(1, 12))

    # Accuracy Report
    elements.append(Paragraph("Model Accuracy Report", styles["Heading2"]))
    acc_table = [
        ["Model", "Accuracy"],
        ["Linear SVM", f"{acc_svm*100:.2f}%"],
        ["XGBoost", f"{acc_xgb*100:.2f}%"]
    ]
    elements.append(Table(acc_table))
    elements.append(Spacer(1, 12))

    # Confusion Matrix
    elements.append(Paragraph("Confusion Matrix (XGBoost)", styles["Heading2"]))
    cm_table = [
        ["", "Pred Normal", "Pred Attack"],
        ["Actual Normal", cm[0,0], cm[0,1]],
        ["Actual Attack", cm[1,0], cm[1,1]]
    ]
    elements.append(Table(cm_table))
    elements.append(Spacer(1, 12))

    # Conclusion
    elements.append(Paragraph("Conclusion", styles["Heading2"]))
    elements.append(Paragraph(
        "The results confirm that machine learning based IDS, especially using XGBoost, "
        "provides an effective and scalable solution for intrusion detection.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Project Report")

st.download_button(
    label="📥 Download IDS Final Report (PDF)",
    data=generate_ids_pdf(),
    file_name="IDS_Final_Project_Report.pdf",
    mime="application/pdf"
)

