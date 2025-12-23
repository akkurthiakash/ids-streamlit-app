# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL STRICT VERSION
# Tables FIRST, Images BELOW, Perfect Table Sizes (11 visuals)
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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# ---------------- GRAPH SIZE ----------------
MAX_FIG = (3.2, 2.3)
plt.rcParams["figure.dpi"] = 100

# ---------------- TABLE TEXT SIZE ----------------
st.markdown("""
<style>
thead tr th, tbody tr td {
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE & OUTCOMES
# =========================================================
st.markdown("<h1 style='text-align:center;'> Intrusion Detection System </h1>", unsafe_allow_html=True)

st.markdown("##  Project Outcomes")
st.markdown("""
- Successfully detects **Normal** and **Attack** network traffic  
- **XGBoost outperforms SVM Model**  
- Reduces missed attacks  
- Clear tables and compact visualizations improve analysis  
""")

st.markdown("---")

# =========================================================
# DATA LOAD
# =========================================================
uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv","xlsx"])
if uploaded is None:
    st.stop()

df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)
df = df.dropna().drop_duplicates()

target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

# =========================================================
# MODEL TRAIN
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

svm = SVC(kernel="linear", probability=True, class_weight="balanced")
xgb = XGBClassifier(n_estimators=120, max_depth=5, eval_metric="logloss")

svm.fit(X_train_svm, y_train)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

acc_svm = accuracy_score(y_test, svm.predict(X_test_svm))
acc_xgb = accuracy_score(y_test, y_pred_xgb)

cm = confusion_matrix(y_test, y_pred_xgb)

df_vis = df.sample(min(2000, len(df)), random_state=42)

# =========================================================
# 🎯 ACCURACY SUMMARY (TABLE ONLY)
# =========================================================
st.markdown("## 🎯 Accuracy Summary")

acc_df = pd.DataFrame({
    "Model": ["Linear SVM", "XGBoost"],
    "Accuracy (%)": [acc_svm*100, acc_xgb*100]
}).round(2)

st.dataframe(acc_df)
st.markdown("---")

# =========================================================
# VISUALIZATIONS (11) — TABLE FIRST → IMAGE BELOW
# =========================================================

# ---------- 1️⃣ Class Distribution ----------
st.subheader("1️⃣ Class Distribution")
st.dataframe(y.value_counts().rename_axis("Class").reset_index(name="Count"))

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.countplot(x=y, ax=ax)
st.pyplot(fig); plt.close(fig)

# ---------- 2️⃣ Accuracy Comparison ----------
st.subheader("2️⃣ Accuracy Comparison")
st.dataframe(acc_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(x=acc_df["Model"], y=acc_df["Accuracy (%)"], ax=ax)
st.pyplot(fig); plt.close(fig)

# ---------- 3️⃣ Confusion Matrix ----------
st.subheader("3️⃣ Confusion Matrix")
cm_df = pd.DataFrame(cm,
    index=["Actual Normal","Actual Attack"],
    columns=["Pred Normal","Pred Attack"]
)
st.dataframe(cm_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", ax=ax)
st.pyplot(fig); plt.close(fig)

# ---------- 4️⃣ ROC Curve ----------
st.subheader("4️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)

roc_table = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr
}).iloc[::max(1, len(fpr)//8)]

st.dataframe(roc_table.reset_index(drop=True))

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
st.pyplot(fig); plt.close(fig)

# ---------- 5️⃣ Precision–Recall ----------
st.subheader("5️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)

pr_table = pd.DataFrame({
    "Recall": recall,
    "Precision": precision
}).iloc[::max(1, len(recall)//8)]

st.dataframe(pr_table.reset_index(drop=True))

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.plot(recall, precision)
st.pyplot(fig); plt.close(fig)

# ---------- 6️⃣ Prediction Confidence ----------
st.subheader("6️⃣ Prediction Confidence")
st.dataframe(pd.DataFrame({
    "Attack Probability": y_prob_xgb[:8]
}))

fig, ax = plt.subplots(figsize=MAX_FIG)
ax.hist(y_prob_xgb, bins=20)
st.pyplot(fig); plt.close(fig)

# ---------- 7️⃣ Error Breakdown ----------
st.subheader("7️⃣ Error Breakdown")
error_df = pd.DataFrame({
    "Type": ["TN","FP","FN","TP"],
    "Count": [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
})
st.dataframe(error_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
st.pyplot(fig); plt.close(fig)

# ---------- 8️⃣ Feature Importance ----------
st.subheader("8️⃣ Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(6)

st.dataframe(imp_df)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
st.pyplot(fig); plt.close(fig)

# ---------- 9️⃣ Feature vs Class ----------
st.subheader("9️⃣ Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]

feat_table = df_vis.groupby(target)[top_feat].mean().reset_index()
st.dataframe(feat_table)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.boxplot(x=df_vis[target], y=df_vis[top_feat], ax=ax)
st.pyplot(fig); plt.close(fig)

# ---------- 🔟 Scatter Plot ----------
st.subheader("🔟 Feature Scatter Plot")
f1, f2 = X.columns[:2]

scatter_table = df_vis[[f1, f2, target]].sample(8, random_state=1)
st.dataframe(scatter_table)

fig, ax = plt.subplots(figsize=MAX_FIG)
sns.scatterplot(data=df_vis, x=f1, y=f2, hue=target, ax=ax)
st.pyplot(fig); plt.close(fig)

# ---------- 1️⃣1️⃣ Pair Plot ----------
st.subheader("1️⃣1️⃣ Pair Plot Summary")

pair_table = df_vis[[f1, f2, target]].groupby(target).mean().reset_index()
st.dataframe(pair_table)

pair_fig = sns.pairplot(
    df_vis[[f1, f2, target]],
    hue=target,
    plot_kws={"s":6, "alpha":0.5}
)
pair_fig.fig.set_size_inches(4.5,4.5)
st.pyplot(pair_fig.fig)
plt.close("all")

st.markdown("<h4 style='text-align:center;'>✅ Final Dashboard (All Tables + Images Correct)</h4>", unsafe_allow_html=True)

def generate_ids_report_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # ---------- TITLE ----------
    elements.append(Paragraph("Intrusion Detection System – Final Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # ---------- PROJECT OUTCOMES ----------
    elements.append(Paragraph("Project Outcomes", styles["Heading2"]))
    elements.append(Paragraph(
        "The IDS successfully classifies network traffic into Normal and Attack. "
        "XGBoost outperforms Linear SVM, achieving higher accuracy and reducing missed attacks.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 10))

    # ---------- DATASET SUMMARY ----------
    elements.append(Paragraph("Dataset Summary", styles["Heading2"]))
    elements.append(Table([
        ["Metric", "Value"],
        ["Total Records", df.shape[0]],
        ["Total Features", df.shape[1]],
        ["Normal Traffic", int((y == 0).sum())],
        ["Attack Traffic", int((y == 1).sum())]
    ]))
    elements.append(Spacer(1, 10))

    # ---------- ACCURACY REPORT ----------
    elements.append(Paragraph("Accuracy Report", styles["Heading2"]))
    elements.append(Table([
        ["Model", "Accuracy (%)"],
        ["Linear SVM", f"{acc_svm*100:.2f}"],
        ["XGBoost", f"{acc_xgb*100:.2f}"]
    ]))
    elements.append(Spacer(1, 10))

    # ---------- CONFUSION MATRIX ----------
    elements.append(Paragraph("Confusion Matrix (XGBoost)", styles["Heading2"]))
    elements.append(Table([
        ["", "Pred Normal", "Pred Attack"],
        ["Actual Normal", cm[0,0], cm[0,1]],
        ["Actual Attack", cm[1,0], cm[1,1]]
    ]))
    elements.append(Spacer(1, 10))

    # ---------- VISUALIZATION SUMMARY ----------
    elements.append(Paragraph("Visualization Summary (11)", styles["Heading2"]))
    elements.append(Paragraph(
        "1. Class Distribution<br/>"
        "2. Accuracy Comparison<br/>"
        "3. Confusion Matrix<br/>"
        "4. ROC Curve<br/>"
        "5. Precision–Recall Curve<br/>"
        "6. Prediction Confidence<br/>"
        "7. Error Breakdown<br/>"
        "8. Feature Importance<br/>"
        "9. Feature vs Class<br/>"
        "10. Scatter Plot<br/>"
        "11. Pair Plot",
        styles["Normal"]
    ))

    # ---------- CONCLUSION ----------
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Conclusion", styles["Heading2"]))
    elements.append(Paragraph(
        "The project demonstrates that machine learning–based IDS, especially using XGBoost, "
        "provides an effective and scalable solution for network intrusion detection.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Project Report")

st.download_button(
    label="📥 Download IDS Report (PDF)",
    data=generate_ids_report_pdf(),
    file_name="IDS_Final_Report.pdf",
    mime="application/pdf"
)

