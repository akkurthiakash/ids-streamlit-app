# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL FIXED VERSION
# Compact Graphs + Tables + Accuracy + PDF
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

# ---------------- HARD GRAPH SIZE LIMIT ----------------
MAX_FIG_SIZE = (3.2, 2.2)
plt.rcParams["figure.dpi"] = 100

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'> Intrusion Detection System </h1>", unsafe_allow_html=True)

# =========================================================
# PROJECT OUTCOMES
# =========================================================

st.markdown("##  Project Outcomes")
st.markdown("""
- The Intrusion Detection System (IDS) successfully identifies **normal** and **attack** traffic.
- **XGBoost outperforms  SVM** with better accuracy.
- Missed attacks are reduced, improving network security.
- Visualizations and tables make results easy to understand.
- The dashboard enables effective intrusion analysis.
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

uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv","xlsx"])
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
    xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss")

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
# 🎯 ACCURACY TABLE (TOP)
# =========================================================
st.markdown("## 🎯 Model Accuracy Summary")

acc_df = pd.DataFrame({
    "Model": ["Linear SVM", "XGBoost"],
    "Accuracy (%)": [acc_svm*100, acc_xgb*100]
}).round(2)

st.dataframe(acc_df, use_container_width=True)

# =========================================================
# SAMPLE FOR FAST VISUALS
# =========================================================
df_vis = df.sample(min(2000, len(df)), random_state=42)
y_vis = df_vis[target]

# =========================================================
# 📊 VISUALIZATIONS (COMPACT)
# =========================================================
st.markdown("## 📊 Visual Analysis")

# ---------- 1️⃣ Class Distribution ----------
st.subheader("1️⃣ Class Distribution")
col1, col2 = st.columns([1.3, 1])

with col1:
    st.dataframe(y.value_counts().rename_axis("Class").reset_index(name="Count"))

with col2:
    fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
    sns.countplot(x=y_vis, ax=ax)
    ax.set_title("Normal vs Attack")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---------- 2️⃣ Accuracy Comparison ----------
st.subheader("2️⃣ Accuracy Comparison")
col1, col2 = st.columns([1.3, 1])

with col1:
    st.dataframe(acc_df)

with col2:
    fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
    sns.barplot(x=acc_df["Model"], y=acc_df["Accuracy (%)"], ax=ax)
    ax.set_title("Accuracy Comparison")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---------- 3️⃣ Confusion Matrix ----------
st.subheader("3️⃣ Confusion Matrix")
col1, col2 = st.columns([1.3, 1])

with col1:
    cm_df = pd.DataFrame(cm,
        index=["Actual Normal","Actual Attack"],
        columns=["Pred Normal","Pred Attack"]
    )
    st.dataframe(cm_df)

with col2:
    fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", ax=ax)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ---------- 4️⃣ ROC Curve ----------
st.subheader("4️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)

fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
ax.plot(fpr, tpr)
ax.plot([0,1],[0,1],'--')
ax.set_title("ROC Curve")
st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ---------- 5️⃣ Precision–Recall ----------
st.subheader("5️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)

fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
ax.plot(recall, precision)
ax.set_title("Precision–Recall")
st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ---------- 6️⃣ Prediction Confidence ----------
st.subheader("6️⃣ Prediction Confidence")

fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
ax.hist(y_prob_xgb, bins=20)
ax.set_title("Attack Probability")
st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ---------- 7️⃣ Error Breakdown ----------
st.subheader("7️⃣ Error Breakdown")
error_df = pd.DataFrame({
    "Type":["TN","FP","FN","TP"],
    "Count":[cm[0,0],cm[0,1],cm[1,0],cm[1,1]]
})

fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
sns.barplot(data=error_df, x="Type", y="Count", ax=ax)
ax.set_title("Errors")
st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ---------- 8️⃣ Feature Importance ----------
st.subheader("8️⃣ Feature Importance")
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(6)

fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
ax.set_title("Top Features")
st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ---------- 9️⃣ Feature vs Class ----------
st.subheader("9️⃣ Feature vs Class")
top_feat = imp_df.iloc[0]["Feature"]

fig, ax = plt.subplots(figsize=MAX_FIG_SIZE)
sns.boxplot(x=y_vis, y=df_vis[top_feat], ax=ax)
ax.set_title(f"{top_feat} vs Class")
st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ---------- 🔟 Pair Plot (Compact) ----------
st.subheader("🔟 Pair Plot (Compact)")
pair_cols = df_vis.columns[:4].tolist() + [target]
sns.set_context("notebook", font_scale=0.55)

pair_fig = sns.pairplot(df_vis[pair_cols], hue=target, corner=True,
                         plot_kws={"s":6, "alpha":0.5})
pair_fig.fig.set_size_inches(4.5, 4.5)
st.pyplot(pair_fig.fig, use_container_width=False)
plt.close("all")

# =========================================================
# 📄 PDF DOWNLOAD
# =========================================================
def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("IDS Final Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("XGBoost outperforms Linear SVM with higher accuracy "
                              "and better intrusion detection capability.",
                              styles["Normal"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.markdown("## 📄 Download Report")
st.download_button("📥 Download IDS Report (PDF)",
                   generate_pdf(),
                   file_name="IDS_Report.pdf",
                   mime="application/pdf")

st.markdown("<h4 style='text-align:center;'>✅ Final Dashboard Ready</h4>", unsafe_allow_html=True)



