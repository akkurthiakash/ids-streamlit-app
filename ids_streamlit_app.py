# =========================================================
# IDS STREAMLIT DASHBOARD — FINAL COMPLETE PROJECT
# Tables + Graphs + CSV + Image Download (Each Section)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# =========================================================
# 🔐 LOGIN
# =========================================================
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

USERS = {"akash": hash_password("ids@2025")}

if "auth" not in st.session_state:
    st.session_state.auth = False

def login():
    st.markdown("<h2 style='text-align:center;'> IDS Dashboard Login</h2>", unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u in USERS and hash_password(p) == USERS[u]:
            st.session_state.auth = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid username or password")

if not st.session_state.auth:
    login()
    st.stop()

# =========================================================
# PAGE STYLE
# =========================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white;}
.center-text {text-align:center;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE + SIMPLE REPORT
# =========================================================
st.markdown("<h1 class='center-text'> Intrusion Detection System </h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; font-size:15px; line-height:1.8; max-width:900px; margin:auto;">
<b>Purpose:</b> Detect network attacks using machine learning.<br>
<b>Outcome:</b> XGBoost outperforms SVM with higher accuracy and fewer missed attacks.<br>
<b>Usage:</b> Helps administrators monitor traffic and detect intrusions early.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# DATA UPLOAD
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
# TRAIN MODELS
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

svm = SVC(kernel="linear", probability=True, class_weight="balanced")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.9, colsample_bytree=0.9, eval_metric="logloss"
)

svm.fit(X_train_svm, y_train)
xgb.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test_svm)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

acc_svm = accuracy_score(y_test, y_pred_svm)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# =========================================================
# HELPER: SAVE FIG TO BUFFER
# =========================================================
def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf

# =========================================================
# 🔟 VISUALIZATIONS (TABLE + GRAPH + CSV + IMAGE)
# =========================================================

# 1️⃣ Class Distribution
st.subheader("1️⃣ Class Distribution")

class_df = y.value_counts().rename_axis("Class").reset_index(name="Count")
st.dataframe(class_df)

fig = plt.figure(figsize=(4,3))
sns.countplot(x=y)
plt.title("Normal vs Attack Distribution")
plt.xlabel("Class (0 = Normal, 1 = Attack)")
plt.ylabel("Count")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   class_df.to_csv(index=False),
                   "class_distribution.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "class_distribution.png")
plt.close()

# 2️⃣ Model Performance
st.subheader("2️⃣ Model Performance Summary")

metrics_df = pd.DataFrame({
    "Model":["SVM","XGBoost"],
    "Accuracy":[acc_svm, acc_xgb],
    "Precision":[precision_score(y_test,y_pred_svm), precision_score(y_test,y_pred_xgb)],
    "Recall":[recall_score(y_test,y_pred_svm), recall_score(y_test,y_pred_xgb)],
    "F1-Score":[f1_score(y_test,y_pred_svm), f1_score(y_test,y_pred_xgb)]
}).round(4)

st.dataframe(metrics_df)

fig = plt.figure(figsize=(4,3))
sns.barplot(x=metrics_df["Model"], y=metrics_df["Accuracy"])
plt.title("Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   metrics_df.to_csv(index=False),
                   "model_performance.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "accuracy_comparison.png")
plt.close()

# 3️⃣ Confusion Matrix
st.subheader("3️⃣ Confusion Matrix")

cm = confusion_matrix(y_test, y_pred_xgb)
cm_df = pd.DataFrame(cm,
                     index=["Actual Normal","Actual Attack"],
                     columns=["Predicted Normal","Predicted Attack"])
st.dataframe(cm_df)

fig = plt.figure(figsize=(4,3))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix (XGBoost)")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   cm_df.to_csv(),
                   "confusion_matrix.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "confusion_matrix.png")
plt.close()

# 4️⃣ ROC Curve
st.subheader("4️⃣ ROC Curve")

fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_df = pd.DataFrame({"FPR":fpr,"TPR":tpr})
st.dataframe(roc_df.head(10))

fig = plt.figure(figsize=(4,3))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   roc_df.to_csv(index=False),
                   "roc_curve.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "roc_curve.png")
plt.close()

# 5️⃣ Precision–Recall Curve
st.subheader("5️⃣ Precision–Recall Curve")

precision, recall, _ = precision_recall_curve(y_test, y_prob_xgb)
pr_df = pd.DataFrame({"Recall":recall,"Precision":precision})
st.dataframe(pr_df.head(10))

fig = plt.figure(figsize=(4,3))
plt.plot(recall, precision)
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   pr_df.to_csv(index=False),
                   "precision_recall.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "precision_recall.png")
plt.close()

# 6️⃣ Prediction Confidence
st.subheader("6️⃣ Prediction Confidence")

conf_df = pd.DataFrame({"Attack Probability":y_prob_xgb})
st.dataframe(conf_df.head(10))

fig = plt.figure(figsize=(4,3))
plt.hist(y_prob_xgb, bins=25)
plt.title("Attack Probability Distribution")
plt.xlabel("Attack Probability")
plt.ylabel("Frequency")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   conf_df.to_csv(index=False),
                   "prediction_confidence.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "prediction_confidence.png")
plt.close()

# 7️⃣ Error Breakdown
st.subheader("7️⃣ Error Breakdown")

error_df = pd.DataFrame({
    "Type":["TN","FP","FN","TP"],
    "Count":[cm[0,0],cm[0,1],cm[1,0],cm[1,1]]
})
st.dataframe(error_df)

fig = plt.figure(figsize=(4,3))
sns.barplot(data=error_df, x="Type", y="Count")
plt.title("Prediction Errors")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   error_df.to_csv(index=False),
                   "error_breakdown.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "error_breakdown.png")
plt.close()

# 8️⃣ Feature Importance
st.subheader("8️⃣ Feature Importance")

imp_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":xgb.feature_importances_
}).sort_values("Importance",ascending=False)
st.dataframe(imp_df.head(8))

fig = plt.figure(figsize=(5,3))
sns.barplot(data=imp_df.head(8), x="Importance", y="Feature")
plt.title("Top Important Features")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   imp_df.to_csv(index=False),
                   "feature_importance.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "feature_importance.png")
plt.close()

# 9️⃣ Feature vs Class
st.subheader("9️⃣ Feature vs Class")

top_feat = imp_df.iloc[0]["Feature"]
feat_df = df.groupby(target)[top_feat].describe().reset_index()
st.dataframe(feat_df)

fig = plt.figure(figsize=(4,3))
sns.boxplot(x=y, y=df[top_feat])
plt.title(f"{top_feat} by Class")
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   feat_df.to_csv(index=False),
                   "feature_vs_class.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "feature_vs_class.png")
plt.close()

# 🔟 Feature Interaction
st.subheader("🔟 Feature Interaction")

f1, f2 = imp_df.iloc[0]["Feature"], imp_df.iloc[1]["Feature"]
inter_df = df[[f1,f2,target]].sample(n=min(4000,len(df)),random_state=42)
st.dataframe(inter_df.head(10))

fig = plt.figure(figsize=(4,3))
sns.scatterplot(x=inter_df[f1], y=inter_df[f2], hue=inter_df[target], alpha=0.6)
plt.title("Feature Interaction")
plt.xlabel(f1)
plt.ylabel(f2)
st.pyplot(fig)

st.download_button("⬇️ Download CSV",
                   inter_df.to_csv(index=False),
                   "feature_interaction.csv")

st.download_button("⬇️ Download Image",
                   fig_to_png(fig),
                   "feature_interaction.png")
plt.close()

# =========================================================
# END
# =========================================================
st.markdown("<h4 class='center-text'>✅ Full IDS Project Dashboard Ready</h4>", unsafe_allow_html=True)
