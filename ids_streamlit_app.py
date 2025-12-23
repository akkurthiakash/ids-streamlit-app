# =========================================================
# IDS STREAMLIT DASHBOARD — FAST & FINAL VERSION
# Optimized for Speed | Tables + Graphs + Downloads
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
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# =========================================================
# 🔐 LOGIN (FAST)
# =========================================================
def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

USERS = {"akash": hash_password("ids@2025")}

if "auth" not in st.session_state:
    st.session_state.auth = False

def login():
    st.markdown("<h2 style='text-align:center;'>🔐 IDS Dashboard Login</h2>", unsafe_allow_html=True)
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
# STYLE
# =========================================================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); color:white;}
.center-text {text-align:center;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE + SIMPLE REPORT
# =========================================================
st.markdown("<h1 class='center-text'>🛡️ Intrusion Detection System</h1>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; font-size:15px; max-width:900px; margin:auto;">
<b>Purpose:</b> Detect cyber attacks using ML<br>
<b>Outcome:</b> XGBoost outperforms SVM<br>
<b>Usage:</b> Network traffic monitoring
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 📂 DATA LOADING (CACHED)
# =========================================================
@st.cache_data
def load_data(file):
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    return df.dropna().drop_duplicates()

uploaded = st.file_uploader("Upload IDS Dataset (CSV / XLSX)", ["csv","xlsx"])
if uploaded is None:
    st.stop()

df = load_data(uploaded)

target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target].astype(int)

# =========================================================
# 🤖 MODEL TRAINING (CACHED)
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
        n_estimators=150,        # ↓ reduced for speed
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

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:,1]

# =========================================================
# FAST SAMPLE FOR VISUALS (IMPORTANT)
# =========================================================
VIS_SAMPLE = min(3000, len(df))
df_vis = df.sample(VIS_SAMPLE, random_state=42)
y_vis = df_vis[target]

# =========================================================
# HELPER: SAVE FIG TO IMAGE BUFFER
# =========================================================
def fig_to_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    return buf

# =========================================================
# 🔟 FAST VISUALIZATIONS (OPTIMIZED)
# =========================================================

# 1️⃣ Class Distribution
st.subheader("1️⃣ Class Distribution")
class_df = y.value_counts().reset_index()
class_df.columns = ["Class","Count"]
st.dataframe(class_df)

fig = plt.figure(figsize=(4,3))
sns.countplot(x=y_vis)
plt.title("Normal vs Attack")
st.pyplot(fig)
st.download_button("CSV", class_df.to_csv(index=False), "class_distribution.csv")
st.download_button("IMG", fig_to_png(fig), "class_distribution.png")
plt.close()

# 2️⃣ Confusion Matrix
st.subheader("2️⃣ Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_xgb)
cm_df = pd.DataFrame(cm, index=["Actual Normal","Actual Attack"],
                     columns=["Pred Normal","Pred Attack"])
st.dataframe(cm_df)

fig = plt.figure(figsize=(4,3))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix")
st.pyplot(fig)
st.download_button("CSV", cm_df.to_csv(), "confusion_matrix.csv")
st.download_button("IMG", fig_to_png(fig), "confusion_matrix.png")
plt.close()

# 3️⃣ ROC Curve
st.subheader("3️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob_xgb)
roc_df = pd.DataFrame({"FPR":fpr,"TPR":tpr})
st.dataframe(roc_df.head(10))

fig = plt.figure(figsize=(4,3))
plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
st.pyplot(fig)
st.download_button("CSV", roc_df.to_csv(index=False), "roc_curve.csv")
st.download_button("IMG", fig_to_png(fig), "roc_curve.png")
plt.close()

# 4️⃣ Feature Importance
st.subheader("4️⃣ Feature Importance")
imp_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":xgb.feature_importances_
}).sort_values("Importance",ascending=False).head(8)

st.dataframe(imp_df)

fig = plt.figure(figsize=(5,3))
sns.barplot(data=imp_df, x="Importance", y="Feature")
plt.title("Top Features")
st.pyplot(fig)
st.download_button("CSV", imp_df.to_csv(index=False), "feature_importance.csv")
st.download_button("IMG", fig_to_png(fig), "feature_importance.png")
plt.close()

# 5️⃣ Feature Interaction (FAST)
st.subheader("5️⃣ Feature Interaction")
f1, f2 = imp_df.iloc[0]["Feature"], imp_df.iloc[1]["Feature"]
inter_df = df_vis[[f1,f2,target]]

st.dataframe(inter_df.head(10))

fig = plt.figure(figsize=(4,3))
sns.scatterplot(x=inter_df[f1], y=inter_df[f2],
                hue=inter_df[target], alpha=0.5)
plt.title("Feature Interaction")
st.pyplot(fig)
st.download_button("CSV", inter_df.to_csv(index=False), "feature_interaction.csv")
st.download_button("IMG", fig_to_png(fig), "feature_interaction.png")
plt.close()

st.markdown("<h4 class='center-text'>⚡ Fast IDS Dashboard Loaded</h4>", unsafe_allow_html=True)
