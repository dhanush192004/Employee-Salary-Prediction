# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

# ─── Config ────────────────────────────────────────────────────────
st.set_page_config(page_title="Employee Salary Prediction", layout="wide")
sns.set_style("whitegrid")

# ─── Load Model ───────────────────────────────────────────────────
try:
    model, scaler, columns = joblib.load("best_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ─── Title ────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;'>💼 Employee Salary Prediction</h1>", unsafe_allow_html=True)
st.write("Predict whether income is <=50K or >50K based on demographic/work features.")

# ─── Sidebar Input Form ───────────────────────────────────────────
with st.sidebar.form("input_form"):
    st.header("🔧 Enter Your Details")
    age = st.slider("Age", 18, 90, 30)
    education_num = st.slider("Education Number", 1, 16, 10)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    net_capital = st.number_input("Net Capital (gain − loss)", value=0)

    workclass = st.selectbox("Workclass", ["Private","Self-emp-not-inc","Self-emp-inc",
                                           "Federal-gov","Local-gov","State-gov","Without-pay","Never-worked"])
    marital_status = st.selectbox("Marital Status", ["Never-married","Married-civ-spouse","Divorced",
                                                     "Separated","Widowed","Married-spouse-absent"])
    occupation = st.selectbox("Occupation", ["Tech-support","Craft-repair","Other-service",
                                             "Sales","Exec-managerial","Prof-specialty","Handlers-cleaners",
                                             "Machine-op-inspct","Adm-clerical","Farming-fishing","Transport-moving",
                                             "Priv-house-serv","Protective-serv","Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Wife","Own-child","Husband","Not-in-family",
                                                 "Other-relative","Unmarried"])
    race = st.selectbox("Race", ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"])
    sex = st.selectbox("Sex", ["Male","Female"])
    country_group = st.selectbox("Country Group", ["United-States","Other"])

    submit = st.form_submit_button("🚀 Predict")

# ─── Prediction Logic ─────────────────────────────────────────────
if submit:
    # 1) Start with a zero-filled DataFrame matching training columns
    input_df = pd.DataFrame(columns=columns)
    input_df.loc[0] = 0

    # 2) Set numeric features
    numeric_fields = {
        "age": age,
        "education-num": education_num,
        "hours-per-week": hours_per_week,
        "net_capital": net_capital
    }
    for fld, val in numeric_fields.items():
        input_df.at[0, fld] = val

    # 3) One-hot flags setter
    def flag(prefix, value):
        col = f"{prefix}_{value}"
        if col in columns:
            input_df.at[0, col] = 1

    flag("workclass", workclass)
    flag("marital-status", marital_status)
    flag("occupation", occupation)
    flag("relationship", relationship)
    flag("race", race)
    flag("sex", sex)
    flag("country_group", "US" if country_group=="United-States" else "Other")

    # 4) Scale only those four numeric columns
    num_cols = ["age","education-num","hours-per-week","net_capital"]
    try:
        input_df[num_cols] = scaler.transform(input_df[num_cols])
    except Exception:
        pass  # skip if mismatch

    # 5) Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    label = ">50K" if pred==1 else "<=50K"
    conf = proba[pred]

    # 6) Result Card
    if pred == 1:
        bg, bd, txt = "#d4edda","#28a745","#155724"
    else:
        bg, bd, txt = "#f8d7da","#dc3545","#721c24"
    st.markdown(f"""
    <div style='background:{bg};border:2px solid {bd};padding:15px;border-radius:8px;margin-top:15px;'>
      <h2 style='color:{txt};'>🎯 Prediction: <strong>{label}</strong></h2>
      <h4 style='color:{txt};'>Confidence: {conf:.2%}</h4>
    </div>""", unsafe_allow_html=True)

    # 7) Confidence Bar Chart
    st.subheader("Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar([">50K","<=50K"], [proba[1],proba[0]], color=["#85C1E9","#F1948A"])
    ax.set_ylim(0,1)
    st.pyplot(fig)

    # 8) Review Table
    st.subheader("🔎 Review Your Selections")
    review = pd.DataFrame({
      "Feature": ["Age","Education Num","Hours/Week","Net Capital",
                  "Workclass","Marital Status","Occupation","Relationship","Race","Sex","Country Group"],
      "Value": [age,education_num,hours_per_week,net_capital,
                workclass,marital_status,occupation,relationship,race,sex,country_group]
    })
    st.table(review)

    # 9) Full Input Preview
    st.subheader("📋 Full Model Input Vector")
    st.dataframe(input_df)

# ─── Batch CSV Prediction ─────────────────────────────────────────
st.markdown("---")
st.header("📂 Batch Prediction")
upload = st.file_uploader("Upload CSV", type="csv")
if upload:
    batch = pd.read_csv(upload)
    batch['net_capital'] = batch['capital-gain'] - batch['capital-loss']
    batch['country_group'] = batch['native-country'].apply(lambda x: 'US' if x=='United-States' else 'Other')
    batch.drop(["capital-gain","capital-loss","native-country","fnlwgt","education"],axis=1,errors='ignore',inplace=True)
    batch = pd.get_dummies(batch,drop_first=True).reindex(columns=columns,fill_value=0)
    try:
        batch[num_cols] = scaler.transform(batch[num_cols])
    except Exception:
        pass
    preds = model.predict(batch)
    probs = model.predict_proba(batch).max(axis=1)
    batch["Predicted Income"] = np.where(preds==1,">50K","<=50K")
    batch["Confidence"] = probs
    st.write(batch)
