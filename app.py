import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Satellite Scheduler", layout="wide")

st.title("🛰 Self-Evolving Satellite Task Scheduler")

st.sidebar.header("Simulation Controls")

TOTAL_BATTERY = st.sidebar.slider("Total Battery", 10, 100, 50)
TOTAL_TIME = st.sidebar.slider("Total Time", 5, 50, 20)

if st.sidebar.button("Run Simulation"):

    np.random.seed()

    num_tasks = 25

    df = pd.DataFrame({
        "Priority": np.random.randint(1, 10, num_tasks),
        "Battery": np.random.randint(5, 20, num_tasks),
        "Time": np.random.randint(1, 10, num_tasks)
    })

    df = df.sort_values(by="Priority", ascending=False)

    remaining_battery = TOTAL_BATTERY
    remaining_time = TOTAL_TIME
    scheduled = []

    for _, row in df.iterrows():
        if row["Battery"] <= remaining_battery and row["Time"] <= remaining_time:
            scheduled.append(1)
            remaining_battery -= row["Battery"]
            remaining_time -= row["Time"]
        else:
            scheduled.append(0)

    df["Scheduled"] = scheduled

    X = df[["Priority", "Battery", "Time"]]
    y = df["Scheduled"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Scheduling Outcome")
        accepted = df["Scheduled"].sum()
        rejected = len(df) - accepted

        fig1, ax1 = plt.subplots()
        ax1.bar(["Accepted", "Rejected"], [accepted, rejected])
        st.pyplot(fig1)

    with col2:
        st.subheader("🔋 Battery Usage")
        used = TOTAL_BATTERY - remaining_battery

        fig2, ax2 = plt.subplots()
        ax2.bar(["Used", "Remaining"], [used, remaining_battery])
        st.pyplot(fig2)

    st.subheader("📈 Priority vs Scheduling")
    fig3, ax3 = plt.subplots()
    ax3.scatter(df["Priority"], df["Scheduled"])
    ax3.set_xlabel("Priority")
    ax3.set_ylabel("Scheduled")
    st.pyplot(fig3)

    st.success(f"Model Accuracy: {accuracy:.2f}")
