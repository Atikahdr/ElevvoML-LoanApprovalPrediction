import pickle
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
import os


# Load Model dan preprocessor
model = joblib.load("dt_model.pkl") #DecisionTree
preprocessor = joblib.load('preprocessor.pkl')
history_file = "loan_history.csv"

# 🔹 Session State Initializer
if "page" not in st.session_state:
    st.session_state.page = "home"
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "probability" not in st.session_state:
    st.session_state.probability = None
if os.path.exists(history_file):
    st.session_state.history = pd.read_csv(history_file).to_dict("records")
else:
    st.session_state.history = []

# Configuration Page
st.set_page_config(
    page_title="Loan Approval Decision Tree App",
    page_icon="💰",
    layout="wide"
)

# Page Navigation
st.sidebar.title("🏦 Loan Approval App")

page_map = {
    "🎓 Home Page": "home",
    "📝 Input Data": "input",
    "📋 Data Table": "table",
    "📊 Line Chart": "linechart",
    "🧾 History": "history"
}
reverse_map = {v: k for k, v in page_map.items()}

menu = st.sidebar.radio(
    "Select Page:",
    list(page_map.keys()),
    index=list(page_map.values()).index(st.session_state.page)
)
st.session_state.page = page_map[menu]
st.sidebar.markdown("---")
st.sidebar.caption("Created by **Atikah DR**")

# Prediction Function
def predict_loan(model, preprocessor, data_input):
    try:
        if data_input is None or data_input.empty:
            return None, None

        X_processed = preprocessor.transform(data_input)
        prediction = model.predict(X_processed)
        probability = model.predict_proba(X_processed)

        return prediction, probability
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None
    
def generate_dynamic_insight(input_data, prediction):
    
    income = input_data["income_annum"]
    loan = input_data["loan_amount"]
    cibil = input_data["cibil_score"]
    dependents = input_data["no_of_dependents"]
    
    total_assets = (
        input_data["residential_assets_value"] +
        input_data["commercial_assets_value"] +
        input_data["luxury_assets_value"] +
        input_data["bank_asset_value"]
    )

    reasons = []

    # Credit Score Analysis
    if cibil < 600:
        reasons.append("Low CIBIL score indicates higher credit risk.")
    elif cibil >= 750:
        reasons.append("Strong CIBIL score reflects good creditworthiness.")

    # Income vs Loan
    if income < loan:
        reasons.append("Annual income is lower than the requested loan amount.")
    else:
        reasons.append("Income level supports the requested loan amount.")

    # Asset Strength
    if total_assets < loan:
        reasons.append("Total asset value is relatively low compared to loan amount.")
    else:
        reasons.append("Strong asset backing improves financial stability.")

    # Dependents Risk
    if dependents >= 4:
        reasons.append("Higher number of dependents increases financial burden.")

    # Final Message
    if prediction == 0:
        title = "✅ Loan Approved"
        summary = "The applicant demonstrates sufficient financial stability and manageable risk profile."
    else:
        title = "❌ Loan Rejected"
        summary = "The applicant shows higher financial risk based on the provided profile."

    return title, summary, reasons

# Home Page
if st.session_state.page == "home":

    st.title("🏦 Loan Approval Prediction System")
    header_img = "loan.jpg"
    st.image(header_img, use_container_width=True)
    st.markdown("---")

    st.markdown("""
    ## Welcome to the Loan Approval Decision Tree App! 💰

    This application helps financial institutions and analysts evaluate loan applications 
    using a **Machine Learning Decision Tree model** trained on historical loan data.

    The model analyzes applicant financial profiles, credit scores, assets, and demographic information 
    to predict whether a loan application is likely to be **Approved or Rejected**.
    """)

    st.markdown("### 📌 Why Loan Approval Prediction Matters")

    st.markdown("""
    Loan approval decisions are critical for minimizing financial risk and improving capital allocation.

    By using **Machine Learning**, we can:
    - Detect patterns in applicant financial behavior
    - Reduce human bias in decision-making
    - Improve approval consistency
    - Minimize loan default risk
    - Support faster decision processes
    """)

    st.markdown("### 🤖 Model Used: Tuned Decision Tree Classifier")

    st.markdown("""
    This application uses a **Decision Tree model** that has been optimized using hyperparameter tuning 
    to improve predictive performance and generalization.
    """)

    st.markdown("### 📊 Model Performance After Tuning")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "96.82%")
    col2.metric("ROC AUC", "99.23%")
    col3.metric("Precision", "92.98%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", "99.07%")
    col5.metric("F1 Score", "95.93%")
    col6.metric("F2 Score", "97.79%")

    st.markdown("""
    🔎 **Interpretation:**

    - The high **ROC AUC (99.23%)** indicates excellent class separation ability.
    - A **Recall of 99.07%** means the model successfully identifies almost all approved loans.
    - The strong **F2 Score (97.79%)** highlights that the model prioritizes recall, 
      which is important in financial decision-making scenarios.
    """)

    st.markdown("---")

    st.markdown("""
    ### 🚀 App Features

    - 📝 Input loan applicant data
    - 🎯 Instant prediction with probability score
    - 📈 View approval trends over time
    - 📊 Analyze approval vs rejection distribution
    - 🧾 Track historical predictions
    - 🔍 Explore model behavior and insights

    Use the **sidebar navigation** to start exploring the system.
    """)

    st.success("Click the enter data button to start making your first prediction..")
    if st.button("📝 Input Loan Data"):
        st.session_state.page = "input"
        st.rerun()

# Input Data Page
elif st.session_state.page == "input":
    st.title("Input Applicant Data 📝")
    st.markdown("Please fill in the details of the loan application below:")
    
    # Applicant Profile
    st.subheader("👤 Applicant Profile")

    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, value=2, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])

    with col2:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    st.divider()

    # Financial Information
    st.subheader("💰 Financial Information")

    col3, col4 = st.columns(2)

    with col3:
        income_annum = st.number_input("Annual Income", min_value=200000, value=9600000, step=100000)
        loan_amount = st.number_input("Loan Amount", min_value=300000, value=29900000, step=100000)    
        
    with col4:
        cibil_score = st.slider("CIBIL Score", 300, 900, 778)
        loan_term = st.slider("Loan Term (years)", 1, 20, 1)

    st.divider()

    # Asset Information
    st.subheader("🏠 Asset Information")

    col5, col6 = st.columns(2)

    with col5:
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=2400000, step=100000)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=17600000, step=100000)
    
    with col6:
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=300000, value=22700000, step=100000)
        bank_asset_value = st.number_input("Bank Assets Value", min_value=0, value=8000000,step=100000)

    st.divider()

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        if st.button("🔍 Save & Predict Loan Approval", use_container_width=True):

            input_data = pd.DataFrame([{
                "no_of_dependents": no_of_dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value
            }])
        
            st.session_state.input_data = input_data
    
            # Predict with try/except
            try:
                prediction, probability = predict_loan(model, preprocessor, input_data)
                if prediction is not None:
                        st.session_state.prediction = prediction[0]
                        st.session_state.probability = probability[0] if probability is not None else None
                        st.success("✔ Data saved & predicted successfully!")
                else:
                        st.warning("⚠ Prediction failed.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

            # Save history
            confidence = None
            if st.session_state.probability is not None:
                confidence = round(np.max(st.session_state.probability)*100,2)
            
            # Mapping label numeric ke text
            status_text = "Approval" if st.session_state.prediction == 0 else "Reject"

            history_row = st.session_state.input_data.iloc[0].to_dict()
            history_row["Predicted_Status"] = status_text
            history_row["Confidence (%)"] = confidence
            history_row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.session_state.history.append(history_row)
            history_df = pd.DataFrame(st.session_state.history)
            history_df.to_csv(history_file, index=False)
            st.session_state.page = "table"
            st.rerun()

# 📋 Table Session
elif st.session_state.page == "table":

    st.title("📝Loan Approval Result")

    if st.session_state.input_data is None:
        st.warning("⚠ Please enter loan data first in INPUT PAGE.")
    
    else:
        st.markdown("Please verify the loan application details before proceeding")

        data_t = st.session_state.input_data.T.astype(str)
        data_t.columns = ["Value"]
        st.dataframe(data_t, use_container_width=True)

        st.divider()

        # Prediction 
        try:
            prediction, prob = predict_loan(
                model,
                preprocessor,
                st.session_state.input_data
            )

            if prediction is not None:

                predicted_label = prediction[0]
                probabilities = prob[0]
                confidence = np.max(probabilities) * 100

                st.subheader("🎯 Prediction Result")

                if predicted_label == 0:
                    st.success("✔️ Loan Approved")
                else:  
                    st.error("⚠️ Loan Rejected")

                 # Dynamic Insight Section
                title, summary, reasons = generate_dynamic_insight(
                    st.session_state.input_data.iloc[0].to_dict(),
                    predicted_label
        )
                st.write(summary)

                st.markdown("#### Key Factors:")
                for r in reasons:
                    st.write(f"- {r}")

            else:
                st.error("Prediction failed. Please check your inputs.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Line Chart Session
elif st.session_state.page == "linechart":
    st.title("📊 Loan Approval Trends")

    if len(st.session_state.history) == 0:
        st.info("No prediction history available to display trends.")
    
    else:
        history_df = pd.DataFrame(st.session_state.history)
        history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])
        history_df["Date"] = history_df["Timestamp"].dt.date

        trend_data = history_df.groupby(["Date", "Predicted_Status"]).size().reset_index(name="Count")

        fig = px.line(
            trend_data,
            x="Date",
            y="Count",
            color="Predicted_Status",
            markers=True,
            color_discrete_map={
                "Approval": "green", 
                "Reject": "red"
            },
            title="Loan Approval Trends Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

# History Session
elif st.session_state.page == "history":

    st.title("🧾 Loan Approval History")

    if len(st.session_state.history) == 0:
        st.info("No prediction history available.")
    
    else:
        history_df = pd.DataFrame(st.session_state.history)

        # Summary
        if "Predicted_Status" in history_df.columns:

            total = len(history_df)
            approval = (history_df["Predicted_Status"] == "Approval").sum()
            reject = (history_df["Predicted_Status"] == "Reject").sum()
          
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Predictions", total)
            col2.metric("Approval Performance", approval)
            col3.metric("Reject / Low", reject)

        st.divider()

        # Format confidence
        if "Confidence (%)" in history_df.columns:
            history_df["Confidence (%)"] = history_df["Confidence (%)"].apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
            )
        
        # Function for conditional formatting
        def highlight_reject(row):
            if row["Predicted_Status"] == "Reject":
                return ["background-color: #B22222"] * len(row)  
            elif row["Predicted_Status"] == "Approval":
                return ["background-color: #228B22"] * len(row)  
            else:
                return [""] * len(row)

        styled_df = history_df.iloc[::-1].style.apply(highlight_reject, axis=1)

        st.dataframe(styled_df, use_container_width=True)
        st.divider()

        # Download CSV
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download History",
            csv,
            "loan_approval_history.csv",
            "text/csv"
        )

    if st.button("🗑️ Delete History"):
        st.session_state.history = []
        if os.path.exists(history_file):
            os.remove(history_file)
        
        st.success("History deleted successfully!")
        st.rerun()

#  Footer
st.markdown("---")


st.caption("💡Decision Tree Regression | Machine Learning Prediction Project")
