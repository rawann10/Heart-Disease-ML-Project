import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

@st.cache_resource
def load_model():
    return joblib.load('../models/final_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('../data/heart_disease_selected.csv')

def main():
    st.title("Heart Disease Prediction App")
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio("Go to", ["Prediction", "Data Exploration", "Model Info"])
    
    if page == "Prediction":
        prediction_page()
    elif page == "Data Exploration":
        exploration_page()
    else:
        model_info_page()

def prediction_page():
    st.header("Heart Disease Risk Prediction")
    
    model = load_model()
    df = load_data()
    
    st.subheader("Enter Patient Information:")
    
    features = df.columns[:-1].tolist()
    
    input_data = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        default_val = float(df[feature].mean())
        
        input_data[feature] = col.slider(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            key=feature
        )
    
    if st.button("Predict Heart Disease Risk"):
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("High Risk of Heart Disease")
            else:
                st.success("Low Risk of Heart Disease")
        
        with col2:
            risk_percentage = probability[1] * 100
            st.metric("Risk Probability", f"{risk_percentage:.1f}%")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if risk_percentage > 50 else "darkgreen"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        st.plotly_chart(fig)

def exploration_page():
    st.header("Data Exploration")
    
    df = load_data()
    
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {df.shape}")
    st.write(df.describe())
    
    st.subheader("Feature Correlations")
    fig = px.imshow(df.corr(), text_auto=True, aspect="auto")
    st.plotly_chart(fig)
    
    st.subheader("Heart Disease Distribution")
    
    target_binary = (df.iloc[:, -1] > 0).astype(int)
    target_counts_binary = target_binary.value_counts()
    
    fig = px.pie(
        values=target_counts_binary.values, 
        names=['No Disease', 'Disease Present'],
        title="Binary Disease Distribution",
        color_discrete_sequence=['lightgreen', 'salmon']
    )
    st.plotly_chart(fig)
    
    st.subheader("Detailed Disease Severity")
    detailed_counts = df.iloc[:, -1].value_counts().sort_index()
    st.bar_chart(detailed_counts)

def model_info_page():
    st.header("Model Information")
    
    st.write("""
    This heart disease prediction model uses machine learning to assess the risk 
    of heart disease based on various health indicators.
    
    **Model Pipeline:**
    1. Data Preprocessing & Feature Selection
    2. Model Training (Random Forest/Logistic Regression/SVM)
    3. Hyperparameter Tuning
    4. Model Validation
    
    **Features Used:**
    - Selected most important features through statistical analysis
    - Standardized input features
    - Cross-validated model performance
    
    **Disclaimer:** This tool is for educational purposes only and should not 
    replace professional medical advice.
    """)

if __name__ == "__main__":
    main()