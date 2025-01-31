import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stRadio > div {
        padding: 10px;
        background-color: black;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_resource
def load_models():
    data = pd.read_csv('Churn_Modelling.csv')
    model = tf.keras.models.load_model('ann_model.h5')
    regression_model = tf.keras.models.load_model('regression_model.h5')    
    with open("geo_encoder.pkl",'rb') as f:
        ohe_geo_encoder=pickle.load(f)
    with open("gender_encoder.pkl",'rb') as f:
        lb_gender_encoder=pickle.load(f)
    with open("scaler.pkl",'rb') as f:
        scaler=pickle.load(f)
    with open("scalling_regression.pkl",'rb') as f:
        regression_scaler=pickle.load(f)

    return data, model, regression_model, ohe_geo_encoder, lb_gender_encoder, scaler, regression_scaler

data, model, regression_model, ohe_geo_encoder, lb_gender_encoder, scaler, regression_scaler = load_models()

# Main title with emoji
st.title("🎯 Customer Analysis Tool")

# Model selection
analysis_type = st.radio(
    "Select Analysis Type",
    ["Churn Prediction (Classification)", "Salary Estimation (Regression)"],
    horizontal=True
)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    geography = st.selectbox("📍 Geography", ohe_geo_encoder.categories_[0])
    gender = st.selectbox('👤 Gender', lb_gender_encoder.classes_)
    age = st.slider("🎂 Age", 
                    int(data['Age'].min()), 
                    int(data['Age'].max()),
                    value=30)
    balance = st.number_input('💰 Balance', 
                             min_value=0.0,
                             value=0.0,
                             format="%.2f")

with col2:
    st.subheader("Account Details")
    credit_score = st.number_input('📈 Credit Score',
                                  min_value=300,
                                  max_value=850,
                                  value=600)
    tenure = st.slider("⏳ Tenure (Years)",
                      int(data['Tenure'].min()),
                      int(data['Tenure'].max()),
                      value=5)
    num_of_products = st.slider('🛍️ Number of Products',
                               int(data['NumOfProducts'].min()),
                               int(data['NumOfProducts'].max()),
                               value=1)
    
# Additional features in a single column for better organization
col3, col4, col5 = st.columns(3)

with col3:
    has_credit_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col4:
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if analysis_type=="Churn Prediction (Classification)":
    with col5:
        estimated_salary = st.number_input("💵 Estimated Salary",
                                        min_value=0.0,
                                        value=0.0,
                                      format="%.2f")
else:
     with col5:
        exited = st.selectbox('Exited', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if analysis_type=="Churn Prediction (Classification)":
    # Create input data dictionary
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }
else:
     input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_credit_card,
        'IsActiveMember': is_active_member,
        'Exited': exited
    }

input_df = pd.DataFrame(input_data, index=[0])

# Data preprocessing
def preprocess_data(input_df):
    # Geography encoding
    geo_encoded = ohe_geo_encoder.transform([[input_df["Geography"].iloc[0]]])
    geo_encoded_df = pd.DataFrame(geo_encoded, 
                                 columns=ohe_geo_encoder.get_feature_names_out())
    
    # Merge and drop original geography column
    processed_df = pd.concat([input_df.reset_index(drop=True), 
                            geo_encoded_df], axis=1)
    processed_df.drop(columns=['Geography'], inplace=True)
    
    # Gender encoding
    processed_df['Gender'] = lb_gender_encoder.transform(processed_df['Gender'])
    
    return processed_df

# Process data
processed_data = preprocess_data(input_df)

# Prediction section
st.markdown("---")
st.subheader("📊 Analysis Results")

if analysis_type == "Churn Prediction (Classification)":
    # Scale data for classification
    final_data_cl = scaler.transform(processed_data)
    
    # Make prediction
    prediction = model.predict(final_data_cl)
    prediction_prob = prediction[0][0]
    
    # Display results with progress bar
    col6, col7 = st.columns(2)
    
    with col6:
        st.metric("Churn Probability", f"{prediction_prob*100:.1f}%")
        
    
    # Display prediction message
    if prediction_prob > 0.5:
        st.error("⚠️ High Risk: The customer is likely to churn!")
        st.markdown("""
            **Recommended Actions:**
            - Schedule a customer review meeting
            - Offer personalized retention benefits
            - Analyze pain points
        """)
    else:
        st.success("✅ Low Risk: The customer is likely to stay!")
        st.markdown("""
            **Recommended Actions:**
            - Continue monitoring engagement
            - Consider upselling opportunities
            - Maintain regular communication
        """)

else:  # Salary Estimation
    # Scale data for regression
    final_data_reg = regression_scaler.transform(processed_data)
    
    # Make prediction
    salary_prediction = regression_model.predict(final_data_reg)
    
    # Display results
    st.metric("Estimated Salary", f"${salary_prediction[0][0]:,.2f}")
    
    # Add some context
    st.info("ℹ️ This estimation is based on the customer's profile and market data")


