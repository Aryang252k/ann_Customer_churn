import streamlit as st
import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pandas as pd
import joblib


data=pd.read_csv('Churn_Modelling.csv')

#Load model, encode and scaler 
# model=tf.keras.models.load_model('ann_model.h5')

ohe_geo_encoder=joblib.load('geo_encoder.pkl')

lb_gender_encoder=joblib.load('gender_encoder.pkl')

scaler=joblib.load('scaler.pkl')


#streamlit app
st.write('Customer Churn Prediction')



#user input
geography=st.selectbox("Geography", ohe_geo_encoder.categories_)
gender=st.selectbox('Gender',lb_gender_encoder.classes_)
age=st.slider("Age",data['Age'].min(),data['Age'].max())
balance= st.number_input('Balance')
credit_score= st.number_input('Credit Score')
estimated_salary= st.number_input("Estimated Salary")
tenure=st.slider("Tenure",data['Tenure'].min(),data['Tenure'].max())
num_of_products=st.slider('Number of Products',data['NumOfProducts'].min(),data['NumOfProducts'].max())
has_or_card = st.selectbox('Has credit card',[0,1])
is_active_member=st.selectbox('Is active Number',[0,1])




