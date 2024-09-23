import streamlit as st
import numpy as np
import pandas as pd
import pickle

# import the model
brand_le=pickle.load(open('models/brand_le.pkl','rb'))
model_le=pickle.load(open('models/model_le.pkl','rb'))
cpu_le=pickle.load(open('models/cpu_le.pkl','rb'))

standard_scaler=pickle.load(open('models/standard_scaler.pkl','rb'))
XGB_Regressor=pickle.load(open('models/XGB_Regressor.pkl','rb'))

new_df=pickle.load(open('models/new_df.pkl','rb'))

st.title("Laptop Price Predictor")

# status
status=st.selectbox('Status',['New','Refurbished/Reconditioned'])

# brand
brand=st.selectbox('Brand',new_df['Brand'].unique())

# model
model=st.selectbox('Model',new_df['Model'].unique())

# cpu
cpu=st.selectbox('CPU',new_df['CPU'].unique())

# RAM
RAM=st.selectbox('RAM Size (in GB)',[2,4,8,12,16,32,64])

# SSD Size
SSD_size=st.selectbox('SSD Size (in GB)',[32,64,128,256,512,1024,2048])

# Screen Size
Screen_size=st.selectbox('Screen Size (in inches)',[11.6,12.3,12.4,12.5,13,13.3,13.4,13.5,13.6,13.9,14,14.2,14.4,14.5,15,15.3,15.4,15.6,16,16.1,16.2,17,17.3,18])

# Touch
touch=st.selectbox('Touch',['No','Yes'])

if st.button('Predict Price'):
    # query
    if status=="New":
        status=1
    else:
        status=0
    
    if touch=="Yes":
        touch=1
    else:
        touch=0

    query = pd.DataFrame({
    'Status': [status],
    'Brand': [brand],
    'Model': [model],
    'CPU': [cpu],
    'RAM': [RAM],
    'SSD Size': [SSD_size],
    'Screen': [Screen_size],
    'Touch': [touch]
    })
    
    # applying label encoders
    query['Brand']=brand_le.transform(query['Brand'])
    query['Model']=model_le.transform(query['Model'])
    query['CPU']=cpu_le.transform(query['CPU'])
    
    # applying standard scaling
    query_scaled=standard_scaler.transform(query)
    
    
    # predicting the price
    price=XGB_Regressor.predict(query_scaled)
    price=f"{price[0]:.0f} USD"
    
    st.title(price)
    
    
    