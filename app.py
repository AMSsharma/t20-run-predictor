import streamlit as st
import pickle
import pandas as pd
import numpy as np
import gdown
import os

# Google Drive direct download URL
url = "https://drive.google.com/uc?id=1Z7Etb5Ctn44aE_oxuOpdzMIzhtRwvrLX"
model_path = "pipe4.pkl"

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    with st.spinner('Downloading model...'):
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {e}")

# Load the trained pipeline
try:
    pipe = pickle.load(open(model_path, 'rb'))
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
# Teams and cities
teams = [
    'Australia','India','Bangladesh','New Zealand','South Africa','England',
    'West Indies','Pakistan','Sri Lanka'
]

cities = [
    'Colombo','Dubai','Johannesburg','Auckland','Mirpur','Dhaka','Sydney',
    'Lahore','Cape Town','London','Durban','Lauderhill','Wellington','Melbourne',
    'Pallekele','Barbados','Centurion','Christchurch','Abu Dhabi','Mount Maunganui',
    'Southampton','Hamilton','Gros Islet','Manchester','Nottingham','St Lucia',
    'Karachi','Kolkata','Cardiff','Bridgetown','Adelaide','Mumbai','Kingston',
    'Kandy','Tarouba','Birmingham','Brisbane',"St George's",'Sharjah','Chittagong',
    'Delhi','Ahmedabad','Basseterre','Providence','Harare','Chandigarh','Perth',
    'Rajkot','Dambulla','Napier','Nagpur','Pune','Bristol','Bangalore','Sylhet',
    'Hobart','Trinidad'
]

st.title('T20 Cricket Score Predictor')

# Input columns
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

col3, col4 = st.columns(2)
with col3:
    city = st.selectbox('Select the city', sorted(cities))
with col4:
    current_score = st.number_input('Current Score', min_value=0, step=1)

col5, col6 = st.columns(2)
with col5:
    overs_done = st.number_input('Overs Done', min_value=0.0, max_value=20.0, step=0.1)
with col6:
    wickets_down = st.number_input('Wickets Down', min_value=0, max_value=10, step=1)

last_five = st.number_input('Runs in last 5 overs', min_value=0, step=1)

# Predict button
if st.button('Predict Score'):
    balls_left=120-(overs_done*6)
    wickets_left=10-wickets_down
    crr=current_score/overs_done
    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'current_score':[current_score],'balls_left':[balls_left],'wickets_left':[wickets_left],'crr':[crr],'last_five':[last_five]})
    result=pipe.predict(input_df)
    st.header(f"Predicted Score: {result[0]}")