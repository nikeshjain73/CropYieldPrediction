import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from streamlit.components.v1 import html


page_bg_img = """ 
<style>
[data-testid="stAppViewContainer"]{
    background-image: url("static/img/bg.jpg")
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Function to preprocess input
def preprocess_input(year, rain_fall, pesticides, temp, area, item):
    features = np.array([[year, rain_fall, pesticides, temp, area, item]], dtype=object)
    transformed_features = preprocessor.transform(features)
    return transformed_features


# Function to display home page with background image
def home():
    st.title('Crop Yield Prediction')
    st.header('All Field Required')
    year = st.number_input('Year', min_value=1990, max_value=3000, step=1)
    rain_fall = st.number_input('Average Rain Fall (mm/year)')
    pesticides = st.number_input('Pesticides (tonnes)')
    temp = st.number_input('Average Temperature (°C)')
    area = st.text_input('Area (Country)')
    item = st.text_input('Crop')  # Add your items here
    

    if st.button('Predict'):
        transformed_input = preprocess_input(year, rain_fall, pesticides, temp, area, item)
        prediction = dtr.predict(transformed_input)
        st.write('Predicted Yield: ' + str(prediction) + " (Hectogram/Hectare (Hg/Ha))")

   

def about():
    st.title('About')
    st.write("""
    <div style="text-align: justify;">
    The use of machine learning techniques to forecast crop yield is examined in this work. We make use of a dataset that contains a variety of variables that could affect yield, such as crop type, weather, and geographic data. Several machine learning models are trained and assessed following data cleaning and pre-processing, which includes resolving missing values, converting data formats, and carrying out feature engineering. Decision tree regression, Lasso regression, Ridge regression, and linear regression are some of these models.
    <br>
    <br>
    The decision tree regression model outperforms the other models in terms of mean absolute error (MAE) and R-squared score on the hold-out test set. Lastly, a prediction function that enables users to enter precise values for the previously mentioned factors and get a projected crop yield is created using the selected model. For later use, the pre-processor and model are also serialized.
    <br>
    <br>

    This strategy provides a data-driven way for predicting crop yield, which may enable farmers to decide on resource allocation and harvest scheduling with knowledge.
    </div>
    <ul style="list-style-type: none;">
    <h3>Guided By:</h3>
    <h4>
    Abdul Aziz md
    <br>
    Master trainer
    <br>
    Edunet Foundation
    </h4>
    <h4> Presented By:</h4>
    <li><h5>Nikesh Jain</h5></li>
    <li><h5>Ayushi Patel</h5></li>
    <li><h5>Archi Borad</h5></li>
    <li><h5>Prachi Pancholi</h5></li>
    </ul>
    """, unsafe_allow_html=True)
       
# Sidebar
with st.sidebar:
    selected = option_menu("Crop Yield Prediction", ["Home", "About"], 
        icons=['house', 'info'],menu_icon="gear", default_index=0)

# Display appropriate page based on selection
if selected == "Home":
    home()
elif selected == "About":
    about()


 # Footer
# st.markdown(
#     """
#     <footer style="bottom:0; width:100%; padding:10px; text-align:right;">
#         <p>Made with  by: Nikesh Jain</p>
#     </footer>
#     """,
#     unsafe_allow_html=True
# )