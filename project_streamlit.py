
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Graphing packages
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# stuff for machine
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pydeck as pdk

# Container Setup

header = st.container()
map = st.container()
model = st.container()

## Setup
st.markdown(
    """
<style>
.main {
    background-color: #3D405B;
}
</style>
    """,
    unsafe_allow_html=True,
)

# --Body--
# header

with header:
    st.title("Where Should I Place My Business")


with map:

    data = pd.read_csv("https://raw.githubusercontent.com/BryceBlignaut/FF_Business_Development_Application/main/main_export.csv")
    
    
    df = data[["tract","region","county_name","city","total_visits", "avg_income","total_businesses","lat","lon",'n_lsRestaurants',"predicted_difference", "growth_area"]]
    
    value_selected = False

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        state = st.multiselect("Select State",df["region"].unique())
        if(state != []):
            df = df[(df.region == state[0])]
            value_selected = True
    with col2:
        county = st.multiselect("Select County",df["county_name"].unique())
        if(county != []):
            df = df[(df.county_name == county[0])]
            value_selected = True
    with col3:
        city = st.multiselect("Select City",df["city"].unique())
        if(city != []):
            df = df[(df.city == city[0])]
            value_selected = True
    with col4:
        state = st.multiselect("Growth Area",df["growth_area"].unique())
        if(state != []):
            df = df[(df.growth_area == state[0])]
            value_selected = True

    # st.map(df) #This is the basic map. Can't change colors. But it looks nice by itself. And it filters way nice
    st.write("The more green an area is, the more growth potential there")
    #IDK about the colors in this map. But it works.
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=pdk.ViewState(
            latitude=42.860851,
            longitude=-110.117649,
            zoom=4.25,
            min_zoom=4,
            max_zoom=15
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df[['lat','lon','predicted_difference','tract']].dropna(),
                get_position=['lon', 'lat'],
                auto_highlight=True,
                pickable=True,
                get_radius=1000,
                get_fill_color=['predicted_difference < 0 ? predicted_difference * -255 : 0', 'predicted_difference > 0 ? predicted_difference * 255 : 0', 0, 160],
                coverage=1,
                radius_min_pixels=3,
                radius_max_pixels=5)
        ], # Added a tooltip here
        tooltip={
        'html': '<b>Tract:</b> {tract}',
        'style': {
            'color': 'white'
        }
    }
    ))

    if value_selected:    
        st.write("What makes these places a great place to build?")

        st.write(df[["tract","growth_area","region","county_name","city","total_visits", "avg_income","total_businesses"]])




    # new_business = df['growth_area'].drop_duplicates()
    #new_business_choice = st.sidebar.selectbox('Growth Area:', new_business)
    # region = df['region'].loc[df['growth_area'] == new_business_choice].drop_duplicates()
    # region_choice = st.sidebar.selectbox('Region', region)
    # city = df['city'].loc[df['growth_area'] == new_business_choice | df['region'] == new_business_choice].drop_duplicates()
    # city_choice = st.sidebar.selectbox('City', city)


    # Filter dataset
    #st.write(df.loc[(df['region']==region_choice) & (df['growth_area']==new_business_choice) & (df['city']==city_choice)])
