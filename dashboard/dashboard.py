import pandas as pd
import streamlit as st
from PIL import Image
import os
import sys

# setting path to file and folders
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
scripts_dir = os.path.join(parent_dir, "scripts")
data_dir = os.path.join(parent_dir, "data")

sys.path.insert(1, scripts_dir)
from dashboard_viz import VizManager

viz_man = VizManager()


user_df= pd.read_csv(data_dir+"/SmartAd_original_data.csv")

st.title("What is your customer's satisfaction level?")
st.subheader("This model will predict the satisfaction score of a user")

st.subheader("The Dataset")

st.write(
  user_df
)

viz_man.load_charts()

