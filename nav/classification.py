import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from streamlit_option_menu import option_menu

from nav import (data_info, cls_preprocessing, cls_prediction, cls_training, 
                   cls_model_analysis)

def write(state):
    state = st.session_state
    st.header("Classification")

    selection1 = option_menu("", ['Data', 'Prepare', 'Training', 'Analysis', 'Save'],
        icons=['', '', "", ''], 
        menu_icon="", default_index=0, orientation="horizontal")
    selection1
    if selection1 == "Data":
        state.df = None
        state.is_set_up = False
        state.trained_model = None
        state = data_info.write(state)  
        state.log_history = {}
        state.is_remove = False
        state.ignore_columns = []
    if selection1 == "Prepare":
        state = cls_preprocessing.write(state)
    if selection1 == "Training":
        state = cls_training.write(state)
    if selection1 == "Analysis":
        state = cls_model_analysis.write(state)
    if selection1 == "Save":
        state = cls_prediction.write(state)
