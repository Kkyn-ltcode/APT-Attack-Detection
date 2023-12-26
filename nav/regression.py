import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from streamlit_option_menu import option_menu

from nav import (data_info, reg_preprocessing, reg_prediction, reg_training, 
                   reg_model_analysis, backward_analysis)

def write(state):
    state = st.session_state
    st.header("Regression")

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
        state = reg_preprocessing.write(state)
    if selection1 == "Training":
        state = reg_training.write(state)
    if selection1 == "Analysis":
        state = reg_model_analysis.write(state)
    if selection1 == "Save":
        state = reg_prediction.write(state)
    if selection1 == "Backward Analysis":
        backward_analysis.write(state)


