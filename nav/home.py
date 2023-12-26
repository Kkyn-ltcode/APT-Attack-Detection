import streamlit as st
import pandas as pd
from st_aggrid import AgGrid
from streamlit_option_menu import option_menu

from nav import (home,data_info, reg_preprocessing,prediction, reg_training, 
                   reg_model_analysis,cls_preprocessing,cls_training,cls_model_analysis,
                   cluster_preprocessing,cluster_training,cluster_model_analysis, backward_analysis)

PAGES = {
    "Home": home,
    "DataInfo": data_info,
    "Preprocessing": (reg_preprocessing,cls_preprocessing,cluster_preprocessing),
    "Training" : (reg_training, cls_training,cluster_training),
    "Model Analysis": (reg_model_analysis, cls_model_analysis,cluster_model_analysis),
    "Prediction and Save": prediction,
    "Backward Analysis": backward_analysis,
}


def write(state):
    with st.spinner("Loading Home ..."):
        # selection = option_menu("Main Menu", ["Home", 'DataInfo', 'Preprocessing', 'Training', 'Model Analysis', 'Prediction and Save', 'Backward Analysis'], 
        #     icons=['house', 'gear'], menu_icon="cast", default_index=0)
        # selection
        selection = option_menu("", ["Home", 'DataInfo', 'Preprocessing', 'Training', 'Analysis', 'Save'],
            icons=['house', 'cloud-upload', "list-task", 'gear'], 
            menu_icon="", default_index=0, orientation="horizontal")
        selection

        task_type = st.radio("Please Select the Task Type: ", options=["Regression", "Classification", "Clustering"])
        file = st.file_uploader('Upload csv file for project', type=["csv", "xlsx"])
        if file is not None:
            file_extension = file.name.split('.')[1]
            if file_extension == "csv":
                state.df = pd.read_csv(file)
            else:
                state.df = pd.read_excel(file)
            st.header("The First 20 Rows of Data")
            #st.table(state.df.head(20))
            AgGrid(state.df.head(20))
            
            if state.df is not None:
                state.task_type=task_type
                return state.df, state.task_type
            return state.df,"No Defined"


