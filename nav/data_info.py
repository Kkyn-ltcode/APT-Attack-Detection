import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
from st_aggrid import AgGrid

#@st.cache(suppress_st_warning=True)
def write(state):
    st.subheader("Data Explotary Analysis")
    
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
        
        
        with st.spinner("Loading Data Info ..."):
            pr = ProfileReport( state.df, explorative=True,minimal=True)
            st_profile_report(pr)

    
    return state
    	