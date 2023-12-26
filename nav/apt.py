# Import necessary libraries and modules
import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import numpy as np
import pandas as pd
import json
import random
import joblib
import pickle
import warnings
from nav import Model
from io import StringIO
from datetime import datetime
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
warnings.filterwarnings('ignore')

# Function to extract features from file paths
# def extract_feature(target):
#     feature = Feature_Extraction.generate_path(target)
#     return feature

# # Function to make predictions on file paths using a trained model
def make_path_prediction(df, model_path):
    label = ["BENIGN", "APT"]
    model = joblib.load(model_path)
    predictions = model.predict(df.iloc[:, 8:-1])
    df['Prediction'] = predictions
    df['Prediction'] = df['Prediction'].apply(lambda x: label[x])
    display_df = df.iloc[:, [0, 1, -2, -1]]
    with st.expander('Prediction Result'):
        st.write(display_df)

# # Function to display the results in a table using AgGrid
# def show_tabel(state, df):
#     gb = GridOptionsBuilder.from_dataframe(df)
#     gb.configure_pagination(paginationAutoPageSize=True)
#     gb.configure_columns('path', editable=True)
#     gridOptions = gb.build()
#     grid_response = AgGrid(
#         df,
#         gridOptions=gridOptions,
#         update_mode=GridUpdateMode.SELECTION_CHANGED,
#         fit_columns_on_grid_load=False,
#         height=None,
#         columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
#         allow_unsafe_jscode=True,
#         editable=True,
#     )

def chart(df):
    feature_list = list(df.columns)
    plot_list = ['Line', 'Scatter']
    mode = st.radio(
        label='',
        options=['Random', 'Manual'],
        horizontal=True,
    )
    if mode == 'Manual':
        feature_options = st.multiselect(
            label='Features', 
            options=feature_list, 
            max_selections=2)
        
    elif mode == 'Random':
        feature_random = random.sample(feature_list, k=2)
        feature_options = st.multiselect(
            label='Features', 
            options=feature_list, 
            default=feature_random, 
            max_selections=2)
        # plot_type = random.sample(plot_list, k=1)
    plot_type = st.selectbox(label='Plot Type', options=plot_list)
    if len(feature_options) == 2:
        on = st.toggle('Switch axis')
        if plot_type == 'Line':
            fig = px.line(
                data_frame=df, 
                x=feature_options[0],
                y=feature_options[1]
            )
            if on:
                fig = px.line(
                    data_frame=df, 
                    x=feature_options[1],
                    y=feature_options[0]
                )
        elif plot_type == 'Scatter':
            fig = px.scatter(
                data_frame=df, 
                x=feature_options[0],
                y=feature_options[1]
            )
            if on:
                fig = px.scatter(
                    data_frame=df, 
                    x=feature_options[1],
                    y=feature_options[0]
                )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    

# # Function to display the model selection and file upload interface
def display(state):
    # Load configuration from JSON file
    with open('config.json') as json_file:
        file = json.load(json_file)

    # Extract default and custom model lists
    default_list = list(file['model']['File Path']['default'].keys())
    custom_list = list(file['model']['File Path']['custom'].keys())
    model_list = default_list + custom_list

    # Select the model to use
    selected_model = st.selectbox(
        'Select model', model_list)
    state.selected_model = selected_model

    # Determine the model path based on the selected model
    if selected_model in default_list:
        model_path = file['model']['File Path']['default'][selected_model]['value']
    elif selected_model in custom_list:
        model_path = file['model']['File Path']['custom'][selected_model]['value']
    state.model_path = model_path

    # Create a container for the user interface
    with st.container():
        with st.expander('File Upload'):
            # Allow users to upload a file
            uploaded_file = st.file_uploader("Choose a file", type="csv")
        if uploaded_file is not None:
            # Process the uploaded file and display results
            df = pd.read_csv(uploaded_file)

            with st.expander('Data Overview'):
                data_tab, chart_tab = st.tabs(['Data', 'Chart'])
                with data_tab:
                    st.write(df)
                with chart_tab:
                    chart(df)

            with st.spinner('Processing...'):
                result = make_path_prediction(df, model_path)

# # Function to handle custom datasets and display information
# def custom_dataset(uploaded_file):
#     # Read a CSV file and extract file paths
#     file_extension = uploaded_file.name.split('.')[1]
#     if file_extension == 'csv':
#         data = pd.read_csv(uploaded_file)
#         path_list = data.iloc[:, 0].values.tolist()

#         # Extract features from file paths
#         # with st.spinner('Extracting...'):
#         #     feature_list = extract_feature(path_list)

#         # Create a DataFrame with features and labels
#         # vect_df = pd.DataFrame(np.stack(feature_list))
#         # vect_df['Label'] = data.iloc[:, 1]
#         # vect_df.to_csv(
#             # 'Dataset/File Path/custom_training.csv', index=False)

#         return data

# Function to handle custom model training and display options
# def custom_model(state):
#     # Select the type of machine learning model
#     model_type = st.selectbox(label='Model Type', options=[
#         'Random Forest', 'Decision Tree'])

#     # Select the dataset to use for training
#     dataset = st.selectbox(label='Dataset', options=[
#         'Default Dataset', 'Custom Dataset'])

#     # Handle custom datasets
#     if dataset == 'Custom Dataset':
#         uploaded_file = st.file_uploader("Upload a dataset", type=['csv'])
#         if uploaded_file is not None:
#             data, vect_df = custom_dataset(uploaded_file)

#             # Display the first 5 rows of the dataset
#             st.write("The first 5 rows of data")
#             AgGrid(data.head(5))

#             # Display the feature DataFrame for the first 5 rows
#             st.write("Feature DataFrame for the first 5 rows")
#             vect_cols = [f'd_{i}' for i in range(len(vect_df.columns) - 1)]
#             vect_cols.append('Label')
#             vect_df.columns = vect_cols
#             AgGrid(vect_df.head())
#         else:
#             st.warning('You need to upload a csv file.')

#     # Set options for model training
#     options = {
#         'type': 'file_path'
#     }
#     options['dataset'] = 'default' if dataset == 'Default Dataset' else 'custom'
#     with st.form(key='custom'):
#         with open('config.json') as json_file:
#             file = json.load(json_file)

#         # Configure options based on the selected model type
#         if model_type == 'Random Forest':
#             options['model'] = 'rf'
#             params = file['model config']['random forest']['config']
#             custom_params = {}
#             custom_params['n_estimators'] = st.slider(label='n_estimators', value=params['n_estimators'],
#                                                       min_value=1, max_value=1000, step=1)
#             # ... (similar configuration for other parameters)
#             options['config'] = custom_params
#         elif model_type == 'Decision Tree':
#             options['model'] = 'dt'
#             params = file['model config']['decision tree']['config']
#             custom_params = {}
#             custom_params['max_depth'] = st.slider(label='max_depth', value=params['max_depth'],
#                                                    min_value=1, max_value=100, step=1)
#             # ... (similar configuration for other parameters)
#             options['config'] = custom_params

#         # Get model labels and configure additional options
#         labels = list(file['model']['URL']['custom'].keys()) + \
#             list(file['model']['URL']['default'].keys())
#         model_name = st.text_input(label='Model Name')
#         save_model = st.radio('Save Model?', ('Yes', 'No'), horizontal=True)
#         options['save'] = 1 if save_model == 'Yes' else 0
#         show_results = st.radio(
#             'Show Results?', ('Yes', 'No'), horizontal=True)
#         options['mode'] = 'testing' if show_results == 'Yes' else 'training'
#         submitted = st.form_submit_button("Training")
#         if submitted:
#             # Validate model name and initiate training
#             if model_name == '':
#                 st.warning('You must assign a name!')
#             elif model_name in labels:
#                 st.warning('Name already exists!')
#             else:
#                 options['name'] = model_name
#                 with st.spinner('Processing...'):
#                     Model.train_model(options)
#                 st.success('Done!')

# Function to write the main content of the Streamlit app
def write(state):
    st.write("# APT Attack Detection")
    display(state)

# Entry point for the Streamlit app
