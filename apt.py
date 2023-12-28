import subprocess
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import json
import random
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

def highlight_row(s):
    return ['background-color: #ebf9ee']*len(s) if s.Label == s.Prediction else ['background-color: #ffeded']*len(s)

def make_path_prediction(df, model_path, uploaded_file):
    label = ["BENIGN", "APT"]
    model = joblib.load(model_path)
    if uploaded_file.name.split('.')[-1] == 'csv':
        predictions = model.predict(df.iloc[:, 8:-1])
    else:
        predictions = model.predict(df.iloc[:, 8:])
    df['Prediction'] = predictions
    df['Prediction'] = df['Prediction'].apply(lambda x: label[x])
    
    with st.expander('Prediction Result'):
        if uploaded_file.name.split('.')[-1] == 'csv':
            df = df.iloc[:, [0, 1, -2, -1]]
            st.dataframe(df.style.apply(highlight_row, axis=1))
        else:
            df = df.iloc[:, [0, 1, -1]]
            st.dataframe(df)

def chart(df):
    feature_list = list(df.columns)
    feature_options = []
    plot_list = ['Line', 'Scatter', 'Bar', 'Histogram', 'HeatMaps']
    mode = st.radio(
        label='Mode',
        options=['Random', 'Manual', 'Automatic'],
        horizontal=True,
    )
    if mode == 'Manual':
        feature_options = st.multiselect(
            label='Features', 
            options=feature_list, 
            max_selections=2)
        plot_type = st.selectbox(label='Plot Type', options=plot_list)

    elif mode == 'Random':
        feature_random = random.sample(feature_list, k=2)
        feature_options = st.multiselect(
            label='Features', 
            options=feature_list, 
            default=feature_random, 
            max_selections=2)
        plot_type = random.sample(plot_list, k=1)[0]

    elif mode == 'Automatic':
        feature = st.selectbox(
            label='Features', 
            options=feature_list,
        )
        remaining_features = list(df.drop(columns=[feature]).columns)
        tab_selection = st.selectbox(label='Tabs', options=[f'Feature {i * 10 + 1} -> {min((i + 1) * 10, int(len(remaining_features)))}' for i in range(int(len(remaining_features) / 10) + 1)])
        idx = int(tab_selection.split()[1])
        tabs = st.tabs(remaining_features[(idx - 1):idx + 9])
        for i in range(len(tabs)):
            with tabs[i]:
                plot_tab = st.tabs(['Scatter', 'Bar', 'Histogram', 'Heatmap', 'Line'])
                with plot_tab[0]:
                    on = st.toggle('Switch axis', key=f'{i}_{plot_tab}')
                    if not on:
                        fig = px.line(
                            data_frame=df, 
                            x=feature,
                            y=remaining_features[(idx - 1) + i]
                        )
                    else:
                        fig = px.line(
                            data_frame=df, 
                            x=remaining_features[(idx - 1) + i],
                            y=feature
                        )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with plot_tab[1]:
                    on = st.toggle('Switch axis', key=f'{i}_{plot_tab}')
                    if not on:
                        fig = px.scatter(
                            data_frame=df, 
                            x=feature,
                            y=remaining_features[(idx - 1) + i]
                        )
                    else:
                        fig = px.scatter(
                            data_frame=df, 
                            x=remaining_features[(idx - 1) + i],
                            y=feature
                        )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with plot_tab[2]:
                    on = st.toggle('Switch axis', key=f'{i}_{plot_tab}')
                    if not on:
                        fig = px.bar(
                            data_frame=df, 
                            x=feature,
                            y=remaining_features[(idx - 1) + i]
                        )
                    else:
                        fig = px.bar(
                            data_frame=df, 
                            x=remaining_features[(idx - 1) + i],
                            y=feature
                        )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with plot_tab[3]:
                    on = st.toggle('Switch axis', key=f'{i}_{plot_tab}')
                    if not on:
                        fig = px.histogram(
                            data_frame=df, 
                            x=feature,
                            y=remaining_features[(idx - 1) + i]
                        )
                    else:
                        fig = px.histogram(
                            data_frame=df, 
                            x=remaining_features[(idx - 1) + i],
                            y=feature
                        )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with plot_tab[4]:
                    on = st.toggle('Switch axis', key=f'{i}_{plot_tab}')
                    if not on:
                        fig = px.density_heatmap(
                            data_frame=df, 
                            x=feature,
                            y=remaining_features[(idx - 1) + i]
                        )
                    else:
                        fig = px.density_heatmap(
                            data_frame=df, 
                            x=remaining_features[(idx - 1) + i],
                            y=feature
                        )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

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
        elif plot_type == 'Bar':
            fig = px.bar(
                data_frame=df, 
                x=feature_options[0],
                y=feature_options[1]
            )
            if on:
                fig = px.bar(
                    data_frame=df, 
                    x=feature_options[1],
                    y=feature_options[0]
                )
        elif plot_type == 'Histogram':
            fig = px.histogram(
                data_frame=df, 
                x=feature_options[0],
                y=feature_options[1]
            )
            if on:
                fig = px.histogram(
                    data_frame=df, 
                    x=feature_options[1],
                    y=feature_options[0]
                )
        elif plot_type == 'HeatMaps':
            fig = px.density_heatmap(
                data_frame=df, 
                x=feature_options[0],
                y=feature_options[1]
            )
            if on:
                fig = px.density_heatmap(
                    data_frame=df, 
                    x=feature_options[1],
                    y=feature_options[0]
            )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def display(state):
    with open('config.json') as json_file:
        file = json.load(json_file)

    default_list = list(file['model']['File Path']['default'].keys())
    custom_list = list(file['model']['File Path']['custom'].keys())
    model_list = default_list + custom_list

    selected_model = st.selectbox(
        'Select model', model_list)
    state.selected_model = selected_model

    if selected_model in default_list:
        model_path = file['model']['File Path']['default'][selected_model]['value']
    elif selected_model in custom_list:
        model_path = file['model']['File Path']['custom'][selected_model]['value']
    state.model_path = model_path

    with st.container():
        with st.expander('File Upload'):
            uploaded_file = st.file_uploader("Choose a file", type=["csv", 'pcap'])
        if uploaded_file is not None:
            if uploaded_file.name.split('.')[-1] == 'csv':
                df = pd.read_csv(uploaded_file)

            else:
                save_folder = 'UploadedFile'
                save_path = Path(save_folder, uploaded_file.name)
                csv_path = Path(save_folder, f"{uploaded_file.name.split('.')[0]}.csv")
                if not save_path.exists():
                    with open(save_path, mode='wb') as w:
                        w.write(uploaded_file.getvalue())
                    command = f"cicflowmeter -f {save_path} -c {csv_path}"

                    result = subprocess.run(command, shell=True, capture_output=True, text=True)

                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df.insert(0, 'publicIP', value=df['dst_ip'].values)
                    df.insert(1, 'flowID', value=0)
                    df['flowID'] = df.apply(lambda x: f"{x['src_ip']}-{x['publicIP']}-{x['src_port']}-{x['dst_port']}-{x['protocol']}", axis=1)
                    new_column_names = {
                        'src_ip': 'SrcIP',
                        'dst_ip': 'DstIP',
                        'src_port': 'SrcPort',
                        'dst_port': 'DstPort',
                        'protocol': 'Protocol',
                        'timestamp': 'Timestamp',
                        'flow_duration': 'FlowDuration',
                        'flow_byts_s': 'FlowByts/s',
                        'flow_pkts_s': 'FlowPkts/s',
                        'fwd_pkts_s': 'FwdPkts/s',
                        'bwd_pkts_s': 'BwdPkts/s',
                        'tot_fwd_pkts': 'TotFwdPkts',
                        'tot_bwd_pkts': 'TotBwdPkts',
                        'totlen_fwd_pkts': 'TotLenFwdPkts',
                        'totlen_bwd_pkts': 'TotLenBwdPkts',
                        'fwd_pkt_len_max': 'FwdPktLenMax',
                        'fwd_pkt_len_min': 'FwdPktLenMin',
                        'fwd_pkt_len_mean': 'FwdPktLenMean',
                        'fwd_pkt_len_std': 'FwdPktLenStd',
                        'bwd_pkt_len_max': 'BwdPktLenMax',
                        'bwd_pkt_len_min': 'BwdPktLenMin',
                        'bwd_pkt_len_mean': 'BwdPktLenMean',
                        'bwd_pkt_len_std': 'BwdPktLenStd',
                        'pkt_len_max': 'PktLenMax',
                        'pkt_len_min': 'PktLenMin',
                        'pkt_len_mean': 'PktLenMean',
                        'pkt_len_std': 'PktLenStd',
                        'pkt_len_var': 'PktLenVar',
                        'fwd_header_len': 'FwdHeaderLen',
                        'bwd_header_len': 'BwdHeaderLen',
                        'fwd_seg_size_min': 'FwdSegSizeMin',
                        'fwd_act_data_pkts': 'FwdActDataPkts',
                        'flow_iat_mean': 'FlowIATMean',
                        'flow_iat_max': 'FlowIATMax',
                        'flow_iat_min': 'FlowIATMin',
                        'flow_iat_std': 'FlowIATStd',
                        'fwd_iat_tot': 'FwdIATTot',
                        'fwd_iat_max': 'FwdIATMax',
                        'fwd_iat_min': 'FwdIATMin',
                        'fwd_iat_mean': 'FwdIATMean',
                        'fwd_iat_std': 'FwdIATStd',
                        'bwd_iat_tot': 'BwdIATTot',
                        'bwd_iat_max': 'BwdIATMax',
                        'bwd_iat_min': 'BwdIATMin',
                        'bwd_iat_mean': 'BwdIATMean',
                        'bwd_iat_std': 'BwdIATStd',
                        'fwd_psh_flags': 'FwdPSHFlags',
                        'bwd_psh_flags': 'BwdPSHFlags',
                        'fwd_urg_flags': 'FwdURGFlags',
                        'bwd_urg_flags': 'BwdURGFlags',
                        'fin_flag_cnt': 'FINFlagCnt',
                        'syn_flag_cnt': 'SYNFlagCnt',
                        'rst_flag_cnt': 'RSTFlagCnt',
                        'psh_flag_cnt': 'PSHFlagCnt',
                        'ack_flag_cnt': 'ACKFlagCnt',
                        'urg_flag_cnt': 'URGFlagCnt',
                        'ece_flag_cnt': 'ECEFlagCnt',
                        'down_up_ratio': 'Down/UpRatio',
                        'pkt_size_avg': 'PktSizeAvg',
                        'init_fwd_win_byts': 'InitFwdWinByts',
                        'init_bwd_win_byts': 'InitBwdWinByts',
                        'active_max': 'ActiveMax',
                        'active_min': 'ActiveMin',
                        'active_mean': 'ActiveMean',
                        'active_std': 'ActiveStd',
                        'idle_max': 'IdleMax',
                        'idle_min': 'IdleMin',
                        'idle_mean': 'IdleMean',
                        'idle_std': 'IdleStd',
                        'fwd_byts_b_avg': 'FwdByts/bAvg',
                        'fwd_pkts_b_avg': 'FwdPkts/bAvg',
                        'bwd_byts_b_avg': 'BwdByts/bAvg',
                        'bwd_pkts_b_avg': 'BwdPkts/bAvg',
                        'fwd_blk_rate_avg': 'FwdBlkRateAvg',
                        'bwd_blk_rate_avg': 'BwdBlkRateAvg',
                        'fwd_seg_size_avg': 'FwdSegSizeAvg',
                        'bwd_seg_size_avg': 'BwdSegSizeAvg',
                        'cwe_flag_count': 'CWEFlagCount',
                        'subflow_fwd_pkts': 'SubflowFwdPkts',
                        'subflow_bwd_pkts': 'SubflowBwdPkts',
                        'subflow_fwd_byts': 'SubflowFwdByts',
                        'subflow_bwd_byts': 'SubflowBwdByts',
                        'publicIP': 'publicIP',
                        'flowID': 'FlowID'
                    }
                    df.rename(columns=new_column_names, inplace=True)
                    order = ['publicIP', 'FlowID', 'SrcIP', 'SrcPort', 'DstIP', 'DstPort',
                            'Protocol', 'Timestamp', 'FlowDuration', 'TotFwdPkts', 'TotBwdPkts',
                            'TotLenFwdPkts', 'TotLenBwdPkts', 'FwdPktLenMax', 'FwdPktLenMin',
                            'FwdPktLenMean', 'FwdPktLenStd', 'BwdPktLenMax', 'BwdPktLenMin',
                            'BwdPktLenMean', 'BwdPktLenStd', 'FlowByts/s', 'FlowPkts/s',
                            'FlowIATMean', 'FlowIATStd', 'FlowIATMax', 'FlowIATMin', 'FwdIATTot',
                            'FwdIATMean', 'FwdIATStd', 'FwdIATMax', 'FwdIATMin', 'BwdIATTot',
                            'BwdIATMean', 'BwdIATStd', 'BwdIATMax', 'BwdIATMin', 'FwdPSHFlags',
                            'BwdPSHFlags', 'FwdURGFlags', 'BwdURGFlags', 'FwdHeaderLen',
                            'BwdHeaderLen', 'FwdPkts/s', 'BwdPkts/s', 'PktLenMin', 'PktLenMax',
                            'PktLenMean', 'PktLenStd', 'PktLenVar', 'FINFlagCnt', 'SYNFlagCnt',
                            'RSTFlagCnt', 'PSHFlagCnt', 'ACKFlagCnt', 'URGFlagCnt', 'CWEFlagCount',
                            'ECEFlagCnt', 'Down/UpRatio', 'PktSizeAvg', 'FwdSegSizeAvg',
                            'BwdSegSizeAvg', 'FwdByts/bAvg', 'FwdPkts/bAvg', 'FwdBlkRateAvg',
                            'BwdByts/bAvg', 'BwdPkts/bAvg', 'BwdBlkRateAvg', 'SubflowFwdPkts',
                            'SubflowFwdByts', 'SubflowBwdPkts', 'SubflowBwdByts', 'InitFwdWinByts',
                            'InitBwdWinByts', 'FwdActDataPkts', 'FwdSegSizeMin', 'ActiveMean',
                            'ActiveStd', 'ActiveMax', 'ActiveMin', 'IdleMean', 'IdleStd', 'IdleMax',
                            'IdleMin']
                    df = df[order]
                else:
                    st.write("Can't read file")

            with st.expander('Data Overview'):
                    data_tab, chart_tab = st.tabs(['Data', 'Chart'])
                    with data_tab:
                        st.write(df)
                    with chart_tab:
                        chart(df)

            with st.spinner('Processing...'):
                result = make_path_prediction(df, model_path, uploaded_file)

                

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
